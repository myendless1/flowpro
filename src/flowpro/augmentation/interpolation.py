from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from flowpro.data.types import Frame, PreferenceSample, TrajectoryPair
from wan_va.action_representation import apply_relative_pose7


@dataclass
class InterpolationConfig:
    horizon: int = 16
    position_weight: float = 1.0
    rotation_weight: float = 0.5
    gripper_weight: float = 0.2
    bridge_fraction: float = 0.5
    max_position_step: float = 0.04
    tangent_scale: float = 1.0


def _normalize(q):
    q = np.asarray(q, np.float32); n = np.linalg.norm(q)
    return q / max(float(n), 1e-8)


def _slerp(a, b, t):
    a, b = _normalize(a), _normalize(b); dot = float(np.dot(a, b))
    if dot < 0: b, dot = -b, -dot
    if dot > .9995: return _normalize(a + t * (b-a))
    theta = np.arccos(np.clip(dot, -1, 1)); s = np.sin(theta)
    return np.sin((1-t)*theta)/s*a + np.sin(t*theta)/s*b


def _distance(a, b, cfg):
    total = 0.0
    for off in (0, 8):
        total += cfg.position_weight * np.linalg.norm(a[off:off+3] - b[off:off+3])
        dot = abs(float(np.dot(_normalize(a[off+3:off+7]), _normalize(b[off+3:off+7]))))
        total += cfg.rotation_weight * (2 * np.arccos(np.clip(dot, 0, 1)))
        total += cfg.gripper_weight * abs(float(a[off+7] - b[off+7]))
    return total


def _pad_chunk(frames: list[Frame], start: int, horizon: int) -> np.ndarray:
    values = [_frame_target(frames[min(i, len(frames)-1)]) for i in range(start, start+horizon)]
    return np.stack(values).astype(np.float32)


def _frame_state(frame: Frame) -> np.ndarray:
    state = frame.observation.get("state_action16") if isinstance(frame.observation, dict) else None
    if state is None and isinstance(frame.observation, dict):
        history = frame.observation.get("wam4d", {}).get("observation.state", [])
        if len(history):
            state = history[-1]
    return np.asarray(frame.action if state is None else state, np.float32).reshape(16)


def _frame_target(frame: Frame) -> np.ndarray:
    """Recover absolute geometry from the canonical stored frame delta."""
    state = _frame_state(frame)
    delta = np.asarray(frame.action, np.float32).reshape(16)
    target = state.copy()
    target[0:7] = apply_relative_pose7(state[0:7], delta[0:7])
    target[7] = delta[7]
    target[8:15] = apply_relative_pose7(state[8:15], delta[8:15])
    target[15] = delta[15]
    return target


def _bridge(start: np.ndarray, target: np.ndarray, arrival_next: np.ndarray, cfg: InterpolationConfig):
    """Cubic Bezier position, quaternion Slerp, linear gripper interpolation."""
    h = cfg.horizon; out = np.empty((h, 16), np.float32)
    bridge_n = max(1, min(h, round(h * cfg.bridge_fraction)))
    for i in range(h):
        if i >= bridge_n:
            out[i] = target[min(i, len(target)-1)]; continue
        u = (i + 1) / bridge_n
        dst = target[min(bridge_n-1, len(target)-1)]
        for off in (0, 8):
            p0, p3 = start[off:off+3], dst[off:off+3]
            tangent = arrival_next[off:off+3] - p3
            # Appendix D: the first control point is the midpoint.  This avoids
            # inheriting the erroneous loser tangent while keeping the bridge
            # inside the source/target lens.
            p1 = (p0 + p3) * .5
            p2 = p3 - cfg.tangent_scale * tangent * bridge_n / 3
            p = (1-u)**3*p0 + 3*(1-u)**2*u*p1 + 3*(1-u)*u*u*p2 + u**3*p3
            delta = p - (start[off:off+3] if i == 0 else out[i-1, off:off+3])
            norm = np.linalg.norm(delta)
            if norm > cfg.max_position_step: p -= delta * (1 - cfg.max_position_step/norm)
            out[i, off:off+3] = p
            out[i, off+3:off+7] = _slerp(start[off+3:off+7], dst[off+3:off+7], u)
            out[i, off+7] = (1-u)*start[off+7] + u*dst[off+7]
    return out


def augment_pair(pair: TrajectoryPair, config: InterpolationConfig | None = None) -> list[PreferenceSample]:
    """Paper §3.4: dense tuples for both loser and winner states."""
    cfg = config or InterpolationConfig(); pair.validate(); result = []
    winner_states = np.stack([_frame_state(x) for x in pair.winner])
    # Appendix E excludes the potentially contact-rich tail: every sampled
    # loser state must have a complete (unpadded) H-step negative chunk.
    negative_count = max(0, len(pair.loser) - cfg.horizon + 1)
    for i, frame in enumerate(pair.loser[:negative_count]):
        state = _frame_state(frame)
        closest = min(range(len(pair.winner)), key=lambda j: _distance(state, winner_states[j], cfg))
        target = _pad_chunk(pair.winner, closest, cfg.horizon)
        bridge_n = max(1, min(cfg.horizon, round(cfg.horizon * cfg.bridge_fraction)))
        arrival = _frame_target(pair.winner[min(closest + bridge_n, len(pair.winner)-1)])
        result.append(PreferenceSample(frame.observation, _bridge(state, target, arrival, cfg),
                                       _pad_chunk(pair.loser, i, cfg.horizon), "negative", pair.pair_id))
    for i, frame in enumerate(pair.winner):
        chunk = _pad_chunk(pair.winner, i, cfg.horizon)
        result.append(PreferenceSample(frame.observation, chunk, chunk.copy(), "positive", pair.pair_id))
    return result
