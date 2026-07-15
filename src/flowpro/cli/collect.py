from __future__ import annotations

import argparse
import json
import signal
import time

import numpy as np

from flowpro.collection.astribot_runtime import (
    AstribotRobotIO,
    AstribotRuntimeConfig,
    FakeAstribotRobotIO,
    QuestControlSource,
    WanVAPolicy,
)
from flowpro.collection.controller import InputState, InterventionCollector, Phase
from flowpro.collection.rollback import RollbackConfig
from flowpro.data import PairStore
from astribot_env.initial_pose import normalize_init_joint_action


def _init_joint_action(value: str) -> list[list[float]]:
    try:
        return normalize_init_joint_action(json.loads(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(f"invalid --init-joint-action: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FlowPRO Astribot preference collector")
    p.add_argument("--output", required=True)
    p.add_argument("--quest-state-url", default="")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8006)
    p.add_argument("--prompt", default="perform the task")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--rollback-horizon", type=int, default=64)
    p.add_argument("--rollback-capacity", type=int, default=200)
    p.add_argument("--rollback-rate-hz", type=float, default=20)
    p.add_argument("--trigger-threshold", type=float, default=.5)
    p.add_argument("--control-rate-hz", type=float, default=10)
    p.add_argument("--policy-rate-hz", type=float, default=10)
    p.add_argument("--takeover-rate-hz", type=float, default=50)
    p.add_argument("--record-rate-hz", type=float, default=10)
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument("--state-history-len", type=int, default=16)
    p.add_argument("--obs-history-len", type=int, default=9)
    p.add_argument("--video-guidance-scale", type=float, default=1.0)
    p.add_argument("--action-guidance-scale", type=float, default=1.0)
    p.add_argument("--action-representation", choices=("absolute", "delta"), default="delta")
    p.add_argument("--max-translation-step-m", type=float, default=.06)
    p.add_argument("--takeover-max-translation-step-m", type=float, default=.01)
    p.add_argument("--takeover-max-rotation-step-deg", type=float, default=2.5)
    p.add_argument("--takeover-max-gripper-step", type=float, default=.02)
    p.add_argument("--gripper-trigger-threshold", type=float, default=.2)
    p.add_argument("--first-policy-waypoint-duration", type=float, default=.6)
    p.add_argument("--policy-waypoint-duration", type=float, default=.1)
    p.add_argument("--disable-policy-left-arm", action="store_true")
    p.add_argument("--sdk-root", default="")
    p.add_argument(
        "--init-joint-action",
        type=_init_joint_action,
        help="JSON six-group target: torso, left arm, left gripper, right arm, right gripper, head",
    )
    p.add_argument("--reset-prelift-height-m", type=float, default=.10)
    p.add_argument("--reset-prelift-duration", type=float, default=1.0)
    p.add_argument("--left-xyz-low", type=float, nargs=3)
    p.add_argument("--left-xyz-high", type=float, nargs=3)
    p.add_argument("--right-xyz-low", type=float, nargs=3)
    p.add_argument("--right-xyz-high", type=float, nargs=3)
    p.add_argument("--right-min-z", type=float, default=.862)
    p.add_argument("--fake", action="store_true", help="run a deterministic no-hardware collection smoke test")
    p.add_argument("--fake-pairs", type=int, default=1)
    p.add_argument("--target-pairs", type=int, default=0,
                   help="stop after this many committed pairs; 0 keeps collecting")
    return p


def _run_fake(args: argparse.Namespace) -> int:
    robot = FakeAstribotRobotIO(args.action_representation)
    policy = WanVAPolicy(host=args.host, port=args.port, prompt=args.prompt,
                         replan_steps=args.replan_steps, fake=True,
                         action_representation=args.action_representation)
    collector = InterventionCollector(
        robot, policy, PairStore(args.output), round_id=args.round,
        rollback=RollbackConfig(args.rollback_capacity, args.rollback_horizon, 0.0),
        trigger_threshold=args.trigger_threshold,
    )
    for pair_id in range(args.fake_pairs):
        for _ in range(max(2, args.rollback_horizon)):
            collector.tick(InputState())
        collector.tick(InputState(b=True))
        collector.tick(InputState())
        expert = robot.state_action16(); expert[0] += .001 * (pair_id + 1)
        collector.tick(InputState(middle=1, expert_action=expert))
        collector.tick(InputState(a=True))
    return args.fake_pairs


def _wait_for_a_reset(
    controls: QuestControlSource,
    stopped,
    period: float,
    episode: int,
) -> bool:
    print(
        f"[Episode {episode}] Press A to raise both arms, then move the robot "
        "to the initial pose.",
        flush=True,
    )
    released = False
    while not stopped():
        started = time.monotonic()
        control = controls.poll()
        if not control.a:
            released = True
        elif released:
            return True
        time.sleep(max(0.0, period - (time.monotonic() - started)))
    return False


def _wait_for_a_start(controls: QuestControlSource, stopped, period: float, episode: int) -> bool:
    print(f"[Episode {episode}] Prepare the scene, then press A to start policy inference.", flush=True)
    released = False
    while not stopped():
        started = time.monotonic()
        control = controls.poll()
        if not control.a:
            released = True
        elif released:
            return True
        time.sleep(max(0.0, period - (time.monotonic() - started)))
    return False


def _wait_for_chunk_decision(
    controls: QuestControlSource,
    stopped,
    period: float,
    *,
    long_press_seconds: float = 2.0,
) -> str:
    print(
        "Chunk complete: short press A=next chunk, hold A for 2s=finish episode, "
        "B=rollback this chunk and enter takeover.",
        flush=True,
    )
    buttons_released = False
    previous_b = False
    a_started: float | None = None
    while not stopped():
        started = time.monotonic()
        control = controls.poll()
        now = time.monotonic()
        if not buttons_released:
            buttons_released = not control.a and not control.b
        else:
            if control.b and not previous_b:
                return "rollback"
            if control.a:
                if a_started is None:
                    a_started = now
                elif now - a_started >= long_press_seconds:
                    return "finish"
            elif a_started is not None:
                return "continue"
        previous_b = control.b
        time.sleep(max(0.0, period - (time.monotonic() - started)))
    return "stop"


def _gate_takeover_retry_b(control: InputState, armed: bool) -> bool:
    """Ignore the rollback-triggering B hold until it has been released."""
    if armed:
        return True
    if not control.b:
        return True
    control.b = False
    return False


def main() -> None:
    args = _parser().parse_args()
    if args.fake:
        print(f"collected {_run_fake(args)} fake preference pair(s)")
        return
    if not args.quest_state_url:
        raise SystemExit("--quest-state-url is required unless --fake is used")
    if not args.prompt.strip():
        raise SystemExit("--prompt must describe the real robot task")

    runtime = AstribotRuntimeConfig(
        action_representation=args.action_representation,
        sdk_root=args.sdk_root or AstribotRuntimeConfig.sdk_root,
        sdk_frequency=args.takeover_rate_hz,
        init_joint_action=args.init_joint_action or AstribotRuntimeConfig().init_joint_action,
        reset_prelift_height_m=args.reset_prelift_height_m,
        reset_prelift_duration=args.reset_prelift_duration,
        reset_to_initial_on_startup=False,
        max_translation_step_m=args.max_translation_step_m,
        left_xyz_low=args.left_xyz_low, left_xyz_high=args.left_xyz_high,
        right_xyz_low=args.right_xyz_low, right_xyz_high=args.right_xyz_high,
        right_min_z=args.right_min_z,
        takeover_max_translation_step_m=args.takeover_max_translation_step_m,
        takeover_max_rotation_step_deg=args.takeover_max_rotation_step_deg,
        takeover_max_gripper_step=args.takeover_max_gripper_step,
        first_policy_waypoint_duration=args.first_policy_waypoint_duration,
        policy_waypoint_duration=args.policy_waypoint_duration,
        state_history_len=args.state_history_len,
        obs_history_len=args.obs_history_len,
    )
    robot = AstribotRobotIO(runtime)
    policy = WanVAPolicy(host=args.host, port=args.port, prompt=args.prompt,
                         replan_steps=args.replan_steps, state_history_len=args.state_history_len,
                         obs_history_len=args.obs_history_len,
                         control_left_arm=not args.disable_policy_left_arm,
                         video_guidance_scale=args.video_guidance_scale,
                         action_guidance_scale=args.action_guidance_scale,
                         action_representation=args.action_representation)
    controls = QuestControlSource(
        robot,
        state_url=args.quest_state_url,
        trigger_threshold=args.trigger_threshold,
        gripper_trigger_threshold=args.gripper_trigger_threshold,
    )
    collector = InterventionCollector(
        robot, policy, PairStore(args.output), round_id=args.round,
        rollback=RollbackConfig(args.rollback_capacity, args.rollback_horizon,
                                  1.0 / max(args.rollback_rate_hz, 1e-6)),
        trigger_threshold=args.trigger_threshold,
    )
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    period = 1.0 / max(args.control_rate_hz, 1e-6)
    takeover_period = 1.0 / max(args.takeover_rate_hz, 1e-6)
    record_period = 1.0 / max(args.record_rate_hz, 1e-6)
    initial_pairs = sum(1 for path in PairStore(args.output).root.iterdir()
                        if path.is_dir() and (path / "metadata.json").exists())
    committed = initial_pairs
    episode = 1
    print(
        "Collector ready: A=reset/start/next/finish, B=rollback/retry, "
        "hold A 2s=finish, middle=takeover, Ctrl-C=stop",
        flush=True,
    )
    try:
        while not stopped:
            controls.reset()
            if not _wait_for_a_reset(
                controls, lambda: stopped, period, episode
            ):
                break
            print(f"[Episode {episode}] Moving to the initial pose...", flush=True)
            robot.move_to_initial_pose()
            print(f"[Episode {episode}] Initial pose reached.", flush=True)

            controls.reset()
            collector.start_episode()
            if not _wait_for_a_start(controls, lambda: stopped, period, episode):
                break

            episode_complete = False
            pair_saved = False
            retry_episode = False
            while not stopped and not episode_complete:
                print(f"[Episode {episode}] Executing one policy action chunk...", flush=True)
                collector.tick(InputState())
                decision = _wait_for_chunk_decision(controls, lambda: stopped, period)
                if decision == "stop":
                    break
                if decision == "continue":
                    print(f"[Episode {episode}] A short press: continuing to the next chunk.", flush=True)
                    continue
                if decision == "finish":
                    print(f"[Episode {episode}] A held for 2s: episode finished.", flush=True)
                    episode_complete = True
                    continue

                print(f"[Episode {episode}] B pressed: rolling back the completed chunk.", flush=True)
                collector.tick(InputState(b=True))
                collector.tick(InputState())
                print(
                    f"[Episode {episode}] Takeover active: hold middle to control and record; "
                    "press A to save and finish; press B to discard and retry.",
                    flush=True,
                )
                next_takeover = next_record = time.monotonic()
                retry_b_armed = False
                while not stopped and not episode_complete:
                    started = time.monotonic()
                    now = time.monotonic()
                    control = controls.poll()
                    retry_b_armed = _gate_takeover_retry_b(control, retry_b_armed)
                    control.record = now >= next_record
                    if control.record:
                        next_record = now + record_period
                    if now >= next_takeover:
                        previous_phase = collector.phase
                        collector.tick(control)
                        if (
                            control.b
                            and previous_phase in (Phase.ROLLED_BACK, Phase.TAKEOVER)
                            and collector.phase is Phase.POLICY
                        ):
                            retry_episode = bool(
                                getattr(collector, "last_pair_discarded", False)
                            )
                            episode_complete = True
                        elif (
                            control.a
                            and previous_phase in (Phase.ROLLED_BACK, Phase.TAKEOVER)
                            and collector.phase is Phase.POLICY
                        ):
                            pair_saved = bool(getattr(collector, "last_pair_saved", False))
                            episode_complete = True
                        next_takeover = now + takeover_period
                    time.sleep(max(0.0, period - (time.monotonic() - started)))

                if retry_episode:
                    print(
                        f"[Episode {episode}] Pair discarded; restarting this episode.",
                        flush=True,
                    )
                elif pair_saved:
                    print(f"[Episode {episode}] Takeover pair saved; episode finished.", flush=True)
                elif episode_complete:
                    print(
                        f"[Episode {episode}] No correction recorded; episode finished without saving a pair.",
                        flush=True,
                    )
                if episode_complete:
                    break

            if stopped:
                break
            if retry_episode:
                continue
            if pair_saved:
                committed += 1
                if args.target_pairs > 0 and committed - initial_pairs >= args.target_pairs:
                    print(f"Target reached: {args.target_pairs} new preference pairs", flush=True)
                    break
            episode += 1
    finally:
        # Re-issue the measured pose so a filtered Cartesian controller holds
        # position if Quest disconnects, the operator interrupts, or a safety
        # check aborts collection.
        try:
            state = robot.state_action16()
            robot.execute_absolute(state)
        except Exception as exc:
            print(f"WARNING: failed to send final hold command: {exc}", flush=True)


if __name__ == "__main__":
    main()
