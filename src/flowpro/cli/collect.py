from __future__ import annotations

import argparse
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
    p.add_argument("--record-rate-hz", type=float, default=10)
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument("--state-history-len", type=int, default=16)
    p.add_argument("--obs-history-len", type=int, default=9)
    p.add_argument("--sdk-root", default="")
    p.add_argument("--init-hdf5", default="")
    p.add_argument("--init-frame-idx", type=int, default=0)
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
    robot = FakeAstribotRobotIO()
    policy = WanVAPolicy(host=args.host, port=args.port, prompt=args.prompt,
                         replan_steps=args.replan_steps, fake=True)
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
        sdk_root=args.sdk_root or AstribotRuntimeConfig.sdk_root,
        init_hdf5=args.init_hdf5, init_frame_idx=args.init_frame_idx,
        left_xyz_low=args.left_xyz_low, left_xyz_high=args.left_xyz_high,
        right_xyz_low=args.right_xyz_low, right_xyz_high=args.right_xyz_high,
        right_min_z=args.right_min_z,
        state_history_len=args.state_history_len,
        obs_history_len=args.obs_history_len,
    )
    robot = AstribotRobotIO(runtime)
    policy = WanVAPolicy(host=args.host, port=args.port, prompt=args.prompt,
                         replan_steps=args.replan_steps, state_history_len=args.state_history_len,
                         obs_history_len=args.obs_history_len)
    controls = QuestControlSource(robot, state_url=args.quest_state_url,
                                  trigger_threshold=args.trigger_threshold)
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
    policy_period = 1.0 / max(args.policy_rate_hz, 1e-6)
    record_period = 1.0 / max(args.record_rate_hz, 1e-6)
    next_policy = next_record = time.monotonic()
    initial_pairs = sum(1 for path in PairStore(args.output).root.iterdir()
                        if path.is_dir() and (path / "metadata.json").exists())
    print("Collector active: B=rollback, hold middle=takeover, A=commit, Ctrl-C=stop", flush=True)
    try:
        while not stopped:
            started = time.monotonic()
            now = time.monotonic()
            control = controls.poll()
            should_tick = collector.phase is not Phase.POLICY or control.b or now >= next_policy
            if collector.phase in (Phase.ROLLED_BACK, Phase.TAKEOVER):
                control.record = now >= next_record
                if control.record:
                    next_record = now + record_period
            if should_tick:
                collector.tick(control)
                if collector.phase is Phase.POLICY:
                    next_policy = now + policy_period
            if args.target_pairs > 0:
                committed = sum(1 for path in PairStore(args.output).root.iterdir()
                                if path.is_dir() and (path / "metadata.json").exists())
                if committed - initial_pairs >= args.target_pairs:
                    print(f"Target reached: {args.target_pairs} new preference pairs", flush=True)
                    break
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        # Re-issue the measured pose so a filtered Cartesian controller holds
        # position if Quest disconnects, the operator interrupts, or a safety
        # check aborts collection.
        try:
            robot.execute(robot.state_action16())
        except Exception as exc:
            print(f"WARNING: failed to send final hold command: {exc}", flush=True)


if __name__ == "__main__":
    main()
