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
    p = argparse.ArgumentParser(description="FlowPRO Astribot 偏好数据采集器")
    p.add_argument("--output", required=True)
    p.add_argument("--quest-state-url", default="")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8006)
    p.add_argument("--prompt", default="perform the task")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--rollback-horizon", type=int, default=64)
    p.add_argument("--rollback-capacity", type=int, default=72)
    p.add_argument("--rollback-rate-hz", type=float, default=20)
    p.add_argument("--trigger-threshold", type=float, default=.5)
    p.add_argument("--control-rate-hz", type=float, default=10)
    p.add_argument("--policy-rate-hz", type=float, default=10)
    p.add_argument("--takeover-rate-hz", type=float, default=50)
    p.add_argument("--record-rate-hz", type=float, default=10)
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument("--state-history-len", type=int, default=16)
    p.add_argument("--obs-history-len", type=int, default=9)
    p.add_argument("--image-from-s1-topic", dest="image_from_s1_topic", action="store_true")
    p.add_argument("--sdk-image-polling", dest="image_from_s1_topic", action="store_false")
    p.set_defaults(image_from_s1_topic=True)
    p.add_argument("--camera-sync-slop-s", type=float, default=.05)
    p.add_argument("--camera-sync-rate-hz", type=float, default=40.0)
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
    p.add_argument("--policy-waypoint-batch-actions", type=int, default=8)
    p.add_argument("--disable-policy-left-arm", action="store_true")
    p.add_argument("--sdk-root", default="")
    p.add_argument(
        "--init-joint-action",
        type=_init_joint_action,
        help="JSON 六组目标：躯干、左臂、左夹爪、右臂、右夹爪、头部",
    )
    p.add_argument("--reset-prelift-height-m", type=float, default=.10)
    p.add_argument("--reset-prelift-duration", type=float, default=1.0)
    p.add_argument("--left-xyz-low", type=float, nargs=3)
    p.add_argument("--left-xyz-high", type=float, nargs=3)
    p.add_argument("--right-xyz-low", type=float, nargs=3)
    p.add_argument("--right-xyz-high", type=float, nargs=3)
    p.add_argument("--right-min-z", type=float, default=.862)
    p.add_argument(
        "--disable-right-gripper-angle-constraint-during-takeover",
        dest="right_gripper_angle_constraint_during_takeover",
        action="store_false",
    )
    p.set_defaults(right_gripper_angle_constraint_during_takeover=True)
    p.add_argument("--right-gripper-target-angle-deg", type=float, default=45.0)
    p.add_argument(
        "--right-gripper-ray-axis",
        choices=("+x", "-x", "+y", "-y", "+z", "-z"),
        default="+z",
    )
    p.add_argument(
        "--disable-right-gripper-twist-level-constraint",
        dest="right_gripper_twist_level_constraint",
        action="store_false",
    )
    p.set_defaults(right_gripper_twist_level_constraint=True)
    p.add_argument(
        "--right-gripper-level-axis",
        choices=("+x", "-x", "+y", "-y", "+z", "-z"),
        default="+x",
    )
    p.add_argument("--fake", action="store_true", help="运行无需硬件的确定性采集冒烟测试")
    p.add_argument("--fake-pairs", type=int, default=1)
    p.add_argument("--target-pairs", type=int, default=0,
                   help="目录内有效样本总数达到该值后停止；0 表示持续采集")
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
        f"[第 {episode} 轮] 按 A 抬起双臂，并将机器人移动到初始位姿。",
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
    print(f"[第 {episode} 轮] 请整理任务场景，准备完成后按 A 开始策略推理。", flush=True)
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


def _gate_takeover_retry_b(control: InputState, armed: bool) -> bool:
    """Ignore the rollback-triggering B hold until it has been released."""
    if armed:
        return True
    if not control.b:
        return True
    control.b = False
    return False


def _resume_progress(store: PairStore, target_pairs: int) -> tuple[int, int, bool]:
    completed = store.completed_count()
    next_episode = completed + 1
    target_reached = int(target_pairs) > 0 and completed >= int(target_pairs)
    return completed, next_episode, target_reached


def main() -> None:
    args = _parser().parse_args()
    if args.fake:
        print(f"已采集 {_run_fake(args)} 组模拟偏好样本")
        return
    if not args.quest_state_url:
        raise SystemExit("未使用 --fake 时必须提供 --quest-state-url")
    if not args.prompt.strip():
        raise SystemExit("--prompt 必须描述真实机器人任务")

    store = PairStore(args.output)
    initial_pairs, first_episode, target_reached = _resume_progress(store, args.target_pairs)
    if initial_pairs:
        print(
            f"检测到 {initial_pairs} 组已有有效样本，将从第 {initial_pairs + 1} 轮继续采集。",
            flush=True,
        )
    if target_reached:
        print(
            f"目录内已有 {initial_pairs} 组有效样本，已达到目标 {args.target_pairs} 组。",
            flush=True,
        )
        return

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
        right_gripper_angle_constraint_during_takeover=(
            args.right_gripper_angle_constraint_during_takeover
        ),
        right_gripper_target_angle_deg=args.right_gripper_target_angle_deg,
        right_gripper_ray_axis=args.right_gripper_ray_axis,
        right_gripper_twist_level_constraint=(
            args.right_gripper_twist_level_constraint
        ),
        right_gripper_level_axis=args.right_gripper_level_axis,
        takeover_max_translation_step_m=args.takeover_max_translation_step_m,
        takeover_max_rotation_step_deg=args.takeover_max_rotation_step_deg,
        takeover_max_gripper_step=args.takeover_max_gripper_step,
        first_policy_waypoint_duration=args.first_policy_waypoint_duration,
        policy_waypoint_duration=args.policy_waypoint_duration,
        state_history_len=args.state_history_len,
        obs_history_len=args.obs_history_len,
        image_from_s1_topic=args.image_from_s1_topic,
        camera_sync_slop_s=args.camera_sync_slop_s,
        camera_sync_rate_hz=args.camera_sync_rate_hz,
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
        robot, policy, store, round_id=args.round,
        rollback=RollbackConfig(args.rollback_capacity, args.rollback_horizon,
                                  1.0 / max(args.rollback_rate_hz, 1e-6)),
        trigger_threshold=args.trigger_threshold,
        async_inference=True,
        async_execution=True,
        observation_rate_hz=args.camera_sync_rate_hz,
        policy_waypoint_batch_actions=args.policy_waypoint_batch_actions,
    )
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    period = 1.0 / max(args.control_rate_hz, 1e-6)
    policy_period = 1.0 / max(args.policy_rate_hz, 1e-6)
    takeover_period = 1.0 / max(args.takeover_rate_hz, 1e-6)
    record_period = 1.0 / max(args.record_rate_hz, 1e-6)
    committed = initial_pairs
    episode = first_episode
    print(
        "采集器已就绪：A=复位/开始/接管后保存，B=回退/重试，"
        "middle=接管，Ctrl-C=停止",
        flush=True,
    )
    try:
        while not stopped:
            controls.reset()
            if not _wait_for_a_reset(
                controls, lambda: stopped, period, episode
            ):
                break
            print(f"[第 {episode} 轮] 正在移动到初始位姿...", flush=True)
            robot.move_to_initial_pose()
            print(f"[第 {episode} 轮] 已到达初始位姿。", flush=True)

            controls.reset()
            collector.start_episode()
            if not _wait_for_a_start(controls, lambda: stopped, period, episode):
                break

            episode_complete = False
            pair_saved = False
            retry_episode = False
            print(
                f"[第 {episode} 轮] 策略已启动，将自动连续推理和执行；"
                f"按 B 回退最近 {args.rollback_horizon} 步。",
                flush=True,
            )
            next_policy = time.monotonic()
            while not stopped and collector.phase in (Phase.POLICY, Phase.ARMED):
                started = time.monotonic()
                now = time.monotonic()
                control = controls.poll()
                control.policy_step = now >= next_policy
                if control.policy_step:
                    next_policy = now + policy_period
                collector.tick(control)
                if collector.last_policy_status == "inference_started":
                    print(f"[第 {episode} 轮] 正在推理下一个 action chunk...", flush=True)
                elif collector.last_policy_status == "chunk_started":
                    print(
                        f"[第 {episode} 轮] action chunk 已下发："
                        "开始执行第一个 waypoint batch。",
                        flush=True,
                    )
                elif collector.last_policy_status == "waypoint_batch_started":
                    print(f"[第 {episode} 轮] 下一个 waypoint batch 已下发。", flush=True)
                elif collector.last_policy_status == "waypoint_batch_finished":
                    print(f"[第 {episode} 轮] 当前 waypoint batch 已执行完成。", flush=True)
                elif collector.last_policy_status == "chunk_finished":
                    print(f"[第 {episode} 轮] 当前 chunk 已执行完成，将自动继续。", flush=True)
                elif collector.last_policy_status == "rollback_deferred":
                    print(
                        f"[第 {episode} 轮] 已收到 B：当前 waypoint batch 将继续执行完毕，"
                        "随后自动回退；B 后动作不会计入负样本。",
                        flush=True,
                    )
                time.sleep(max(0.0, period - (time.monotonic() - started)))

            if stopped:
                break
            if collector.phase is Phase.ROLLED_BACK:
                print(
                    f"[第 {episode} 轮] 已进入接管：按住 middle 控制并记录；"
                    "按 A 保存并结束；按 B 丢弃并重新采集本轮。",
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
                        f"[第 {episode} 轮] 已丢弃当前样本对，重新采集本轮。",
                        flush=True,
                    )
                elif pair_saved:
                    print(f"[第 {episode} 轮] 本轮结束。", flush=True)
                elif episode_complete:
                    print(
                        f"[第 {episode} 轮] 未记录纠正动作，本轮结束且不保存样本对。",
                        flush=True,
                    )
            if stopped:
                break
            if retry_episode:
                continue
            if pair_saved:
                committed += 1
                if args.target_pairs > 0 and committed >= args.target_pairs:
                    print(f"已达到目标：目录内共有 {committed} 组偏好样本", flush=True)
                    break
            episode += 1
    finally:
        collector.close()
        # Re-issue the measured pose so a filtered Cartesian controller holds
        # position if Quest disconnects, the operator interrupts, or a safety
        # check aborts collection.
        try:
            state = robot.state_action16()
            robot.execute_absolute(state)
        except Exception as exc:
            print(f"警告：发送最终保持位姿命令失败：{exc}", flush=True)


if __name__ == "__main__":
    main()
