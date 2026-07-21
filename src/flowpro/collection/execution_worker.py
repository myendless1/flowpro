from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any

import numpy as np

from .protocol import RobotIO


@dataclass
class PolicyExecutionResult:
    generation: int
    first_in_chunk: bool = False
    last_in_chunk: bool = False
    actions: np.ndarray | None = None
    targets: np.ndarray | None = None
    start_targets: np.ndarray | None = None
    action_start_times: np.ndarray | None = None
    action_arrival_times: np.ndarray | None = None
    finished_at: float | None = None
    error: BaseException | None = None


class PolicyExecutionWorker:
    """Submit one waypoint batch without blocking Quest button polling."""

    def __init__(self, robot: RobotIO, *, batch_size: int = 8) -> None:
        self.robot = robot
        self.batch_size = int(batch_size)
        self._tasks: queue.Queue = queue.Queue()
        self._results: queue.Queue[PolicyExecutionResult] = queue.Queue()
        self._lock = threading.Lock()
        self._generation = 0
        self._pending = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="flowpro-policy-execution",
            daemon=True,
        )
        self._thread.start()

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending

    def request(
        self,
        actions: np.ndarray,
        *,
        first_in_chunk: bool,
        last_in_chunk: bool,
    ) -> bool:
        with self._lock:
            if self._closed or self._pending:
                return False
            generation = self._generation
            self._pending = True
        batch = np.asarray(actions, np.float32).reshape(-1, 16).copy()
        if not len(batch) or len(batch) > self.batch_size:
            with self._lock:
                self._pending = False
            raise ValueError(
                f"waypoint batch 必须包含 1 到 {self.batch_size} 个动作"
            )
        self._tasks.put(
            (
                "execute",
                generation,
                (batch, bool(first_in_chunk), bool(last_in_chunk)),
            )
        )
        return True

    def poll(self) -> PolicyExecutionResult | None:
        current = None
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                if result.generation != self._generation:
                    continue
                self._pending = False
            if result.error is not None:
                raise RuntimeError("策略 waypoint 执行线程失败") from result.error
            current = result
        return current

    def reset(self) -> None:
        with self._lock:
            if self._pending:
                raise RuntimeError("不能在 waypoint batch 执行期间重置执行线程")
            self._generation += 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._tasks.put(("stop", -1, None))
        # A requested shutdown follows the same rule as B: finish the active
        # waypoint batch before releasing the robot adapter.
        self._thread.join()

    def _run(self) -> None:
        while True:
            operation, generation, payload = self._tasks.get()
            try:
                if operation == "stop":
                    return
                try:
                    actions, first_in_chunk, last_in_chunk = payload
                    execute = getattr(self.robot, "execute_policy_waypoint_batch")
                    data = execute(
                        actions,
                        first_in_chunk=first_in_chunk,
                        last_in_chunk=last_in_chunk,
                    )
                    result = PolicyExecutionResult(
                        generation=generation,
                        first_in_chunk=first_in_chunk,
                        last_in_chunk=last_in_chunk,
                        actions=actions,
                        targets=np.asarray(data["targets"], np.float32),
                        start_targets=np.asarray(data["start_targets"], np.float32),
                        action_start_times=np.asarray(data["action_start_times"], np.float64),
                        action_arrival_times=np.asarray(data["action_arrival_times"], np.float64),
                        finished_at=float(data["finished_at"]),
                    )
                except BaseException as exc:
                    result = PolicyExecutionResult(generation=generation, error=exc)
                self._results.put(result)
            finally:
                self._tasks.task_done()


class ObservationSampler:
    """Keep timestamped observations available while waypoint calls are blocking."""

    def __init__(self, robot: RobotIO, *, rate_hz: float = 40.0, capacity: int = 512) -> None:
        self.robot = robot
        self.period_s = 1.0 / max(float(rate_hz), 1e-6)
        self._samples: deque[tuple[float, dict[str, Any]]] = deque(maxlen=int(capacity))
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self, *, clear: bool = False) -> None:
        with self._lock:
            if clear:
                self._samples.clear()
                self._error = None
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="flowpro-observation-sampler",
                daemon=True,
            )
            self._thread.start()

    def latest(self, *, wait_s: float = 0.0) -> dict[str, Any] | None:
        deadline = time.monotonic() + max(0.0, float(wait_s))
        with self._condition:
            while not self._samples and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._error is not None and not self._samples:
                raise RuntimeError("后台观测线程无法获取机器人观测") from self._error
            return None if not self._samples else self._samples[-1][1]

    def latest_at_or_before(self, timestamp: float) -> dict[str, Any]:
        with self._lock:
            if not self._samples:
                raise RuntimeError("后台观测缓冲区为空")
            selected_time = None
            selected = None
            for sample_time, sample in self._samples:
                if sample_time > timestamp:
                    break
                selected_time, selected = sample_time, sample
            if selected is None:
                raise RuntimeError(
                    "后台观测缓冲区没有 action 开始前的因果观测"
                )
        observation = dict(selected)
        timing = dict(observation.get("_flowpro_timing", {}))
        timing["action_alignment_timestamp"] = float(timestamp)
        timing["action_observation_offset_s"] = float(selected_time - timestamp)
        observation["_flowpro_timing"] = timing
        return observation

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.2, self.period_s * 2))

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                observation = self.robot.observe()
                timestamp = float(observation.get("time", time.time()))
                observation = dict(observation)
                observation["time"] = timestamp
                with self._condition:
                    self._samples.append((timestamp, observation))
                    self._error = None
                    self._condition.notify_all()
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
            self._stop.wait(max(0.0, self.period_s - (time.monotonic() - started)))
