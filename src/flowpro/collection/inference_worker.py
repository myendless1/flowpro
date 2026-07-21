from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
from typing import Any

import numpy as np

from .protocol import Policy


@dataclass
class InferenceResult:
    generation: int
    chunk: np.ndarray | None = None
    error: BaseException | None = None


class PolicyInferenceWorker:
    """Run policy reset/inference serially without blocking the control loop."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self._tasks: queue.Queue = queue.Queue()
        self._results: queue.Queue[InferenceResult] = queue.Queue()
        self._lock = threading.Lock()
        self._generation = 0
        self._pending = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="flowpro-policy-inference",
            daemon=True,
        )
        self._thread.start()

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending

    def reset(self, observation: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._pending = False
        self._tasks.put(("reset", generation, observation))

    def request(self, observation: dict[str, Any]) -> bool:
        with self._lock:
            if self._closed or self._pending:
                return False
            generation = self._generation
            self._pending = True
        self._tasks.put(("infer", generation, observation))
        return True

    def poll(self) -> np.ndarray | None:
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
                raise RuntimeError("策略推理线程执行失败") from result.error
            if result.chunk is not None:
                current = result.chunk
        return current

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._pending = False
        self._tasks.put(("stop", -1, None))
        self._thread.join(timeout=0.2)

    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return not self._closed and generation == self._generation

    def _run(self) -> None:
        while True:
            operation, generation, payload = self._tasks.get()
            try:
                if operation == "stop":
                    return
                if not self._is_current(generation):
                    continue
                if operation == "reset":
                    try:
                        self.policy.reset(payload)
                    except BaseException as exc:
                        self._results.put(InferenceResult(generation, error=exc))
                    continue
                try:
                    chunk = np.asarray(self.policy.infer(payload), dtype=np.float32).reshape(-1, 16)
                    result = InferenceResult(generation, chunk=chunk)
                except BaseException as exc:
                    result = InferenceResult(generation, error=exc)
                self._results.put(result)
            finally:
                self._tasks.task_done()
