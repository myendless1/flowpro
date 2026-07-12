from __future__ import annotations

from dataclasses import dataclass
import queue
import sys
import threading
from typing import Literal


EpisodeOutcome = Literal["success", "failure", "abort"]


@dataclass(frozen=True)
class ManualCommand:
    outcome: EpisodeOutcome
    raw: str


class StdinEpisodeController:
    """Background stdin reader for sparse real-robot episode labels."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled) and bool(getattr(sys.stdin, "isatty", lambda: False)())
        self._queue: queue.Queue[ManualCommand] = queue.Queue()
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        print(
            "[Astribot manual control] type 's' + Enter for success, 'f' for failure, 'a' for abort.",
            flush=True,
        )
        while True:
            try:
                raw = input().strip().lower()
            except EOFError:
                return
            except Exception:
                return
            if raw in {"s", "success", "1", "y", "yes"}:
                self._queue.put(ManualCommand("success", raw))
            elif raw in {"f", "failure", "fail", "0", "n", "no"}:
                self._queue.put(ManualCommand("failure", raw))
            elif raw in {"a", "abort", "q", "quit", "stop"}:
                self._queue.put(ManualCommand("abort", raw))

    def poll(self) -> ManualCommand | None:
        if not self.enabled:
            return None
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


def prompt_for_success(default: EpisodeOutcome = "failure") -> EpisodeOutcome:
    while True:
        try:
            raw = input("Episode ended. Success? [s/f/a] ").strip().lower()
        except EOFError:
            return default
        if raw in {"s", "success", "1", "y", "yes"}:
            return "success"
        if raw in {"f", "failure", "fail", "0", "n", "no"}:
            return "failure"
        if raw in {"a", "abort", "q", "quit", "stop"}:
            return "abort"
