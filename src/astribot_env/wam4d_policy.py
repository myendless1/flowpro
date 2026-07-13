from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from astribot_env.utils import ACTION16_DIM, actions_from_wam4d_response
from wan_va.action_representation import validate_action_representation


class WAM4DPriorClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        prompt: str,
        state_history_len: int = 16,
        obs_history_len: int = 9,
        num_action_groups: int = 2,
        save_visualization: bool = False,
        video_guidance_scale: float = 5.0,
        action_guidance_scale: float = 5.0,
        action_representation: str = "delta",
        fake: bool = False,
    ) -> None:
        self.prompt = prompt
        self.state_history_len = int(state_history_len)
        self.obs_history = deque(maxlen=int(obs_history_len))
        self.action_history: deque[np.ndarray] = deque(maxlen=self.state_history_len)
        self.num_action_groups = int(num_action_groups)
        self.save_visualization = bool(save_visualization)
        self.video_guidance_scale = float(video_guidance_scale)
        self.action_guidance_scale = float(action_guidance_scale)
        self.action_representation = validate_action_representation(action_representation)
        self.fake = bool(fake)
        if self.fake:
            self.policy = None
        else:
            # Keep deployment independent from the reference repository.  The
            # client shipped with wan_va speaks the exact same msgpack protocol
            # as VA_Server.
            from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import (
                WebsocketClientPolicy,
            )

            self.policy = WebsocketClientPolicy(host=host, port=port)
            metadata = self.policy.get_server_metadata()
            server_representation = metadata.get("action_representation")
            if server_representation != self.action_representation:
                raise RuntimeError(
                    "WAM4D action representation mismatch: "
                    f"collector={self.action_representation!r}, "
                    f"server={server_representation!r}"
                )
            if metadata.get("state_action_representation") != "absolute":
                raise RuntimeError(
                    "WAM4D server must declare absolute state/action history"
                )
        self.last_raw_action: np.ndarray | None = None

    def reset(self) -> None:
        self.obs_history.clear()
        self.action_history.clear()
        self.last_raw_action = None
        if self.policy is not None:
            self.policy.reset()

    def append_executed_action(self, action16: np.ndarray) -> None:
        self.action_history.append(np.asarray(action16, dtype=np.float32).reshape(ACTION16_DIM).copy())

    def append_observation(self, obs_payload: dict[str, Any]) -> None:
        self.obs_history.append(obs_payload)

    def infer_prior_action(self, obs_payload: dict[str, Any], *, fallback_state16: np.ndarray | None = None) -> np.ndarray:
        return self.infer_prior_chunk(obs_payload, fallback_state16=fallback_state16, max_steps=1)[0]

    def infer_prior_chunk(
        self,
        obs_payload: dict[str, Any],
        *,
        fallback_state16: np.ndarray | None = None,
        max_steps: int | None = None,
    ) -> np.ndarray:
        if self.fake:
            if self.action_representation == "absolute" and fallback_state16 is not None:
                action = np.asarray(fallback_state16, dtype=np.float32).reshape(ACTION16_DIM).copy()
            else:
                action = np.zeros((ACTION16_DIM,), dtype=np.float32)
                action[[3, 11]] = 1.0
                if fallback_state16 is not None:
                    state = np.asarray(fallback_state16, dtype=np.float32).reshape(ACTION16_DIM)
                    action[[7, 15]] = state[[7, 15]]
            steps = max(1, int(max_steps or 1))
            return np.repeat(action.reshape(1, ACTION16_DIM), steps, axis=0)

        self.append_observation(obs_payload)
        assert self.policy is not None
        ret = self.policy.infer(
            {
                "obs": list(self.obs_history),
                "infer_action": True,
                "num_action_groups": self.num_action_groups,
                "prompt": self.prompt,
                "save_visualization": self.save_visualization,
                "video_guidance_scale": self.video_guidance_scale,
                "action_guidance_scale": self.action_guidance_scale,
            }
        )
        if "action" not in ret:
            raise RuntimeError(f"WAM4D server response missing 'action': keys={list(ret.keys())}")
        self.last_raw_action = np.asarray(ret["action"], dtype=np.float32)
        return actions_from_wam4d_response(self.last_raw_action, max_steps=max_steps)
