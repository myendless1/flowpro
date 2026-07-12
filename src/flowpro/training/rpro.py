from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class RPROConfig:
    beta: float = 1.0
    lambda_pro: float = 1.0
    lambda_sft: float = 1.0


def rpro_loss(current_loss_w, current_loss_l, reference_loss_w, reference_loss_l,
              config: RPROConfig | None = None):
    """Exact FlowPRO Eq. (4-6), accepting per-example flow matching losses."""
    import torch
    import torch.nn.functional as F
    cfg = config or RPROConfig()
    rw = cfg.beta * .5 * (reference_loss_w.detach() - current_loss_w)
    rl = cfg.beta * .5 * (reference_loss_l.detach() - current_loss_l)
    contrastive = -F.logsigmoid(rw - rl)
    proximal = -.5 * (F.logsigmoid(rw) + F.logsigmoid(-rw)
                      + F.logsigmoid(rl) + F.logsigmoid(-rl))
    pro = (contrastive + proximal).mean()
    sft = current_loss_w.mean()
    return cfg.lambda_pro * pro + cfg.lambda_sft * sft, {
        "loss/pro": pro.detach(), "loss/sft": sft.detach(),
        "reward/winner": rw.mean().detach(), "reward/loser": rl.mean().detach(),
    }


def flow_matching_losses(model: Callable, state, actions, *, noise=None, times=None):
    """FlowPRO Eq. (3), returning one MSE scalar per batch element."""
    import torch
    b = actions.shape[0]
    noise = torch.randn_like(actions) if noise is None else noise
    times = torch.rand((b,), device=actions.device, dtype=actions.dtype) if times is None else times
    shape = (b,) + (1,) * (actions.ndim - 1)
    at = (1-times.reshape(shape)) * noise + times.reshape(shape) * actions
    target = actions - noise
    velocity = model(at, times, state)
    return (velocity - target).square().flatten(1).mean(1)

