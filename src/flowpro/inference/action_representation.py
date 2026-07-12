from __future__ import annotations
import numpy as np

EXECUTION_CHANNEL_IDS = tuple(list(range(7)) + [28] + list(range(7, 14)) + [29])


def normalize_quaternion_wxyz(q):
    q = np.asarray(q, np.float32); n = np.linalg.norm(q, axis=-1, keepdims=True)
    identity = np.zeros_like(q); identity[..., 0] = 1
    q = np.where(n > 1e-8, q/np.maximum(n, 1e-8), identity)
    return np.where(q[..., :1] < 0, -q, q).astype(np.float32)


def model30_to_execution16(actions):
    return np.asarray(actions, np.float32)[..., EXECUTION_CHANNEL_IDS]


def decode_execution_sequence(actions, *, initial_absolute):
    x = np.asarray(actions, np.float32).copy()
    previous = np.asarray(initial_absolute, np.float32).reshape(16); out = []
    for delta in x:
        cur = delta.copy()
        for off in (0, 8):
            cur[off:off+3] += previous[off:off+3]
            # Hamilton product previous * delta, wxyz
            w1,x1,y1,z1 = previous[off+3:off+7]; w2,x2,y2,z2 = delta[off+3:off+7]
            cur[off+3:off+7] = normalize_quaternion_wxyz([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
        out.append(cur); previous = cur
    return np.stack(out)
