import numpy as np
import torch
from tqdm.auto import tqdm

from ode import simulate_states


def anchor_des(cfg):
    d = cfg.domain
    return np.concatenate([
        np.arange(d.de_lo, d.de_steep, cfg.data.anchor_step_low),
        np.arange(d.de_steep, d.de_hi, cfg.data.anchor_step_high),
    ])


def generate_anchor_data(cfg, progress=True):
    des = anchor_des(cfg)
    t_list, de_list, y_list, amp_list = [], [], [], []

    bar = tqdm(des, desc="Generating anchor data", disable=not progress)
    for de_val in bar:
        t_np, y_np = simulate_states(de_val, cfg)
        t_list.append(t_np)
        de_list.append(np.full_like(t_np, de_val))
        y_list.append(y_np)
        amp_list.append(np.broadcast_to(
            np.maximum(y_np.std(axis=0), cfg.data.amp_floor), y_np.shape).copy())
        bar.set_postfix({'De': f"{de_val:.3f}"})
    bar.close()

    return (np.concatenate(t_list), np.concatenate(de_list),
            np.concatenate(y_list), np.concatenate(amp_list))


def anchor_tensors(cfg, device, progress=True):
    t, de, y, amp = generate_anchor_data(cfg, progress=progress)
    as_col = lambda a: torch.tensor(a, dtype=torch.float32, device=device).view(-1, 1)
    return (as_col(t), as_col(de),
            torch.tensor(y, dtype=torch.float32, device=device),
            torch.tensor(amp, dtype=torch.float32, device=device))


def sample_De(n, device, cfg):
    d = cfg.domain
    n_focus = n // 2
    uniform = torch.rand(n - n_focus, 1, device=device) * (d.de_hi - d.de_lo) + d.de_lo
    focus = torch.rand(n_focus, 1, device=device) * (d.de_steep - d.de_lo) + d.de_lo
    return torch.cat([uniform, focus], dim=0)
