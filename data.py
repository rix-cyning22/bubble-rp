import numpy as np
import torch
from tqdm.auto import tqdm
import multiprocessing
from functools import partial

from ode import simulate_states


def anchor_des(cfg):
    d = cfg.domain
    return np.concatenate([
        np.arange(d.de_lo, d.de_steep - cfg.data.anchor_step_low, cfg.data.anchor_step_low),
        np.arange(d.de_steep, d.de_hi + cfg.data.anchor_step_high, cfg.data.anchor_step_high),
    ])


def generate_anchor_data(cfg, progress=True):
    des = anchor_des(cfg)
    t_list, de_list, y_list = [], [], []

    workers = min(len(des), multiprocessing.cpu_count())
    simulate = partial(simulate_states, cfg=cfg)
    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.imap(simulate, des)
        bar = tqdm(zip(des, results), total=len(des), desc="Generating data",
                   disable=not progress)
        for de_val, (t_np, y_np) in bar:
            t_list.append(t_np)
            de_list.append(np.full_like(t_np, de_val))
            y_list.append(y_np)
            bar.set_postfix({'De': f"{de_val:.3f}"})
    return np.concatenate(t_list), np.concatenate(de_list), np.concatenate(y_list)


def anchor_tensors(cfg, device, progress=True):
    t, de, y = generate_anchor_data(cfg, progress=progress)
    as_col = lambda a: torch.tensor(a, dtype=torch.float32, device=device).view(-1, 1)
    return as_col(t), as_col(de), torch.tensor(y, dtype=torch.float32, device=device)


def sample_De(n, device, cfg):
    d = cfg.domain
    n_focus = n // 2
    uniform = torch.rand(n - n_focus, 1, device=device) * (d.de_hi - d.de_lo) + d.de_lo
    focus = torch.rand(n_focus, 1, device=device) * (d.de_steep - d.de_lo) + d.de_lo
    return torch.cat([uniform, focus], dim=0)
