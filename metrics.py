import numpy as np
import torch
import pandas as pd

from ode import simulate_states

CHANNELS = [
    (0, r'$R / R_0$', 'R'),
    (1, r'$\hat{v}$', 'v'),
    (2, r'$\hat{\tau}_{rr}$', 'tau_rr'),
    (3, r'$\hat{\tau}_{\theta\theta}$', 'tau_theta'),
]


def relative_l2(pred, true, channel=None, mask=None):
    if mask is not None:
        pred, true = pred[mask], true[mask]
    ref = true - 1.0 if channel == 0 else true
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(ref), 1e-12))


@torch.no_grad()
def predict(model, t_np, de_val, device):
    t = torch.tensor(t_np, dtype=torch.float32, device=device).view(-1, 1)
    return model(t, torch.full_like(t, float(de_val))).cpu().numpy()


def evaluate_at_de(model, cfg, device, de_val):
    was_training = model.training
    model.eval()
    t, y_true = simulate_states(de_val, cfg)
    y_pred = predict(model, t, de_val, device)
    if was_training:
        model.train()

    errors = [relative_l2(y_pred[:, c], y_true[:, c], channel=c)
              for c, _, _ in CHANNELS]
    return t, y_true, y_pred, errors


def r_errors(model, cfg, device, de_values, t_max=None):
    was_training = model.training
    model.eval()
    out = {}
    for de_val in de_values:
        t, y_true = simulate_states(de_val, cfg)
        y_pred = predict(model, t, de_val, device)
        mask = None if t_max is None else (t <= t_max)
        out[de_val] = relative_l2(y_pred[:, 0], y_true[:, 0], channel=0, mask=mask)
    if was_training:
        model.train()
    return out

def save_metrics(R, v, tau_rr, tau_tt, savepath="./metrics.csv"):
    df = pd.DataFrame({
        "R": R,
        "v": v,
        "tau_rr": tau_rr,
        "tau_tt": tau_tt
    })
    df.to_csv(savepath)