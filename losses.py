import numpy as np
import torch
import pandas as pd


def physics_residuals(model, t, De, params, r_floor=1e-5):
    preds = model(t, De)
    R, v, A, B = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3], preds[:, 3:4]

    def d_dt(y):
        return torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]

    R_t, v_t, A_t, B_t = d_dt(R), d_dt(v), d_dt(A), d_dt(B)

    I, P_A, gamma, P_g0, kappa = params['I'], params['P_A'], params['gamma'], params['P_g0'], params['kappa']

    lam = De / (2.0 * np.pi)
    G = params['eta_P0_f'] / De 
    R_safe = torch.clamp(R, min=r_floor)
    v_R = v / R_safe

    res_1 = R_t - v

    dp = P_g0 * (R_safe ** (-3 * kappa)) -1 - P_A * params["waveform"](t, t_end=params["t_end"])
    visc = 2 * G * (A / R_safe**4 + B / R_safe)
    res_2 = (v_t * R_safe) - (dp - gamma / R_safe -visc) / I + 1.5 * v**2
    res_3 = (lam * A_t - lam * R_safe**3 * v + A) / De
    res_4 = (lam * B_t - lam * v + B) / De

    return torch.cat([res_1, res_2, res_3, res_4], dim=1), R


def compute_loss(model, t, De, t_data, De_data, y_data, params,
                 data_weight=1000.0, r_floor=1e-5):
    res, _ = physics_residuals(model, t, De, params, r_floor=r_floor)

    huber = torch.nn.HuberLoss(delta=1.0)
    zeros = torch.zeros_like(res[:, 0:1])
    mse_pde = sum(huber(res[:, i:i + 1], zeros) for i in range(res.shape[1]))
    preds_data = model(t_data, De_data)
    mse_data = torch.mean(((preds_data - y_data)) ** 2)

    return mse_pde + mse_data * data_weight, mse_pde.item(), mse_data.item()


def save_losses(pde_losses, data_losses, learning_rates, savepath):
    df = pd.DataFrame({
        "pde": pde_losses,
        "data": data_losses,
        "learning_rates": learning_rates
    })
    df.to_csv(savepath)