import numpy as np
import torch
import pandas as pd


def physics_residuals(model, t, De, params, r_floor=1e-5):
    preds = model(t, De)
    R, v, tau_rr, tau_th = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3], preds[:, 3:4]

    def d_dt(y):
        return torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]

    R_t, v_t, tau_rr_t, tau_th_t = d_dt(R), d_dt(v), d_dt(tau_rr), d_dt(tau_th)

    I, P_A, gamma = params['I'], params['P_A'], params['gamma']
    P_g0, kappa = params['P_g0'], params['kappa']

    lam = De / (2.0 * np.pi)
    lam_G = params['eta_P0_f'] / (2.0 * np.pi)
    R_safe = torch.clamp(R, min=r_floor)
    v_R = v / R_safe

    res_1 = R_t - v

    term1 = P_g0 * (R_safe ** (-3 * kappa)) - 1.0 + P_A * params['waveform'](t, params["t_end"])
    term2 = - (gamma / R_safe) + (tau_rr - tau_th)
    res_2 = (v_t * R_safe) - (1.0 / I) * (term1 + term2) + 0.5 * (v**2)

    res_3 = (lam * tau_rr_t
             - (-4.0 * lam_G * v_R - tau_rr - 4.0 * lam * v_R * tau_rr)) / De
    res_4 = (lam * tau_th_t
             - (2.0 * lam_G * v_R - tau_th + 2.0 * lam * v_R * tau_th)) / De

    return torch.cat([res_1, res_2, res_3, res_4], dim=1), R


def compute_loss(model, t, De, t_data, De_data, y_data, amp_data, params,
                 data_weight=1000.0, r_floor=1e-5):
    res, _ = physics_residuals(model, t, De, params, r_floor=r_floor)

    huber = torch.nn.HuberLoss(delta=1.0)
    zeros = torch.zeros_like(res[:, 0:1])
    mse_pde = sum(huber(res[:, i:i + 1], zeros) for i in range(res.shape[1]))
    preds_data = model(t_data, De_data)
    mse_data = torch.mean(((preds_data - y_data) / amp_data) ** 2)

    return mse_pde + mse_data * data_weight, mse_pde.item(), mse_data.item()


def residual_scales(params, de_ref=2.0, device=None):
    lam_G = params['eta_P0_f'] / (2.0 * np.pi)
    return torch.tensor([
        1.0,                            
        params['P_A'] / params['I'],     # res_2, forcing-dominated
        4.0 * lam_G / de_ref,            # res_3
        2.0 * lam_G / de_ref,            # res_4
    ], device=device)


def save_losses(pde_losses, data_losses, savepath):
    df = pd.DataFrame({
        "pde": pde_losses,
        "data": data_losses
    })
    df.to_csv(savepath)