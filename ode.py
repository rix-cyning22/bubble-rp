import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from config import load_config
from argparse import ArgumentParser

def simulate(De, cfg):
    p, dv, slv = cfg.physical, cfg.derived, cfg.reference_solver

    lambda_hat = De / (2.0 * np.pi)
    G_hat = (p.eta * 2.0 * np.pi * p.f / De) / p.P0

    P_A_hat, gamma_hat = dv.P_A_hat, dv.gamma_hat
    P_g0_hat, I_param = dv.P_g0_hat, dv.I_param
    kappa = p.kappa

    def rhs(t_hat, y):
        R_hat, v_hat, tau_rr_hat, tau_tt_hat = y
        R_safe = max(R_hat, 1e-12)
        sr = v_hat / R_safe

        PA_hat = -P_A_hat * cfg.waveform(t_hat, t_end=cfg.domain.t_end)
        Pg_hat = P_g0_hat * (1.0 / R_safe) ** (3.0 * kappa)

        v_dot_hat = (1.0 / (I_param * R_safe)) * (
            Pg_hat - 1.0 - PA_hat - (gamma_hat / R_safe) + (tau_rr_hat - tau_tt_hat)
        ) - 0.5 * (v_hat**2 / R_safe)

        d_tau_rr = -4.0 * G_hat * sr - (tau_rr_hat / lambda_hat) - 4.0 * sr * tau_rr_hat
        d_tau_tt = 2.0 * G_hat * sr - (tau_tt_hat / lambda_hat) + 2.0 * sr * tau_tt_hat

        return [v_hat, v_dot_hat, d_tau_rr, d_tau_tt]

    sol = solve_ivp(
        rhs, [0.0, cfg.domain.t_end], [1.0, 0.0, 0.0, 0.0],
        t_eval=dv.t_eval, method=slv.method, rtol=slv.rtol, atol=slv.atol,
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def simulate_states(De, cfg, plot=False):
    t, *states = simulate(De, cfg)
    if plot:
        plt.plot(t, states[0])
        plt.title(f"De={De}")
        plt.show()
    return t, np.stack(states, axis=1)


if __name__ == "__main__":
    arg = ArgumentParser()
    arg.add_argument("--de", type=float, default=3.5, help="Deborah number to simulate")
    arg.add_argument("--config", default=str(load_config()), help="path to config.json")
    arg.add_argument("--plot", action="store_true", help="plot the results")
    args = arg.parse_args()
    simulate_states(args.de, load_config(), plot=args.plot)