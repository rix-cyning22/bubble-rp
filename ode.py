import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from config import load_config
from argparse import ArgumentParser

def simulate(De, cfg):
    p, dv, slv = cfg.physical, cfg.derived, cfg.reference_solver

    lambda_hat = De / (2.0 * np.pi)
    G_hat = dv.eta_P0_f / De if De > 0 else 0

    P_A_hat, gamma_hat, P_g0_hat, I_param, kappa = dv.P_A_hat, dv.gamma_hat, dv.P_g0_hat, dv.I_param, p.kappa

    def rhs(t_hat, y):
        R_hat, v_hat, A_hat, B_hat = y
        R_safe = max(R_hat, cfg.training.r_floor)

        PA_hat = -P_A_hat * cfg.waveform(t_hat, t_end=cfg.domain.t_end)
        Pg_hat = P_g0_hat * (1.0 / R_safe) ** (3.0 * kappa)
        dp_hat = Pg_hat - 1 + PA_hat
        if lambda_hat > 0:
            visc_hat = 2 * G_hat * (A_hat / R_safe ** 4 + B_hat / R_safe)
            dA_hat = R_safe ** 3 * v_hat - A_hat / lambda_hat
            dB_hat = v_hat - B_hat / lambda_hat
        else:
            visc_hat = 0
            dA_hat = 0
            dB_hat = 0
        v_dot_hat = ((dp_hat - gamma_hat / R_safe - visc_hat) / I_param - 1.5 * v_hat ** 2) / R_safe
        return [v_hat, v_dot_hat, dA_hat, dB_hat]

    sol = solve_ivp(
        rhs, [0.0, cfg.domain.t_end], [1.0, 0.0, 0.0, 0.0],
        t_eval=dv.t_eval, method=slv.method, rtol=slv.rtol, atol=slv.atol,
        max_step=1/200
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