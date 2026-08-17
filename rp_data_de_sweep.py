"""
Parametric sweep of the generalized Rayleigh-Plesset equation (Upper Convected
Maxwell viscoelastic model) over Deborah number `De`, solved numerically
(NOT with a PINN) and swept in parallel via multithreading.

Output: a single CSV with columns [t, De, R, tau_rr, tau_tt]
"""

import csv
import threading
import numpy as np
from scipy.integrate import solve_ivp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# -----------------------------------------------------------------------
# Fixed physical / non-dimensionalization parameters (same as base notebook)
# -----------------------------------------------------------------------
R0, f, pA = 1.5e-6, 3.0e6, 0.4e6
rho, gamma, eta = 1000.0, 0.072, 0.03
P0, kappa = 101325.0, 1.4

P_A_hat = pA / P0
gamma_hat = (2.0 * gamma) / (P0 * R0)
P_g0_hat = 1.0 + gamma_hat
I_param = (rho * (R0 ** 2) * (f ** 2)) / P0

t_end_hat = 9.0
t_eval_hat = np.linspace(0, t_end_hat, 1500, dtype=np.float64)

# -----------------------------------------------------------------------
# Deborah number sweep
# -----------------------------------------------------------------------
# NOTE: `np.linspace(0.5, 0.25, 4.5)` as originally written is invalid -
# the 3rd argument to linspace must be an integer *count* of points, not
# an endpoint, and 0.5 -> 0.25 is a decreasing range. Adjust this single
# line to whatever range/count was actually intended, e.g.:
#   De_values = np.linspace(0.5, 4.5, 5)   # 5 points from 0.5 to 4.5
#   De_values = np.arange(0.5, 4.75, 0.25) # step of 0.25
De_values = np.linspace(0.5, 4.5, 5)


# -----------------------------------------------------------------------
# Single-De numerical solve (identical physics to simulate_nondim in the
# original notebook, just De-parameterized and self-contained)
# -----------------------------------------------------------------------
def simulate_nondim(De: float):
    lambda_hat = De / (2.0 * np.pi)
    G_hat = (eta * 2.0 * np.pi * f / De) / P0

    def ode(t_hat, y):
        R_hat, v_hat, tau_rr_hat, tau_tt_hat = y
        R_safe = max(R_hat, 1e-12)
        sr = v_hat / R_safe

        PA_hat = -P_A_hat * np.sin(2.0 * np.pi * t_hat)
        Pg_hat = P_g0_hat * (1.0 / R_safe) ** (3.0 * kappa)

        v_dot_hat = (1.0 / (I_param * R_safe)) * (
            Pg_hat - 1.0 - PA_hat - (gamma_hat / R_safe) + (tau_rr_hat - tau_tt_hat)
        ) - 0.5 * (v_hat ** 2 / R_safe)

        d_tau_rr_hat = -4.0 * G_hat * sr - (tau_rr_hat / lambda_hat) - 4.0 * sr * tau_rr_hat
        d_tau_tt_hat = 2.0 * G_hat * sr - (tau_tt_hat / lambda_hat) + 2.0 * sr * tau_tt_hat

        return [v_hat, v_dot_hat, d_tau_rr_hat, d_tau_tt_hat]

    sol = solve_ivp(
        ode, [0, t_end_hat], [1.0, 0.0, 0.0, 0.0],
        t_eval=t_eval_hat, method="Radau", rtol=1e-6, atol=1e-8
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def worker(De: float):
    """Runs one De value and returns rows ready to write to CSV."""
    t, R, v, tau_rr, tau_tt = simulate_nondim(De)
    rows = [
        (float(t[i]), float(De), float(R[i]), float(tau_rr[i]), float(tau_tt[i]))
        for i in range(len(t))
    ]
    return De, rows


# -----------------------------------------------------------------------
# Multithreaded sweep
# -----------------------------------------------------------------------
def run_sweep(De_values, max_workers=8, out_csv="rp_parametric_sweep.csv"):
    results = {}
    lock = threading.Lock()

    # Swap ThreadPoolExecutor -> ProcessPoolExecutor for a real speedup,
    # since solve_ivp(method="Radau") is CPU-bound and mostly GIL-bound.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, De): De for De in De_values}

        for future in tqdm(as_completed(futures), total=len(futures), desc="De sweep"):
            De = futures[future]
            try:
                De_val, rows = future.result()
                with lock:
                    results[De_val] = rows
            except Exception as exc:
                print(f"De={De} generated an exception: {exc}")

    # Write out in De-sorted order, columns: t, De, R, tau_rr, tau_tt
    with open(out_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["t", "De", "R", "tau_rr", "tau_tt"])
        for De_val in sorted(results.keys()):
            writer.writerows(results[De_val])

    print(f"Saved {sum(len(r) for r in results.values())} rows to {out_csv}")
    return out_csv


if __name__ == "__main__":
    run_sweep(De_values, max_workers=min(8, len(De_values)))
