# Deep Learning based Rayleigh-Plesset Equation Solver

## Introduction

This project trains a physics-informed neural network (PINN) to approximate the dynamics of an acoustically driven bubble governed by the Rayleigh-Plesset equation. The model takes a nondimensionalized time value and a Deborah number as inputs and predicts the bubble state over time, including the radius, radial velocity, and two auxiliary variables used to capture the viscous and memory effects in the bubble dynamics.

The implementation is designed for a single-bubble system with a sinusoidal driving pressure. Instead of solving the full nonlinear ODE directly at inference time, the network learns a surrogate that is constrained by the governing physics and the available reference data. This creates a fast, reusable emulator that can evaluate the bubble response across a range of operating conditions.

The default configuration in `config.json` targets a frequency of 0.5 and a Deborah number range of 1 to 4, with training checkpoints stored under `./checkpoints`.

## Formulation

The physical model is based on the dimensionless Rayleigh-Plesset system implemented in `ode.py` and enforced through the residual loss in `losses.py`.

The state is

- `R(t)`: nondimensional bubble radius
- `v(t)`: radial velocity
- `A(t)`: auxiliary variable for the viscous contribution
- `B(t)`: auxiliary variable for the pressure-memory contribution

The governing system is the dimensionless Rayleigh-Plesset model used in the reference solver and enforced in the training loss. In compact form, the state vector is

$$
\mathbf{y}(t) = \big(R(t), v(t), A(t), B(t)\big)^\top,
$$

with

$$
\dot{R} = v,
$$

$$
\lambda \dot{A} = \lambda R^3 v - A,
$$

$$
\lambda \dot{B} = \lambda v - B,
$$

and the radial momentum equation

$$
R \dot{v} = \frac{1}{I}\left(P_g - 1 - \frac{\gamma}{R} - 2G\left(\frac{A}{R^4} + \frac{B}{R}\right) - P_A \psi(t)\right) - \frac{3}{2}v^2.
$$

Here the gas pressure term is

$$
P_g = P_{g0} R^{-3\kappa},
$$

with

$$
P_{g0} = 1 + \gamma_hat,
\qquad
\gamma_hat = \frac{2\gamma}{P_0 R_0},
\qquad
P_A = \frac{p_A}{P_0},
\qquad
\lambda = \frac{De}{2\pi},
$$

and the viscous coefficient

$$
G = \frac{\eta_P0_f}{De}
= \frac{\eta\,(2\pi f)}{P_0\,De}.
$$

The code uses the drive waveform $\psi(t)$ as the nondimensional pressure forcing. For the default sine case,

$$
\psi(t) = \sin(2\pi f t),
$$

with frequency $f = 0.5$ in the default configuration.

The training residuals are assembled as the residual of the four governing equations:

$$
\mathcal{R}_1 = \dot{R} - v,
$$

$$
\mathcal{R}_2 = R\dot{v} - \frac{1}{I}\left(P_g - 1 - \frac{\gamma}{R} - 2G\left(\frac{A}{R^4} + \frac{B}{R}\right) - P_A\psi(t)\right) + \frac{3}{2}v^2,
$$

$$
\mathcal{R}_3 = \frac{\lambda \dot{A} - \lambda R^3 v + A}{De},
$$

$$
\mathcal{R}_4 = \frac{\lambda \dot{B} - \lambda v + B}{De}.
$$

The model in `model.py` predicts $(R, v, A, B)$ from inputs $(t, De)$ using a Fourier-feature multilayer perceptron, and the training objective is the hybrid loss

$$
\mathcal{L} = \mathcal{L}_{PDE} + \alpha\,\mathcal{L}_{data},
$$

where $\mathcal{L}_{PDE}$ is the residual loss built from the equations above and $\mathcal{L}_{data}$ measures the mismatch against reference bubble trajectories from the high-accuracy ODE solver.

## Repository structure

- `config.json`: core physical parameters, training hyperparameters, and evaluation settings
- `train.py`: trains the PINN model and saves checkpoints
- `evaluate.py`: loads a saved model and evaluates it at chosen Deborah numbers
- `ode.py`: reference Rayleigh-Plesset solver used to generate training data
- `model.py`: neural network architecture and checkpoint I/O
- `losses.py`: residual and training loss definitions
- `data.py`: generation of anchor trajectories and collocation sampling
- `metrics.py` and `plotting.py`: evaluation metrics and visualizations

## Setup

### Prerequisites

- Python 3.13 or newer
- A local virtual environment is recommended

### Install uv if needed

If `uv` is not already installed, install it from the official Astral installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project environment:

```bash
git clone <repository-url>
cd bubble-rp
uv sync
```

This installs the project and the dependency set declared in `pyproject.toml`, including `torch`, `numpy`, `scipy`, `pandas`, and `matplotlib`.

If you want to activate the environment manually, you can do:

```bash
source .venv/bin/activate
```

### Train the model

The default training pipeline reads the settings in `config.json` and begins training immediately:

```bash
uv run train.py
```

You can override config values as needed, for example:

```bash
uv run train.py
```

Training writes checkpoints into `./checkpoints/<run_name>/epoch_<n>/`, including the model weights and evaluation CSV files. The configs used for the training is given in `config.py`

### Evaluate a trained model

After training, evaluate the saved checkpoint at one or more Deborah numbers:

```bash
uv run evaluate.py --model ./checkpoints/<run_name>/epoch_8000/model.pt --de 1.0 2.0 3.0 4.0
```

## Notes

- The training data is generated from the reference solver using anchor values of `De` across the domain.
- The collocation points are sampled over the same range, allowing the network to satisfy the PDE residual across the full operating space.
- Checkpoints are saved periodically so that training can be resumed or evaluated at specific epochs.

