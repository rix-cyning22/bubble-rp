import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

import waveforms as _waveforms

DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def _namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in obj.items()})
    return obj


def load_config(path=DEFAULT_CONFIG):
    raw = json.loads(Path(path).read_text())
    cfg = _namespace(raw)

    p, d = cfg.physical, cfg.domain
    gamma_hat = (2.0 * p.gamma) / (p.P0 * p.R0)
    cfg.derived = SimpleNamespace(
        P_A_hat=p.pA / p.P0,
        gamma_hat=gamma_hat,
        P_g0_hat=1.0 + gamma_hat,
        I_param=(p.rho * p.R0**2 * p.f**2) / p.P0,
        eta_P0_f=(p.eta * 2.0 * np.pi * p.f) / p.P0,
        t_eval=np.linspace(0.0, d.t_end, d.n_t_eval, dtype=np.float32),
    )
    cfg.waveform = _waveforms.get(raw.get('waveform', 'sine'))
    cfg.waveform_name = raw.get('waveform', 'sine')
    cfg.raw = raw
    return cfg


def pde_params(cfg):
    dv = cfg.derived
    return {
        'I': dv.I_param,
        'P_A': dv.P_A_hat,
        'gamma': dv.gamma_hat,
        'P_g0': dv.P_g0_hat,
        'kappa': cfg.physical.kappa,
        'eta_P0_f': dv.eta_P0_f,
        'waveform': cfg.waveform,
        "t_end": cfg.domain.t_end
    }


def describe(cfg):
    dv = cfg.derived
    return (f"waveform: {cfg.waveform_name}  |  "
            f"De in [{cfg.domain.de_lo}, {cfg.domain.de_hi}], "
            f"t_hat in [0, {cfg.domain.t_end}]  |  "
            f"P_A={dv.P_A_hat:.3f} gamma={dv.gamma_hat:.3f} "
            f"P_g0={dv.P_g0_hat:.3f} I={dv.I_param:.4f}")
