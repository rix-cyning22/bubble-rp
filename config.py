import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import numpy as np

import waveforms as _waveforms

DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def _namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in obj.items()})
    return obj


def _call_waveform(function, waveform_args, t, *args, **kwargs):
    call_kwargs = {**waveform_args['kwargs'], **kwargs}
    return function(t, *waveform_args['args'], *args, **call_kwargs)


def load_config(path=DEFAULT_CONFIG):
    cfg = _namespace(json.loads(Path(path).read_text()))

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
    waveform_config = cfg.waveform
    cfg.waveform_name = waveform_config.name
    cfg.waveform_add_args = {
        'args': tuple(waveform_config.args),
        'kwargs': vars(waveform_config.kwargs),
    }
    cfg.waveform = partial(
        _call_waveform,
        _waveforms.get(waveform_config.name),
        cfg.waveform_add_args,
    )
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
    waveform_args = [str(value) for value in cfg.waveform_add_args['args']]
    waveform_args.extend(
        f"{name}={value}"
        for name, value in cfg.waveform_add_args['kwargs'].items()
    )
    waveform = cfg.waveform_name
    if waveform_args:
        waveform += f"({', '.join(waveform_args)})"

    return (f"waveform: {waveform}  |  "
            f"De in [{cfg.domain.de_lo}, {cfg.domain.de_hi}], "
            f"t_hat in [0, {cfg.domain.t_end}]  |  "
            f"P_A={dv.P_A_hat:.3f} gamma={dv.gamma_hat:.3f} "
            f"P_g0={dv.P_g0_hat:.3f} I={dv.I_param:.4f}")
