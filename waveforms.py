import torch
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
import config
from matplotlib import pyplot as plt


def sine(t, *args, **kwargs):
    freq = kwargs.get('freq', args[0] if args else 2.0)
    if isinstance(t, torch.Tensor):
        return torch.sin(freq * np.pi * t)
    return np.sin(2 * freq * np.pi * t)


def square(t, *args, **kwargs):
    t_end = kwargs['t_end']
    num_harmonics = kwargs.get(
        'num_harmonics', args[0] if args else 15,
    )
    freq = kwargs.get('freq', 3.0)
    if t_end <= 0 or num_harmonics < 1:
        raise ValueError("t_end and num_harmonics must be positive")

    fundamental_freq = freq / t_end
    result = 0.0
    for i in range(num_harmonics):
        n = 2 * i + 1 
        amplitude = 4.0 / (np.pi * n)
        harmonic_freq = fundamental_freq * n
        result += amplitude * sine(t, freq=harmonic_freq)
        
    return result


def sawtooth(t, *args, **kwargs):
    t_end = kwargs.get('t_end', 1.0)
    num_harmonics = kwargs.get(
        'num_harmonics', args[0] if args else 15,
    )
    if t_end <= 0 or num_harmonics < 1:
        raise ValueError("t_end and num_harmonics must be positive")

    fundamental_freq = kwargs.get('freq', 3.0 / t_end)
    result = 0.0
    for n in range(1, num_harmonics + 1):
        amplitude = 2.0 / (np.pi * n)
        harmonic_freq = fundamental_freq * n
        result -= amplitude * sine(t, freq=harmonic_freq)
        
    return result

REGISTRY = {
    'sine': sine,
    'square': square,
    'sawtooth': sawtooth
}

def get(name):
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown waveform '{name}'. "
            f"Available: {list(REGISTRY)}. "
            f"To add a custom waveform, register it in waveforms.REGISTRY."
        )
    return REGISTRY[name]

def describe(cfg):
    waveform_args = cfg.waveform_add_args
    values = [str(value) for value in waveform_args['args']]
    values.extend(
        f"{key}_{str(value).replace('.', 'p')}"
        for key, value in waveform_args['kwargs'].items()
    )
    run_name = f"run_{cfg.waveform_name}"
    if values:
        run_name += "_" + "_".join(values)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{run_name}_{timestamp}"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = config.load_config(args.config)
    t = np.linspace(0, cfg.domain.t_end, 1000)
    wave = cfg.waveform(t, t_end=cfg.domain.t_end)
    plt.plot(t, wave)
    plt.title(describe(cfg))
    plt.show()