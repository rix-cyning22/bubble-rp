import torch
import numpy as np


def sine(t, *args, **kwargs):
    if isinstance(t, torch.Tensor):
        return torch.sin(2.0 * np.pi * t)
    return np.sin(2.0 * np.pi * t)


def square(t, t_end, *args, **kwargs):
    half_wavelength = t_end / 3
    if isinstance(t, torch.Tensor):
        return torch.where((t > half_wavelength) & (t < 2 * half_wavelength), -1, 1)
    else:
        return np.where((t > half_wavelength) & (t < 2 * half_wavelength), -1, 1)


def sawtooth(t, *args, **kwargs):
    return 2.0 * (t % 1.0) - 1.0

REGISTRY = {
    'sine': sine,
    'square': square,
    'sawtooth': sawtooth
}

def plot_wave(t, wave_type, ax, t_end, *args, **kwargs):
    ax.plot(t, get(wave_type)(t))

def get(name):
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown waveform '{name}'. "
            f"Available: {list(REGISTRY)}. "
            f"To add a custom waveform, register it in waveforms.REGISTRY."
        )
    return REGISTRY[name]
