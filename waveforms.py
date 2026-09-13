import torch
import math
import numpy as np

TWO_PI = 2.0 * math.pi

def _sin(x):
    if isinstance(x, torch.Tensor):
        return torch.sin(x)
    return np.sin(x)


def _sign(x):
    if isinstance(x, torch.Tensor):
        return torch.sign(x)
    return np.sign(x)


def _clamp(x, lo, hi):
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, lo, hi)
    return np.clip(x, lo, hi)

def sine(t):
    return _sin(TWO_PI * t)


def square(t):
    return _sign(0.5 - (t % 1.0))


def sawtooth(t):
    return 2.0 * (t % 1.0) - 1.0


def triangle(t):
    phase = t % 1.0 
    return 1.0 - 4.0 * _clamp(phase - 0.25, 0.0, 0.5)

REGISTRY = {
    'sine': sine,
    'square': square,
    'sawtooth': sawtooth,
    'triangle': triangle,
}


def get(name):
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown waveform '{name}'. "
            f"Available: {list(REGISTRY)}. "
            f"To add a custom waveform, register it in waveforms.REGISTRY."
        )
    return REGISTRY[name]
