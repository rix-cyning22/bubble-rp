import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, t_end, de_lo, de_hi, p_dim=1, out_dim=4, hidden_dim=64,
                 num_layers=3, num_freqs=64, sigma=3.0, num_de_freqs=32, sigma_de=2.0):
        super().__init__()
        self.activation = nn.SiLU()
        self.t_end = float(t_end)
        self.de_lo = float(de_lo)
        self.de_hi = float(de_hi)
        self.arch = dict(
            t_end=self.t_end, de_lo=self.de_lo, de_hi=self.de_hi,
            p_dim=p_dim, out_dim=out_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_freqs=num_freqs, sigma=sigma,
            num_de_freqs=num_de_freqs, sigma_de=sigma_de,
        )
        self.B_t = nn.Parameter(torch.randn(1, num_freqs) * sigma, requires_grad=False)
        self.B_de = nn.Parameter(torch.randn(p_dim, num_de_freqs) * sigma_de,
                                 requires_grad=False)

        in_dim = num_freqs * 2 + 1 + p_dim + num_de_freqs * 2

        self.U_layer = nn.Linear(in_dim, hidden_dim)
        self.V_layer = nn.Linear(in_dim, hidden_dim)
        self.first_layer = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_dim, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

        nn.init.xavier_normal_(self.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def features(self, t, De):
        t_n = t / self.t_end
        De_n = 2.0 * (De - self.de_lo) / (self.de_hi - self.de_lo) - 1.0

        t_proj = 2.0 * np.pi * t_n @ self.B_t
        de_proj = 2.0 * np.pi * De_n @ self.B_de

        return torch.cat([
            2.0 * t_n - 1.0,
            torch.sin(t_proj), torch.cos(t_proj),
            De_n,
            torch.sin(de_proj), torch.cos(de_proj),
        ], dim=1)

    def forward(self, t, De):
        x = self.features(t, De)

        U = self.activation(self.U_layer(x))
        V = self.activation(self.V_layer(x))

        H = self.activation(self.first_layer(x))
        for layer in self.hidden_layers:
            Z = self.activation(layer(H))
            H = Z * U + (1.0 - Z) * V

        raw = self.output_layer(H)

        R      = 1.0 + t * raw[:, 0:1]
        v      = 0.0 + t * raw[:, 1:2]
        tau_rr = 0.0 + t * raw[:, 2:3]
        tau_th = 0.0 + t * raw[:, 3:4]

        return torch.cat([R, v, tau_rr, tau_th], dim=1)


def build_model(cfg, device):
    m = cfg.model
    return Model(
        t_end=cfg.domain.t_end, de_lo=cfg.domain.de_lo, de_hi=cfg.domain.de_hi,
        hidden_dim=m.hidden_dim, num_layers=m.num_layers, num_freqs=m.num_freqs,
        sigma=m.sigma, num_de_freqs=m.num_de_freqs, sigma_de=m.sigma_de,
    ).to(device)


def n_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def save_model(model, path,**metadata):
    torch.save({
        'state_dict': model.state_dict(), 
        'arch': model.arch, 
        'metadata': metadata,
        'n_params': n_params(model)
    }, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Model(**ckpt['arch']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt
