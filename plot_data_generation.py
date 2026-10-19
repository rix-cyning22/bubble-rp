import torch
from argparse import ArgumentParser
from model import load_checkpoint
from optim import pick_device
import os
from tqdm import tqdm
from metrics import evaluate_at_de
from ode import prepare_input_wave
import pandas as pd
from time import time

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--path", "-p")
    parser.add_argument("--output_dir", "-o", default="plots")
    parser.add_argument("--name", "-n")
    return parser.parse_args()

def get_dominant_amp(R):
    N = R.shape[0]
    R = R - R.mean()
    Y = torch.fft.rfft(R)
    P1 = torch.abs(Y) / N

    if N % 2 == 0:
        P1[1:-1] *= 2
    else:
        P1[1:] *= 2
    return torch.max(P1).item()
    

if __name__ == "__main__":
    args = get_arguments()
    torch.set_default_dtype(torch.float32)
    os.makedirs(args.output_dir, exist_ok=True)
    device = pick_device()
    model, config = load_checkpoint(args.path, device=device)
    input_wave = prepare_input_wave(config)
    de_values = torch.linspace(model.de_lo, model.de_hi, 100)
    true_freq_list, pred_freq_list = [], []
    with tqdm(de_values, desc="Generating") as pbar:
        for de in de_values:
            t, y_true, y_pred, _ = evaluate_at_de(
                model, config, device, de, input_wave=input_wave,
            )
            t = torch.tensor(t, device=device)
            R_true = torch.tensor(y_true[:, 0], device=device)
            R_pred = torch.tensor(y_pred[:, 0], device=device)
            true_amp = get_dominant_amp(R_true)
            pred_amp = get_dominant_amp(R_pred)
            true_freq_list.append(true_amp)
            pred_freq_list.append(pred_amp)
            pbar.set_postfix({
                "De": f"{de:.3f}",
                "True amp": f"{true_amp:.3f}",
                "Pred amp": f"{pred_amp:.3f}"
            })
            pbar.update()
    df = pd.DataFrame({
        "De": de_values.cpu().numpy(),
        "true_amp": true_freq_list,
        "pred_amp": pred_freq_list
    })
    error = (df["pred_amp"] - df["true_amp"])**2
    print(f"Mean squared error: {error.mean():.4e}")
    df.to_csv(os.path.join(args.output_dir, f"{args.name}_{time():.0f}.csv"), index=False)