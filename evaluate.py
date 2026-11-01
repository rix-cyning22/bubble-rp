import argparse
import os
import matplotlib.pyplot as plt
from metrics import CHANNELS, evaluate_at_de
import numpy as np
from model import load_checkpoint, n_params
from optim import pick_device
from plotting import plot_all_states, plot_just_R


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a .pt checkpoint")
    ap.add_argument("--de", required=True, type=float, nargs="+", metavar="DE",
                    help="one or more Deborah numbers to evaluate")
    ap.add_argument("--freq", type=float, default=None,
                    help="drive frequency for the output CSV filename")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    ap.add_argument("--plot_all_states", default=False, help="plot all states (default: only R)")
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise SystemExit(f"checkpoint not found: {args.model}")
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.model), "eval")
    os.makedirs(out_dir, exist_ok=True)
    print(args.plot_all_states)
    device = pick_device()
    model, cfg = load_checkpoint(args.model, device)
    frequency = args.freq
    if frequency is None and cfg is not None:
        frequency = cfg.waveform_add_args["kwargs"].get("freq")
    if frequency is None:
        raise SystemExit("drive frequency unavailable; provide it with --freq")

    lo, hi = model.de_lo, model.de_hi
    print(f"loaded {args.model}")
    print(f"  {n_params(model):,} params")
    print(f"  valid De range: [{lo}, {hi}]   device: {device}")

    outside = [d for d in args.de if not (lo <= d <= hi)]
    if outside:
        raise SystemExit(
            f"De {outside} outside the range this model was trained on "
            f"[{lo}, {hi}]; the network has no support there.")

    print(f"\n{'De':>7}  " + "  ".join(f"{name:>12}" for _, _, name in CHANNELS))
    print("-" * 63)
    figs = []
    for de_val in args.de:
        tag = f"{de_val:.3f}".replace(".", "p")
        de_dir = os.path.join(out_dir, f"{tag}")
        os.makedirs(de_dir, exist_ok=True)
        t, y_true, y_pred, errors = evaluate_at_de(
            model, cfg, device, de_val, savepath=de_dir)
        print(f"{de_val:>7.3f}  " + "  ".join(f"{e*100:11.2f}%" for e in errors))

        plot_fn = plot_all_states if args.plot_all_states else plot_just_R
        figs.append(plot_fn(t, y_true, y_pred, de_val, os.path.join(de_dir, f"eval_{tag}.eps")))
        figs.append(plot_fn(t, y_true, y_pred, de_val, os.path.join(de_dir, f"eval_{tag}.png")))

    if args.show:
        plt.show()
    for f in figs:
        plt.close(f)


if __name__ == "__main__":
    main()
