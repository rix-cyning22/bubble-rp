import argparse
import os
import matplotlib.pyplot as plt

from config import load_config, DEFAULT_CONFIG
from metrics import CHANNELS, evaluate_at_de
from model import load_checkpoint, n_params
from optim import pick_device
from plotting import plot_all_states


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a .pt checkpoint")
    ap.add_argument("--de", required=True, type=float, nargs="+", metavar="DE",
                    help="one or more Deborah numbers to evaluate")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"checkpoint not found: {args.model}")
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config(args.config)
    device = pick_device()
    model, ckpt = load_checkpoint(args.model, device)

    lo, hi = model.de_lo, model.de_hi
    print(f"loaded {args.model}")
    print(f"  trained {ckpt.get('epoch', '?')} epochs, {n_params(model):,} params")
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
        t, y_true, y_pred, errors = evaluate_at_de(model, cfg, device, de_val)
        print(f"{de_val:>7.3f}  " + "  ".join(f"{e*100:11.2f}%" for e in errors))

        tag = f"{de_val:.3f}".replace(".", "p")
        out_path = os.path.join(args.out_dir, f"eval_De_{tag}.png")
        figs.append(plot_all_states(t, y_true, y_pred, de_val, out_path))
        print(f"saved {out_path}")

    if args.show:
        plt.show()
    for f in figs:
        plt.close(f)


if __name__ == "__main__":
    main()
