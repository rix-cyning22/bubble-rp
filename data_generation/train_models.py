import os
import numpy as np
from metrics import evaluate_at_de, save_radius_comparison
from optim import pick_device
from train import parse_args, train
from tqdm import tqdm
from config import load_config

def main(cfg, args):
    sweep = cfg.fft_data_generation
    os.makedirs(sweep.save_dir, exist_ok=True)
    frequencies = np.linspace(*sweep.drive_freq_range, sweep.num_drive_freqs)
    de_values = np.linspace(*sweep.de_range, sweep.num_de)
    print(f"de: {de_values}")
    print(f"freq: {frequencies}")
    device = pick_device()

    for f in frequencies:
        cfg.waveform_add_args["kwargs"]["freq"] = float(f)
        model = train(cfg, args, desc=f"freq={f:g}", save=False, tol=sweep.tolerance, hits_below_tol=sweep.hits_below_tolerance)
        with tqdm(total=sweep.num_de, desc=f"Evaluating for f={f:g}") as pbar:
            for de in de_values:
                t, y_true, y_pred, _ = evaluate_at_de(model, cfg, device, float(de))
                fp = os.path.join(
                    sweep.save_dir, 
                    f"data_de_{str(de).replace('.', "p")}_f_{str(f).replace('.', 'p')}.csv"
                )
                save_radius_comparison(t, y_true, y_pred, fp)
                pbar.set_postfix({"path": fp})
                pbar.update(1)
                
if __name__ == "__main__":
    args = parse_args()
    main(load_config(args.config), args)