from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

import config
from waveforms import describe
from data import anchor_tensors, sample_De
from losses import compute_loss, save_losses
from model import build_model, save_model
from optim import build_scheduler, pick_device
from plotting import plot_R_panels, plot_loss_curves
from metrics import save_metrics



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", default=str(config.DEFAULT_CONFIG))
    parser.add_argument("--epochs", type=int, help="override config training.epochs")
    parser.add_argument("--checkpoint", help="override config training.checkpoint_path")
    parser.add_argument("--save-every", type=int, help="override config training.save_every")
    parser.add_argument("--show", action="store_true", help="also open plot windows")
    return parser.parse_args()


def _create_run_directory(checkpoint_root, cfg):
    run_directory = os.path.join(checkpoint_root, describe(cfg))
    os.makedirs(run_directory, exist_ok=True)
    return run_directory


def _save_checkpoint(model, cfg, t_metrics, run_directory, epoch):
    epoch_directory = os.path.join(run_directory, f"epoch_{epoch}")
    os.makedirs(epoch_directory, exist_ok=True)
    save_model(model, os.path.join(epoch_directory, "model.pt"), cfg=cfg)

    with torch.no_grad():
        for de_value in cfg.training.eval_de:
            de_values = torch.full_like(t_metrics, float(de_value))
            states = model(t_metrics, de_values).cpu().numpy()
            de_tag = f"{de_value:g}".replace(".", "p")
            save_metrics(
                cfg.derived.t_eval,
                states[:, 0], states[:, 1], states[:, 2], states[:, 3],
                savepath=os.path.join(epoch_directory, f"metrics_de_{de_tag}.csv"),
            )


def _train_step(model, optimizer, scheduler, cfg, params, training_data, device):
    t_data, de_data, y_data = training_data
    training = cfg.training
     
    optimizer.zero_grad()
    t_physics = (torch.rand(training.collocation_points, 1, device=device)
                 * cfg.domain.t_end).requires_grad_(True)
    de_physics = sample_De(training.collocation_points, device, cfg)
    data_indices = torch.randint(0, t_data.shape[0], (training.data_batch,), device=device)

    loss, pde_loss, data_loss = compute_loss(
        model, t_physics, de_physics,
        t_data[data_indices], de_data[data_indices], y_data[data_indices],
        params, data_weight=training.data_weight, r_floor=training.r_floor,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training.grad_clip)
    optimizer.step()

    loss_value = loss.item()
    scheduler.step(loss_value)
    return loss_value, pde_loss, data_loss


def _save_plots(model, cfg, device, run_directory, epoch,
                pde_losses, data_losses, learning_rates, show):
    epoch_directory = os.path.join(run_directory, f"epoch_{epoch}")
    os.makedirs(epoch_directory, exist_ok=True)
    figures = [
        plot_R_panels(
            model, cfg, device, epoch,
            out_path=os.path.join(epoch_directory, "R.png") 
        ),
        plot_loss_curves(
            pde_losses, [x * cfg.training.data_weight for x in data_losses], learning_rates, epoch,
            out_path=os.path.join(epoch_directory, "loss.png")
        ),
    ]
    try:
        if show:
            plt.show()
    finally:
        for figure in figures:
            plt.close(figure)


def train(args, desc=None, save=True, early_stop=None, early_stop_wait=1):
    cfg = config.load_config(args.config)
    training = cfg.training
    epochs = training.epochs if args.epochs is None else args.epochs
    save_every = training.save_every if args.save_every is None else args.save_every
    if epochs < 1 or save_every < 1:
        raise ValueError("epochs and save_every must be positive")
    run_directory = None
    if save:
        run_directory = _create_run_directory(args.checkpoint or training.checkpoint_path, cfg)
        print(f"Saving model training data to {run_directory}")
    torch.set_default_dtype(torch.float32)
    device = pick_device()
    print(f"device: {device}")
    print(config.describe(cfg))

    training_data = anchor_tensors(cfg, device)
    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.lr)
    scheduler = build_scheduler(optimizer, cfg)
    params = config.pde_params(cfg)
    t_metrics = torch.tensor(
        cfg.derived.t_eval, dtype=torch.float32, device=device,
    ).view(-1, 1)

    pde_losses, data_losses, lrs = [], [], []
    try:
        progress = tqdm(range(epochs), desc=desc or "Training")
        with progress:
            num_hits = 0
            for step in progress:
                total_loss, pde_loss, data_loss = _train_step(
                    model, optimizer, scheduler, cfg, params, training_data, device,
                )
                pde_losses.append(pde_loss)
                data_losses.append(data_loss)
                lrs.append(scheduler.get_lr())
                progress.set_postfix({
                    "Total": f"{total_loss:.4e}",
                    "PDE": f"{pde_loss:.4e}",
                    "Data": f"{data_loss:.4e}",
                    "LR": f"{scheduler.get_lr():.3e}",
                })
                if early_stop is not None and total_loss < early_stop:
                    num_hits += 1
                    if num_hits >= early_stop_wait:
                        print(f"early stop invoked")
                        break
                epoch = step + 1
                if save and epoch % save_every == 0:
                    _save_checkpoint(model, cfg, t_metrics, run_directory, epoch)
                    _save_plots(
                        model, cfg, device, run_directory, epoch, pde_losses, data_losses, lrs, args.show
                    )
    finally:
        if save:
            save_losses(pde_losses, data_losses, lrs, os.path.join(run_directory, "losses.csv"))
            completed_epochs = len(pde_losses)
            if completed_epochs > 0 and completed_epochs % save_every != 0:
                _save_checkpoint(model, cfg, t_metrics, run_directory, completed_epochs)
                _save_plots(
                    model, cfg, device, run_directory, epoch, pde_losses, data_losses, lrs, args.show
                )
    return model


if __name__ == "__main__":
    args = parse_args()
    train(args)
