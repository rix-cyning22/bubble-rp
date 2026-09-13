import argparse
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from config import load_config, pde_params, describe, DEFAULT_CONFIG
from data import anchor_tensors, sample_De
from losses import compute_loss
from model import build_model, save_checkpoint
from optim import build_scheduler, pick_device
from plotting import plot_R_panels, plot_loss_curves

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--epochs", type=int, help="override config training.epochs")
    ap.add_argument("--checkpoint", help="override config training.checkpoint")
    ap.add_argument("--save-every", type=int, help="override config training.save_every")
    ap.add_argument("--plot-every", type=int, help="override config training.plot_every")
    ap.add_argument("--show", action="store_true", help="also open plot windows")
    return ap.parse_args()

def save_checkpoint(model, losses):
    # write this here
    pass


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tr = cfg.training

    epochs = args.epochs or tr.epochs
    ckpt_path = args.checkpoint or tr.checkpoint
    save_every = args.save_every or tr.save_every
    plot_every = args.plot_every or tr.plot_every

    torch.set_default_dtype(torch.float32)
    device = pick_device()
    print(f"device: {device}")
    print(describe(cfg))

    t_data, De_data, y_data, amp_data = anchor_tensors(cfg, device)
    n_data = t_data.shape[0]

    model = build_model(cfg, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=tr.lr)
    scheduler = build_scheduler(optimizer, cfg)
    params = pde_params(cfg)

    pde_losses, data_losses, total_losses = [], [], []

    with tqdm(range(epochs), desc="Training parametric PINN") as pbar:
        for epoch in pbar:
            optimizer.zero_grad()

            t_physics = (torch.rand(tr.collocation_points, 1, device=device)
                         * cfg.domain.t_end).requires_grad_(True)
            De_physics = sample_De(tr.collocation_points, device, cfg)
            idx = torch.randint(0, n_data, (tr.data_batch,), device=device)

            loss, loss_pde, loss_data = compute_loss(
                model, t_physics, De_physics,
                t_data[idx], De_data[idx], y_data[idx], amp_data[idx],
                params, data_weight=tr.data_weight, r_floor=tr.r_floor,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tr.grad_clip)
            optimizer.step()
            scheduler.step(loss.item())

            pde_losses.append(loss_pde)
            data_losses.append(loss_data)
            total_losses.append(loss.item())

            pbar.set_postfix({
                'Total': f"{loss.item():.4e}",
                'PDE': f"{loss_pde:.4e}",
                'Data': f"{loss_data:.4e}",
                'LR': f"{scheduler.get_lr():.3e}",
            })

            done = epoch + 1
            if done % save_every == 0:
                save_checkpoint(model, ckpt_path, cfg=cfg, epoch=done, loss=loss.item())

            if done % plot_every == 0:
                figs = [
                    plot_R_panels(model, cfg, device, out_path=f"epoch_{done}_R.png", epoch=done),
                    plot_loss_curves(pde_losses, data_losses, out_path=f"epoch_{done}_loss.png", epoch=done),
                ]
                if args.show:
                    plt.show()
                for f in figs:
                    plt.close(f)

    save_checkpoint(model, ckpt_path, cfg=cfg, epoch=epochs, loss=total_losses[-1] if total_losses else None)


if __name__ == "__main__":
    main()
