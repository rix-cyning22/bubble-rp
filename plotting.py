import matplotlib.pyplot as plt

from metrics import CHANNELS, evaluate_at_de, relative_l2


def plot_all_states(t, y_true, y_pred, de_val, out_path, skip=40):
    """2x2 grid of R, v, tau_rr, tau_theta against the numerical solution."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=140)

    for col, ylabel, _ in CHANNELS:
        ax = axes[col // 2, col % 2]
        err = relative_l2(y_pred[:, col], y_true[:, col], channel=col)

        ax.plot(t, y_true[:, col], 'k--', lw=2, label='numerical (solve_ivp)')
        ax.scatter(t[::skip], y_true[::skip, col], marker='o', s=28,
                   facecolors='none', edgecolors='black')
        ax.plot(t, y_pred[:, col], color='#0B51C1', lw=1.5, alpha=0.85, label='PINN')
        ax.scatter(t[::skip], y_pred[::skip, col], marker='o', s=28,
                   facecolors='none', edgecolors='#0B51C1')

        ax.set_xlabel(r'$\hat{t}$')
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f"{ylabel}")
        ax.legend(fontsize=8)

    plt.suptitle(f"De = {de_val}", fontsize=15)
    plt.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_R_panels(model, cfg, device, out_path, epoch=None, skip=25):
    n = len(cfg.training.eval_de)
    rows, cols = (2, (n + 1) // 2) if n > 3 else (1, n)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             dpi=140, squeeze=False)

    for i, de_val in enumerate(cfg.training.eval_de):
        ax = axes[i // cols, i % cols]
        t, y_true, y_pred, errors = evaluate_at_de(model, cfg, device, de_val)

        ax.plot(t, y_true[:, 0], 'k--', lw=2)
        ax.scatter(t[::skip], y_true[::skip, 0], marker='o', s=30,
                   facecolors='none', edgecolors='black')
        ax.plot(t, y_pred[:, 0], color='#0B51C1', lw=1.5, alpha=0.8)
        ax.scatter(t[::skip], y_pred[::skip, 0], marker='o', s=30,
                   facecolors='none', edgecolors='#0B51C1')

        ax.set_xlabel(r'$\hat{t}$')
        ax.set_ylabel(r'$R / R_0$', fontsize=13)
        ax.set_title(f"De = {de_val}", fontsize=12)

    for j in range(len(cfg.training.eval_de), rows * cols):
        axes[j // cols, j % cols].axis('off')

    if epoch is not None:
        plt.suptitle(f"Epoch {epoch}", fontsize=15)
    plt.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_loss_curves(pde_losses, data_losses, out_path, epoch=None):
    fig = plt.figure(figsize=(8, 5), dpi=140)
    plt.plot(pde_losses, "r-", alpha=0.8, label="PDE residual")
    plt.plot(data_losses, "b-", alpha=0.8, label="data (amplitude-normalized)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    if epoch is not None:
        plt.title(f"Epoch {epoch}")
    plt.tight_layout()
    fig.savefig(out_path)
    return fig


def plot_prediction_overlay(de_values, out_path, title, predict_fn):
    n = len(de_values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), dpi=140, squeeze=False)
    for i, de_val in enumerate(de_values):
        ax = axes[0, i]
        t, y_true, y_pred = predict_fn(de_val)
        err = relative_l2(y_pred[:, 0], y_true[:, 0], channel=0)
        ax.plot(t, y_true[:, 0], 'k--', lw=2, label='solve_ivp')
        ax.plot(t, y_pred[:, 0], color='#C1220B', lw=1.5, label='PINN (no data)')
        ax.set_xlabel(r'$\hat{t}$')
        ax.set_ylabel(r'$R / R_0$')
        ax.set_title(f"De = {de_val}")
        ax.legend(fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path)
    return fig
