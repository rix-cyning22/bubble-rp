import torch


class PlateauLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, patience=20, cooldown_period=200,
                 min_lr=1e-6, factor=0.7, eps=1e-8):
        self.optimizer = optimizer
        self.patience = patience
        self.cooldown_period = cooldown_period
        self.cooldown = cooldown_period
        self.min_lr = min_lr
        self.factor = factor
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.eps = eps
        super().__init__(optimizer)

    def step(self, curr_loss=float("inf")):
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss
            return
        if self.cooldown > 0:
            self.cooldown -= 1
            return

        self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            self.bad_epochs = 0
            for group in self.optimizer.param_groups:
                prev_lr = group["lr"]
                new_lr = max(self.factor * prev_lr, self.min_lr)
                if prev_lr - new_lr > self.eps * prev_lr:
                    group["lr"] = new_lr
                    self.cooldown = self.cooldown_period

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


def build_scheduler(optimizer, cfg):
    s = cfg.training.scheduler
    return PlateauLR(optimizer, patience=s.patience,
                     cooldown_period=s.cooldown_period,
                     min_lr=s.min_lr, factor=s.factor)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
