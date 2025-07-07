#runners.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score as pr_auc
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from typing import Sequence
from .model import SparseRowDataset
from .helpers import compute_metrics, save_model, prettyPrintMetrics
from typing import Sequence, Dict, Tuple, Any
from torch.nn.utils import clip_grad_norm_


from contextlib import nullcontext as _nullcontext        

if torch.cuda.is_available():
    def _autocast(enabled: bool = True):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)

    class _GradScaler(torch.amp.GradScaler):
        def __init__(self, enabled: bool = True):
            super().__init__(enabled=enabled)

else:  # fall-back (CPU)
    _autocast = lambda enabled=True: _nullcontext()          

    class _GradScaler:
        def __init__(self, *a, **k): ...
        def scale(self, loss):      return loss
        def unscale_(self, opt):    ...
        def step(self, opt):        opt.step()
        def update(self):           ...


#  Main training loop


def training(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    *,
    pos_neg_ratio: float,
    cycles: int               = 5,
    epochs_per_cycle: int     = 1,
    lr: float                 = 2e-3,
    weight_decay: float       = 1e-2,
    grad_clip_norm: float     = 1.0,
    amp: bool                 = True,
    path: str                 = ""
) -> Dict[str, Any]:
    """
    Train `model` with cyclic checkpointing.

    Returns the performance dictionary of the best global checkpoint.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ----- Optimisation setup -------------------------------------------------
    criterion  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_neg_ratio], device=device)
    )
    optimizer  = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine cycles within each epoch-cycle
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0   = steps_per_epoch * epochs_per_cycle,
        T_mult= 2,
        eta_min=1e-5,
    )

    autocast = _autocast(enabled=amp and device.type == "cuda")
    scaler   = _GradScaler(enabled=amp and device.type == "cuda")

    # Checkpoint directory
    model_dir = path.rstrip("/") + "/models/" if path else "models/"
    os.makedirs(model_dir, exist_ok=True)

    # ----- GLOBAL best trackers ----------------------------------------------
    global_best_metrics: Dict[str, float] = {"accuracy": 0.0}
    global_best_state   = model.state_dict()

    # ==================== CYCLE LOOP =========================================
    for cycle in range(cycles):
        print(f"\n---------------- Cycle {cycle+1:02}/{cycles:02} -----------------------------")

        cycle_best_metrics: Dict[str, float] = {"accuracy": 0.0}
        cycle_best_state   = None

        # --------------- EPOCH loop -----------------
        for epoch in range(1, epochs_per_cycle + 1):
            model.train()
            running_loss = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with autocast:
                    logits = model(xb).view(-1)
                    loss   = criterion(logits, yb)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item() * xb.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            # ---------- VALIDATION ----------
            print(f"→ Epoch {epoch:02}/{epochs_per_cycle} – ", end="")
            val_metrics, _, _ = compute_metrics(model, test_loader, device=device)

            if val_metrics["accuracy"] > cycle_best_metrics["accuracy"]:
                cycle_best_metrics = val_metrics
                cycle_best_state   = model.state_dict()
                prettyPrintMetrics(cycle_best_metrics,
                                  message="\t  New BEST in this cycle ")

        # --------------- END-OF-CYCLE -----------------
        # Promote to GLOBAL best
        if cycle_best_metrics["accuracy"] > global_best_metrics["accuracy"]:
            global_best_metrics = cycle_best_metrics
            global_best_state   = cycle_best_state
            prettyPrintMetrics(global_best_metrics,
                               message="\n Updated GLOBAL best: ")
            save_model(global_best_state,
                       model_dir + "deep_antiphish_best_model.pth",
                       global_best_metrics["accuracy"])

        # Warm-start next cycle from the best weights of **this** cycle
        if cycle_best_state is not None:
            model.load_state_dict(cycle_best_state)

        # Always save a per-cycle checkpoint
        save_model(cycle_best_state,
                   model_dir + f"deep_antiphish_cycle{cycle+1:02}.pth",
                   cycle_best_metrics["accuracy"])

    # ==================== FINAL EVAL =========================================
    print("\n== Final evaluation on TEST loader using GLOBAL best ==")
    model.load_state_dict(global_best_state)
    final_metrics, _, _ = compute_metrics(model, test_loader, device=device)
    prettyPrintMetrics(final_metrics,
                       message=" Final GLOBAL performance : ",
                       print_confusion_matrix=True,
                       print_classification_report=True)

    return final_metrics