#hyperparameters.py

# import sys
# import os
# repo_root = os.path.dirname(os.path.abspath(''))
# if repo_root not in sys.path:
#     sys.path.append(repo_root)

from .runners import training
from .model import DeepAntiPhish
from .helpers import get_feature_count, prettyPrintMetrics, compute_metrics
import torch
import optuna
import torch.nn as nn
from functools import partial
from .feature_engineering import process, imbalance_ratio
from .constants import *
import joblib
from torch.utils.data import DataLoader

def training_hyperparam(
    model:        nn.Module,
    train_loader,
    test_loader,
    *,
    pos_neg_ratio: float,
    cycles: int = 5,
    epochs_per_cycle: int = 1,
    lr: float = 2e-3,
    weight_decay: float = 1e-2,
    grad_clip_norm: float = 1.0,
    amp: bool = True
    ):
    # Local Variables
    kargs = {
        key: val
        for key, val in locals().items()
            if key not in {"model", "train_loader", "test_loader"}
    }
    kargs['path']=''
    vals = training(model, train_loader, test_loader, **kargs)
    prettyPrintMetrics(vals, "Generated HyperParameter Performance: ")
    return vals['accuracy']

def run_once(train_loader, test_loader, hparams: dict) -> float:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return training_hyperparam (
        model = DeepAntiPhish(input_dims=get_feature_count(train_loader)),
        train_loader = train_loader,
        test_loader = test_loader,
        pos_neg_ratio = imbalance_ratio(train_loader),
        cycles = hparams["cycles"],
        epochs_per_cycle = hparams["epochs_per_cycle"],
        lr = hparams["lr"],
        weight_decay = hparams["weight_decay"],
        amp = True,
    )
    return acc

def objective(train_loader, test_loader, trial: optuna.Trial) -> float:
    hparams = {
        "cycles": trial.suggest_int("cycles", HPARAM_CYCLE_MIN, HPARAM_CYCLE_MAX, step=1),
        "epochs_per_cycle": trial.suggest_int("epochs_per_cycle", HPARAM_EPOCH_CYCLE_MIN, HPARAM_EPOCH_CYCLE_MAX, step=1),
        "lr": trial.suggest_float("lr", HPARAM_LR_MIN, HPARAM_LR_MAX, log=True),
        "weight_decay": trial.suggest_float("weight_decay", HPARAM_WD_MIN, HPARAM_WD_MAX, log=True),  
    }

    trial.report(0.0, step=1) 
    accuracy = run_once(train_loader, test_loader, hparams)
    trial.report(accuracy, step=1)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return accuracy

def init_study(name: str = OPTUNA_NAME, direction="maximize"):
    study = optuna.create_study(
        study_name = name,
        direction = direction,
        sampler = optuna.samplers.TPESampler(multivariate=True),
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    return study

def run(train_loader: DataLoader, test_loader: DataLoader):
    study = init_study()
    wrapper = partial(objective, train_loader, test_loader)
    study.optimize(wrapper, n_trials=OPTUNA_TRIALS, n_jobs=OPTUNA_JOBS, timeout=OPTUNA_TIMEOUT)
    return study


if __name__ == '__main__':
    train_loader, test_loader = process(nrows=300)
    study = run(train_loader, test_loader)
    location = 'optuna/' + OPTUNA_NAME + '.pkl'
    joblib.dump(study, location) 
    print("study saved at location: " + location)