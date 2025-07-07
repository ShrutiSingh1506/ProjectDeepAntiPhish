#constants.py
TRAIN_DATA = "data/train_email.csv"
TEST_DATA = "data/test_email.csv"

# Model Parameters
DL_BATCH_SIZE = 1024


# Hyperparameter Optimiztion
HPARAM_CYCLE_MIN = 2
HPARAM_CYCLE_MAX = 4
HPARAM_EPOCH_CYCLE_MIN = 3
HPARAM_EPOCH_CYCLE_MAX = 5
HPARAM_LR_MIN = 1e-3
HPARAM_LR_MAX = 1e-2
HPARAM_WD_MIN = 1e-3
HPARAM_WD_MAX = 1e-2

# Optuna Hyperparameters
OPTUNA_NAME = "hyperparams"
OPTUNA_TRIALS = 4
OPTUNA_JOBS = 64
OPTUNA_TIMEOUT = 256