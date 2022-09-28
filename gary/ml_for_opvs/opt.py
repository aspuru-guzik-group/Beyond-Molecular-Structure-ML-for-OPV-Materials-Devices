import optuna
import ngboost

from sklearn.tree import DecisionTreeRegressor

NGB_HPARAMS = {
    'max_depth': trial.suggest_int("max_depth", 3, 6)
}
