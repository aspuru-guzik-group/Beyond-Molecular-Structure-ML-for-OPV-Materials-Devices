import numpy as np
import pandas as pd

from argparse import ArgumentParser
import pickle
import optuna

from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error as mse

from ml_for_opvs import utils


def objective(trial, x, y, model):
    # wrapper to optimize hyperparameters
    # split the data with CV
    utils.set_seed()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    # train, val = utils.get_cv_splits(x)
    
    # optimize depending on model
    if model == 'ngboost':
        hp = {
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            # 'n_estimators': trial.suggest_int('n_estimators', 500, 1000, step=50),
        }

        # train return loss (minimize)
        m = NGBRegressor(
            Base = DecisionTreeRegressor(criterion='friedman_mse', max_depth=hp['max_depth']), 
            n_estimators=hp['n_estimators'],
            verbose=False, 
            tol=hp['tol'],
        )
        m.fit(x_train, y_train.ravel(), x_val, y_val.ravel(), early_stopping_rounds=50)
        y_pred = m.predict(x_val)
        return mse(y_pred, y_val.ravel())

    elif model == 'gp':
        hp = {
            'kernel': trial.suggest_categorical()
        }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", action="store", type=str, help="Name of model.")
    parser.add_argument("--feature", action="store", type=str, default='fp', help="Name of feature.")
    parser.add_argument("--dataset", action="store", type=str, default="min", help="Dataset of choice.")
    parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults to 1.")

    FLAGS = parser.parse_args()

    # read in the data
    df = pd.read_csv(f'data/{FLAGS.dataset}.csv')
    with open(f'data/{FLAGS.dataset}_{FLAGS.feature}.pkl', 'rb') as f:
        data = pickle.load(f)
    x_donor = utils.remove_zero_variance(data['donor'])
    x_acceptor = utils.remove_zero_variance(data['acceptor'])
    if FLAGS.feature == 'mordred':
        x_donor = utils.pca_features(x_donor)
        x_acceptor = utils.pca_features(x_acceptor)

    x = np.concatenate((x_donor, x_acceptor), axis=-1)
    y = df[['calc_PCE_percent']].to_numpy()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, x, y, FLAGS.model), n_trials=100, n_jobs=FLAGS.num_workers)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # split the dataset
    # training, testing = utils.get_cv_splits(x)

    # # train model over each split
    # test_metric = []
    # for tr_, ts_ in zip(training, testing):
    #     if FLAGS.model == 'ngboost':
    #         m = NGBRegressor(verbose=False).fit(
    #             x[tr_], y[ts_].ravel(), 
    #             val_data[i][0], val_data[i][1].ravel(), 
    #             early_stopping_rounds=20
    #         )
    #         y_pred = m.predict(test_data[i][0])
    #         test_metric.append(utils.r_score(y_pred, test_data[i][1].reshape(y_pred.shape)))
    # import pdb; pdb.set_trace()


    # import pdb; pdb.set_trace()

    # TODO ... 
    # run the training


    

