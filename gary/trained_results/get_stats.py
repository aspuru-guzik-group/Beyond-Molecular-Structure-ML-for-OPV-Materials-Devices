import pandas as pd
import numpy as np

import scipy.stats
from sklearn.metrics import r2_score


def r_score(x, y):
    # return np.corrcoef(x,y)[0,1]
    pearson_r = scipy.stats.pearsonr(x, y)[0]
    return pearson_r

def calculate_metric(metric, y_pred, y_true):
    ''' Calculate the specified metric along each column. 
    '''
    # assert y_pred.shape == y_true.shape, 'The shape of prediction and truth not the same.'

    if metric == 'rmse':
        return np.sqrt(mse(y_true.ravel(), y_pred.ravel()))
    elif metric == 'r':
        return r_score(y_true.ravel(), y_pred.ravel())
    elif metric == 'r2':
        return r2_score(y_true.ravel(), y_pred.ravel())
    elif metric == 'spearman':
        return spearman_score(y_true.ravel(), y_pred.ravel())
    elif metric == 'mse':
        return mse(y_true.ravel(), y_pred.ravel())
    elif metric == 'mae':
        return np.mean(np.abs(y_true.ravel() - y_pred.ravel()))
    else:
        raise ValueError('Invalid metric')


models = ['ngboost', 'gp']
features = ['fp', 'mordred']
values = ['FF_percent', 'Jsc_mA_cm_pow_neg2', 'Voc_V']


name_map = {
    'fp': 'DA_FP_radius_3_nbits_512',
    # 'mordred': 'DA_mordred'
}

for m in models:
    for f in features:
        for v in values:
            r_mean, r_std = [], []
            r2_mean, r2_std = [], []
            filename = f'min_{m}_{f}_{v}.csv'

            df = pd.read_csv(filename)

            r2_list = []
            r_list = []
            for i, gdf in df.groupby('split'):

                r2_list.append(r2_score(gdf['y_true'].to_numpy(), gdf['y_pred'].to_numpy()))
                r_list.append(r_score(gdf['y_true'].to_numpy(), gdf['y_pred'].to_numpy()))



            r_std.append(np.std(r_list))
            r_mean.append(np.mean(r_list))
            r2_std.append(np.std(r2_list))
            r2_mean.append(np.mean(r2_list))



            pd.DataFrame({
                'Dataset': 'OPV_Min',
                'num_of_folds': 5,
                'Features': f,
                'Targets': v,
                'Model': m,
                'num_of_data': 5,
                'r_mean': r_mean,
                'r_std': r_std,
                'r2_mean': r2_mean,
                'r2_std': r2_std
            }).to_csv(f'min_{m}_{f}_{v}_ensemble_summary.csv', index=False)


        # import pdb; pdb.set_trace()