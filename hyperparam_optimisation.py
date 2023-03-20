import torch

# Other files
import utils
from orbis_yf_data import download_all_data
from models.mean_variance_optimisation import MeanVarianceOptimisation
from models.autoencoders import LinearAutoencoder, ConvAutoencoder, ConvLinearAutoEncoder, train_autoencoder
from models.autowarp import AutoWarp

# General imports
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import warnings
import json

# Mean variance optimisation
from pypfopt.expected_returns import mean_historical_return

# Hyperparam optimisation
import optuna

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def objective(trial, model):
    # Params
    input_size = data_valid.shape[1]
    num_epochs = 20

    # Autoencoder hyperparams
    latent_size = trial.suggest_int('latent_size', 5, 50)
    batch_size = trial.suggest_int('batch_size', 10, 50)

    # Autowarp hyperparams
    p = trial.suggest_float('p', 0.1, 0.9)
    autowarp_batch_size = trial.suggest_int('autowarp_batch_size', 10, 50)
    lr = trial.suggest_float('lr', 0.001, 0.1)


    # Risk matrix hyperparams
    C = trial.suggest_float('C', -1, 0)

    # Train autoencoder
    if model == 'Linear + CNN':
        hidden_size1 = trial.suggest_int('hidden_size1', latent_size + 2, input_size)
        hidden_size2 = trial.suggest_int('hidden_size2', latent_size, hidden_size1)
        trained_model = train_autoencoder(ConvLinearAutoEncoder, input_size=input_size, hidden_size=hidden_size1,
                                          hidden_size2=hidden_size2, latent_size=latent_size, num_epochs=num_epochs,
                                          batch_size=batch_size, data=data_valid, verbose=False)

    elif model == 'CNN':
        hidden_size = trial.suggest_int('hidden_size', latent_size, input_size)
        trained_model = train_autoencoder(LinearAutoencoder, input_size=input_size, hidden_size=hidden_size,
                                          latent_size=latent_size, num_epochs=num_epochs, batch_size=batch_size,
                                          data=data_valid, verbose=False)
    elif model == 'Linear':
        hidden_size = trial.suggest_int('hidden_size', latent_size, input_size)
        trained_model = train_autoencoder(ConvAutoencoder, input_size=input_size, hidden_size=hidden_size,
                                          latent_size=latent_size, num_epochs=num_epochs, batch_size=batch_size,
                                          data=data_valid, verbose=False)

    else:
        raise ValueError('Model not found')

    # Autowarp
    learner = AutoWarp(trained_model, data_valid, latent_size=latent_size, p=p,
                       max_iterations=25, batch_size=autowarp_batch_size, lr=lr)
    learner.learn_metric()
    dist_matrix = learner.create_distance_matrix()

    # Setup mean variance optimisation
    e_returns = mean_historical_return(prices_valid_train)
    optimiser = MeanVarianceOptimisation(expected_returns=e_returns, prices=prices_valid_train, solver='OSQP',
                                         weight_bounds=(0, 1))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Get weights
            risk_matrix = optimiser.make_risk_matrix(dist_matrix, C=C)
            weights, train_sr = optimiser.max_sharpe_ratio(risk_matrix=risk_matrix, l2_reg=0)
    except:
        return -np.inf

    valid_sr = utils.calculate_sharpe_ratio(weights, prices_valid_valid)

    # return min(train_sr, valid_sr)
    return train_sr * valid_sr


if __name__ == '__main__':

    # Turn off logging for every trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Setup dates
    start_dates = ['2017-03-01', '2017-09-01', '2018-03-01', '2018-09-01', '2019-03-01']
    date_format = "%Y-%m-%d"
    start_dates = [datetime.datetime.strptime(date, date_format) for date in start_dates]
    valid_dates = [date + relativedelta(years=1) for date in start_dates]
    train_dates = [date + relativedelta(years=2) for date in start_dates]
    end_dates = [date + relativedelta(years=3, months=6) for date in start_dates]

    # Loop through dates
    for i in range(len(start_dates)):

        # Download data
        sp500_ratios = pd.read_excel('data/S&P Ratios.xlsx', index_col=0, sheet_name="Results", usecols='C:CU')
        download_all_data(sp500_ratios, start_dates[i].strftime("%Y-%m-%d"))

        # Get numpy data and prices
        _, prices_valid_train, prices_valid_valid, _ = utils.split_prices(start_date=start_dates[i], valid_date=valid_dates[i], train_date=train_dates[i], end_date=end_dates[i], train_number=200)
        _, data_valid = utils.split_orbis_data(start_date=start_dates[i], valid_date=valid_dates[i], train_date=train_dates[i], returns=True, momentum=True, train_number=200)

        # Sort into a dict
        params = {'Linear': {}, 'CNN': {}, 'Linear + CNN': {}}

        # Optimize
        for m in ['Linear', 'CNN', 'Linear + CNN']:

            # Optuna optimisation
            print(m, start_dates[i])
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, m), n_trials=25, show_progress_bar=True)

            if m != 'Linear + CNN':
                params[m]['autoencoder'] = {'latent_size': study.best_params['latent_size'], 'hidden_size': study.best_params['hidden_size'], 'batch_size': study.best_params['batch_size']}
            else:
                params[m]['autoencoder'] = {'latent_size': study.best_params['latent_size'], 'hidden_size': study.best_params['hidden_size1'], 'hidden_size2': study.best_params['hidden_size2'], 'batch_size': study.best_params['batch_size']}

            params[m]['dist_matrix'] = {'latent_size': study.best_params['latent_size'], 'p': study.best_params['p'],
                                     'max_iterations': 25, 'autowarp_batch_size': study.best_params['autowarp_batch_size'],
                                     'lr': study.best_params['lr']}
            params[m]['risk_matrix'] = {'C': study.best_params['C']}


        # Save best params in a json
        with open(f'params/sp500_{start_dates[i].strftime("%Y-%m-%d")}.json', 'w') as f:
            json.dump(params, f)

        print(study.best_params)
