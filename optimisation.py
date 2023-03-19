import torch

# Other files
import utils
from models.mean_variance_optimisation import MeanVarianceOptimisation
from models.autoencoders import LinearAutoencoder, ConvAutoencoder, ConvLinearAutoEncoder, train_autoencoder, \
    get_distance_matrix
from models.autowarp import AutoWarp

# General imports
import numpy as np
import warnings

# Mean variance optimisation
from pypfopt.expected_returns import mean_historical_return

# Hyperparam optimisation
import optuna

# Set seed for reproducibility
torch.manual_seed(0)

# %%
valid_date = '2018-09-01'
train_date = '2020-01-01'

# Download data
_, prices_valid_train, prices_valid_valid, _ = utils.split_prices(train_date=train_date, valid_date=valid_date)
_, data_valid_valid, _, _ = utils.split_orbis_data(train_date=train_date, valid_date=valid_date, returns=True, momentum=True)


def objective(trial, model):
    # Params
    input_size = data_valid_valid.shape[1]
    num_epochs = 20

    # Hyperparams
    latent_size = trial.suggest_int('latent_size', 5, 50)
    batch_size = trial.suggest_int('batch_size', 10, 50)
    # distance_metric = 'euclidean'
    # distance_metric = trial.suggest_categorical('distance_metric', ['euclidean', 'soft_dtw_normalized'])
    # if distance_metric == 'soft_dtw_normalized':
    #     gamma = trial.suggest_float('gamma', 0, 1)
    C = trial.suggest_float('C', -1, 0)

    # Train autoencoder
    if model == 'Linear + CNN':
        hidden_size1 = trial.suggest_int('hidden_size1', latent_size + 2, input_size)
        hidden_size2 = trial.suggest_int('hidden_size2', latent_size, hidden_size1)
        trained_model = train_autoencoder(ConvLinearAutoEncoder, input_size=input_size, hidden_size=hidden_size1,
                                          hidden_size2=hidden_size2, latent_size=latent_size, num_epochs=num_epochs,
                                          batch_size=batch_size, data=data_valid_valid, verbose=False)

    elif model == 'CNN':
        hidden_size = trial.suggest_int('hidden_size', latent_size, input_size)
        trained_model = train_autoencoder(LinearAutoencoder, input_size=input_size, hidden_size=hidden_size,
                                          latent_size=latent_size, num_epochs=num_epochs, batch_size=batch_size,
                                          data=data_valid_valid, verbose=False)
    elif model == 'Linear':
        hidden_size = trial.suggest_int('hidden_size', latent_size, input_size)
        trained_model = train_autoencoder(ConvAutoencoder, input_size=input_size, hidden_size=hidden_size,
                                          latent_size=latent_size, num_epochs=num_epochs, batch_size=batch_size,
                                          data=data_valid_valid, verbose=False)

    else:
        raise ValueError('Model not found')

    # Calculate distance matrix
    # if distance_metric == 'euclidean':
    #     dist_matrix = get_distance_matrix(trained_model, data_valid_valid, latent_size=latent_size,
    #                                       distance_metric=distance_metric)
    # else:
    #     dist_matrix = get_distance_matrix(trained_model, data_valid_valid, latent_size=latent_size,
    #                                       distance_metric=distance_metric, gamma=gamma)

    # Autowarp
    learner = AutoWarp(trained_model, data_valid_valid, latent_size=latent_size, p=0.2,
                       max_iterations=50, batch_size=25, lr=0.01)
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
    # Create an Optuna study and run the optimization
    #optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Optimize
    #for m in ['Linear', 'CNN', 'Linear + CNN']:
    for m in ['Linear']:
        print(m)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, m), n_trials=50, show_progress_bar=True)
        print(study.best_params)
