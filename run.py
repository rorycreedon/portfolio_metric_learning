import torch

# Other files
import utils
from models.autowarp import AutoWarp
from models.mean_variance_optimisation import MeanVarianceOptimisation
from models.autoencoders import LinearAutoencoder, ConvAutoencoder, ConvLinearAutoEncoder, train_autoencoder
from orbis_yf_data import download_all_data

# General imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import json
import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import os

# Optimisation
from pypfopt.expected_returns import mean_historical_return
from pypfopt.hierarchical_portfolio import HRPOpt

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def load_data(start_date, valid_date, train_date, end_date):

    prices_train, _, _, prices_test = utils.split_prices(start_date=start_date, valid_date=valid_date,
                                                         train_date=train_date, end_date=end_date, train_valid_split=2/3)
    data_train, _ = utils.split_orbis_data(start_date=start_date, valid_date=valid_date, train_date=train_date,
                                           returns=True, momentum=True, train_valid_split=2/3)

    return prices_train, prices_test, data_train


def load_params(start_date):

    with open(f'params/sp500_{start_date}.json') as f:
        params = json.load(f)
    return params


def train_autoenoders(data_train, num_epochs, params):

    models = {}
    models['Linear'] = train_autoencoder(LinearAutoencoder, input_size=data_train.shape[1], num_epochs=num_epochs,
                                         data=data_train, verbose=False, **params['Linear']['autoencoder'])
    models['CNN'] = train_autoencoder(ConvAutoencoder, input_size=data_train.shape[1], num_epochs=num_epochs,
                                      data=data_train, verbose=False, **params['CNN']['autoencoder'])
    models['Linear + CNN'] = train_autoencoder(ConvLinearAutoEncoder, input_size=data_train.shape[1],
                                               num_epochs=num_epochs, data=data_train, verbose=False,
                                               **params['Linear + CNN']['autoencoder'])
    return models


def calculate_dist_matrix(data_train, models, params):

    dist_matrices = {}
    for model in ['Linear', 'Linear + CNN', 'CNN']:
        print(model, " - calculating distance matrix")
        learner = AutoWarp(models[model], data_train, **params[model]['dist_matrix'])
        learner.learn_metric()
        dist_matrices[model] = learner.create_distance_matrix()

    return dist_matrices


def mvo(dist_matrices, params, prices_train):

    # Empty dict for weights
    weights = {}

    for model in ["Linear", "CNN", "Linear + CNN"]:
        # Setup
        e_returns = mean_historical_return(prices_train)
        optimiser = MeanVarianceOptimisation(expected_returns=e_returns, prices=prices_train, solver='OSQP',
                                             weight_bounds=(0, 1))

        # Get weights
        print(model, "making risk matrix")
        risk_matrix = optimiser.make_risk_matrix(dist_matrices[model], **params[model]['risk_matrix'])
        weights[model], train_sr = optimiser.max_sharpe_ratio(risk_matrix=risk_matrix, l2_reg=0)

    for model in ["Covariance", "Covariance Shrinkage", "Exponentially Weighted Covariance"]:
        # Setup
        e_returns = mean_historical_return(prices_train)
        optimiser = MeanVarianceOptimisation(expected_returns=e_returns, prices=prices_train, solver='OSQP',
                                             weight_bounds=(0, 1))

        # Get weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            risk_matrix = optimiser.benchmark_matrix(model)
            weights[model], train_sr = optimiser.max_sharpe_ratio(risk_matrix=risk_matrix, l2_reg=0)

    # Equal weights
    weights['Equal'] = weights['Linear'].copy()
    weights['Equal']['weights'] = 1 / weights['Linear']['weights'].shape[0]

    return weights


def hrp_weights(prices_train):

    rets = (prices_train / prices_train.shift(1) - 1).dropna()
    hrp = HRPOpt(rets)
    hrp.optimize()
    return pd.DataFrame.from_dict(hrp.clean_weights(), orient='index', columns=['weights'])


def calculate_sharpe_ratio(weights, prices_test, train_date, end_date):

    # Empty dict for sharpe ratios
    sharpe_ratios = {}

    for model in ["Linear", "CNN", "Linear + CNN", "Covariance", "Covariance Shrinkage", "Exponentially Weighted Covariance", "HRP", "Equal"]:
        sharpe_ratios[model] = utils.calculate_sharpe_ratio(prices=prices_test, weights=weights[model])

    # Download S&P 500 data
    sp500 = yf.download("^GSPC", start=train_date, end=end_date, period="1d", progress=False)['Adj Close']
    sp500 = sp500.div(sp500.iloc[0]).mul(100)

    # Download EFFR data
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        effr = pd.read_excel('data/EFFR.xlsx', sheet_name='Results', index_col=0, usecols='A:C', engine="openpyxl")
    effr = effr.drop(columns='Rate Type', axis=1)
    effr.index = pd.to_datetime(effr.index)
    effr['Rate (%)'] = effr['Rate (%)'] / (100 * 252)

    # S&P 500 sharpe ratio
    sp500_returns = ((sp500 / sp500.shift(1)) - 1).dropna()
    sp500_returns = pd.merge(sp500_returns, effr, how='left', left_index=True, right_index=True)
    sp500_returns['Rate (%)'] = sp500_returns['Rate (%)'].fillna(method='ffill')
    assert sp500_returns.isnull().sum().sum() == 0
    sp500_returns['excess_returns'] = sp500_returns['Adj Close'] - sp500_returns['Rate (%)']
    avg_return = sp500_returns['excess_returns'].mean() * 252
    std = sp500_returns['excess_returns'].std() * np.sqrt(252)
    sharpe_ratios['S&P 500'] = avg_return / std

    return sharpe_ratios


def save_results(weights, sharpe_ratios, start_date):

    # Weights
    weights_copy = {}
    num_stocks = {}
    for model in weights.keys():
        weights_copy[model] = pd.DataFrame.to_dict(weights[model])
        num_stocks[model] = np.count_nonzero(weights[model])

    results = {'sharpe_ratios': sharpe_ratios, 'num_stocks': num_stocks, 'weights': weights_copy}

    with open(f'results/sp500_{start_dates[i].strftime("%Y-%m-%d")}.json', 'w') as f:
        json.dump(results, f)


def make_plots(prices_test, weights, start_date, train_date, end_date):

    # Setup fig, ax
    fig, ax = plt.subplots(figsize=(6, 4))

    # S&P 500
    sp500 = yf.download("^GSPC", start=train_date, end=end_date, period="1d", progress=False)['Adj Close']
    sp500 = sp500.div(sp500.iloc[0]).mul(100)

    # Plots
    ax.plot(prices_test, alpha=0.05)
    for model in ["Linear", "CNN", "Linear + CNN", "Covariance", "Covariance Shrinkage", "Exponentially Weighted Covariance", "Equal"]:
        ax.plot(prices_test @ weights[model], label=model)
    ax.plot(sp500, label='S&P 500')

    ax.set_ylim([50, 200])
    ax.grid()
    ax.legend(fontsize='x-small')
    ax.tick_params(axis='x', rotation=45)

    # Save plot
    fig.savefig(f'plots/sp500_{start_date}.png', dpi=300, bbox_inches = "tight")


if __name__ == '__main__':

    # Make folders if necessary
    if not os.path.exists('plots'):
        os.mkdir('plots')
    if not os.path.exists('results'):
        os.mkdir('results')

    # Setup dates
    start_dates = ['2017-03-01', '2017-09-01', '2018-03-01', '2018-09-01', '2019-03-01']
    date_format = "%Y-%m-%d"
    start_dates = [datetime.datetime.strptime(date, date_format) for date in start_dates]
    valid_dates = [date + relativedelta(years=1) for date in start_dates]
    train_dates = [date + relativedelta(years=2) for date in start_dates]
    end_dates = [date + relativedelta(years=3, months=6) for date in start_dates]

    for i in range(len(start_dates)):

        print(f"Start date: {start_dates[i].strftime(date_format)}")
        print("========================================")

        # Download data
        sp500_ratios = pd.read_excel('data/S&P Ratios.xlsx', index_col=0, sheet_name="Results", usecols='C:CU')
        download_all_data(sp500_ratios, start_dates[i].strftime("%Y-%m-%d"))

        # Load data
        prices_train, prices_test, data_train = load_data(start_date=start_dates[i], valid_date=valid_dates[i], train_date=train_dates[i], end_date=end_dates[i])

        # Load params
        params = load_params(start_date=start_dates[i].strftime(date_format))

        # Train autoencoders
        models = train_autoenoders(data_train=data_train, num_epochs=20, params=params)

        # Calculate distance matrices
        dist_matrices = calculate_dist_matrix(data_train=data_train, models=models, params=params)

        # MVO
        weights = mvo(dist_matrices=dist_matrices, params=params, prices_train=prices_train)

        # HRP
        weights['HRP'] = hrp_weights(prices_train=prices_train)

        # Calculate sharpe ratios
        sharpe_ratios = calculate_sharpe_ratio(weights=weights, prices_test=prices_test, train_date=train_dates[i], end_date=end_dates[i])

        # Save weights
        save_results(weights=weights, sharpe_ratios=sharpe_ratios, start_date=start_dates[i].strftime(date_format))

        # Make plots
        make_plots(prices_test=prices_test, weights=weights, start_date=start_dates[i].strftime(date_format), train_date=train_dates[i], end_date=end_dates[i])
