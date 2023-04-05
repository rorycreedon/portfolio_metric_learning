import torch

# Other files
import utils
from models.autowarp import AutoWarp
from models.mean_variance_optimisation import MeanVarianceOptimisation
from models.autoencoders import LinearAutoencoder, ConvAutoencoder, ConvLinearAutoEncoder, train_autoencoder
from models.fama_french import FamaFrench

# General imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import json
import datetime
from dateutil.relativedelta import relativedelta
import os
import argparse
import yfinance as yf

# Optimisation
from pypfopt.expected_returns import mean_historical_return
from pypfopt.hierarchical_portfolio import HRPOpt

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def load_data(start_date, valid_date, train_date, end_date):

    data_arrays, price_dfs = utils.split_data(start_date=start_date, valid_date=valid_date, train_date=train_date, end_date=end_date, train_valid_split=2 / 3, returns=False, momentum=False)
    data_train = data_arrays[0]

    prices_train = price_dfs[0]
    prices_test = price_dfs[3]


    return prices_train, prices_test, data_train


def load_params(start_date):
    with open(f'params/sp500_{start_date}.json') as f:
        params = json.load(f)

    return params


def train_autoencoders(data_train, num_epochs, params, opt):
    models = {}
    models['Linear'] = train_autoencoder(LinearAutoencoder, input_size=data_train.shape[1],
                                         num_epochs=num_epochs,
                                         data=data_train, verbose=False,
                                         **params['Linear'][opt]['autoencoder'])
    models['CNN'] = train_autoencoder(ConvAutoencoder, input_size=data_train.shape[1],
                                      num_epochs=num_epochs,
                                      data=data_train, verbose=False,
                                      **params['CNN'][opt]['autoencoder'])
    models['Linear + CNN'] = train_autoencoder(ConvLinearAutoEncoder,
                                               input_size=data_train.shape[1],
                                               num_epochs=num_epochs, data=data_train,
                                               verbose=False,
                                               **params['Linear + CNN'][opt]['autoencoder'])
    return models


def calculate_dist_matrix(data_train, models, params, opt):
    dist_matrices = {}
    for model in ['Linear', 'CNN', 'Linear + CNN']:
        print(model, "- calculating distance matrix")
        learner = AutoWarp(models[model], data_train, **params[model][opt]['dist_matrix'])
        learner.learn_metric(verbose=False)
        dist_matrices[model] = learner.create_distance_matrix()

    return dist_matrices


def mvo(dist_matrices, params, prices_train, opt):
    # Empty dict for weights
    weights = {}

    for model in ["Linear", "CNN", "Linear + CNN"]:
        # Setup
        e_returns = mean_historical_return(prices_train)
        optimiser = MeanVarianceOptimisation(expected_returns=e_returns, prices=prices_train, solver='ECOS',
                                             weight_bounds=(0, 1))

        # Get weights
        risk_matrix = optimiser.make_risk_matrix(dist_matrices[model], **params[model][opt]['risk_matrix'])
        if opt == 'volatility':
            weights[model], train_sr = optimiser.min_volatility(risk_matrix=risk_matrix, l2_reg=0)
        elif opt == 'sharpe':
            weights[model], train_sr = optimiser.max_sharpe_ratio(risk_matrix=risk_matrix, l2_reg=0)
        else:
            raise ValueError("Invalid opt")

    # Benchmark models
    for model in ["Covariance", "Covariance Shrinkage", "EW Covariance"]:
        # Setup
        e_returns = mean_historical_return(prices_train)
        optimiser = MeanVarianceOptimisation(expected_returns=e_returns, prices=prices_train, solver='ECOS',
                                             weight_bounds=(0, 1))

        # Get weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            risk_matrix = optimiser.benchmark_matrix(model)
            if opt == 'volatility':
                weights[model], train_sr = optimiser.min_volatility(risk_matrix=risk_matrix, l2_reg=0)
            elif opt == 'sharpe':
                weights[model], train_sr = optimiser.max_sharpe_ratio(risk_matrix=risk_matrix, l2_reg=0)
            else:
                raise ValueError("Invalid opt")

    # Fama-French
    fama_french = FamaFrench(prices_train, file_path='data/F-F_Research_Data_Factors_daily.CSV', n_rows=25419)
    risk_matrix = fama_french.get_covariance_matrix()
    optimiser = MeanVarianceOptimisation(expected_returns=e_returns, prices=prices_train, solver='ECOS',
                                         weight_bounds=(0, 1))
    if opt == 'volatility':
        weights['Fama-French'], train_sr = optimiser.min_volatility(risk_matrix=risk_matrix, l2_reg=0)
    elif opt == 'sharpe':
        weights['Fama-French'], train_sr = optimiser.max_sharpe_ratio(risk_matrix=risk_matrix, l2_reg=0)
    else:
        raise ValueError("Invalid opt")

    return weights


def hrp_weights(prices_train):
    rets = (prices_train / prices_train.shift(1) - 1).dropna()
    hrp = HRPOpt(rets)
    hrp.optimize()
    return pd.DataFrame.from_dict(hrp.clean_weights(), orient='index', columns=['weights'])


def calculate_sharpe_ratio(weights, prices_test, train_date, end_date):
    # Empty dict for sharpe ratios
    sharpe_ratios = {}

    for model in weights.keys():
        sharpe_ratios[model] = utils.calculate_sharpe_ratio(prices=prices_test, weights=weights[model])

    # Add S&P 500
    sharpe_ratios['S&P 500'] = utils.calculate_sp500_sharpe(train_date=train_date, end_date=end_date)

    print(sharpe_ratios)

    return sharpe_ratios


def save_results(weights, sharpe_ratios, prices_test, start_date, train_date, end_date, opt):
    # Weights
    num_stocks = {}
    max_drawdown = {}
    for model in weights.keys():
        num_stocks[model] = np.count_nonzero(weights[model])
        max_drawdown[model] = utils.calculate_max_drawdown(prices=prices_test, weights=weights[model])

    # Add S&P 500
    max_drawdown['S&P 500'] = utils.calculate_sp500_drawdown(train_date=train_date, end_date=end_date)

    results = {'sharpe_ratios': sharpe_ratios, 'num_stocks': num_stocks, 'max_drawdown': max_drawdown}

    with open(f'results/sp500_{start_date}_{opt}.json', 'w') as f:
        json.dump(results, f)


def make_plots(prices_test, weights, start_date, train_date, end_date, opt):
    # Setup fig, ax
    fig, ax = plt.subplots(figsize=(6, 3))

    # Plots
    for model in weights.keys():
        ax.plot(prices_test @ weights[model], label=model)
    sp500 = yf.download('^GSPC', start=train_date, end=end_date, progress=False)['Adj Close']
    ax.plot(sp500, label='S&P 500')

    # Plot all prices in background
    ax.plot(prices_test, alpha=0.05)

    ax.margins(x=0)
    ax.set_ylim([75, 175])
    ax.set_ylabel("Portfolio value (indexed at 100)", fontsize=9)
    ax.grid()
    ax.set_title("Max Sharpe Ratio Portfolios over period 2020-09-01 to 2022-03-01", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.legend(fontsize='x-small', ncols=2)
    fig.autofmt_xdate()

    # Save plot
    fig.tight_layout()
    fig.savefig(f'plots/sp500_{start_date}_{opt}.png', dpi=300, bbox_inches="tight")


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="sharpe")
    args = parser.parse_args()
    if args.opt not in ["sharpe", "volatility"]:
        raise ValueError("Volatility must be either 'sharpe' or 'volatility'")

    # Plotting adjustments
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams.update({'figure.autolayout': True})

    # Make folders if necessary
    if not os.path.exists('plots'):
        os.mkdir('plots')
    if not os.path.exists('results'):
        os.mkdir('results')

    # Setup dates
    start_dates = ['2017-03-01', '2017-09-01', '2018-03-01', '2018-09-01']
    date_format = "%Y-%m-%d"
    start_dates = [datetime.datetime.strptime(date, date_format) for date in start_dates]
    valid_dates = [date + relativedelta(years=2) for date in start_dates]
    train_dates = [date + relativedelta(years=3) for date in start_dates]
    end_dates = [date + relativedelta(years=4, months=6) for date in start_dates]

    for i in range(len(start_dates)):
        print(f"Start date: {start_dates[i].strftime(date_format)}")
        print(f"Opt - {args.opt}")
        print("========================================")

        # Load params
        params = load_params(start_date=start_dates[i].strftime(date_format))

        # Load data
        prices_train, prices_test, data_train = load_data(start_date=start_dates[i].strftime(date_format),
                                                         valid_date=valid_dates[i].strftime(date_format),
                                                         train_date=train_dates[i].strftime(date_format),
                                                         end_date=end_dates[i].strftime(date_format))

        # Train autoencoders
        models = train_autoencoders(data_train=data_train, num_epochs=20, params=params, opt=args.opt)

        # Calculate distance matrices
        dist_matrices = calculate_dist_matrix(data_train=data_train, models=models, params=params, opt=args.opt)

        # MVO
        weights = mvo(dist_matrices=dist_matrices, params=params, prices_train=prices_train, opt=args.opt)

        # HRP
        weights['HRP'] = hrp_weights(prices_train=prices_train)

        # Calculate sharpe ratios
        sharpe_ratios = calculate_sharpe_ratio(weights=weights, prices_test=prices_test, train_date=train_dates[i], end_date=end_dates[i])

        # Save weights
        save_results(weights=weights, sharpe_ratios=sharpe_ratios, prices_test=prices_test, start_date=start_dates[i].strftime(date_format), train_date=train_dates[i], end_date=end_dates[i], opt=args.opt)

        # Make plots
        make_plots(prices_test=prices_test, weights=weights, start_date=start_dates[i].strftime(date_format), train_date=train_dates[i], end_date=end_dates[i], opt=args.opt)
