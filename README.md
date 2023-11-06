# Learning a Multi-Factor Covariance Matrix for Portfolio Optimisation

This repository contains the code for my COMP0162 Advanced Machine Learning in Finance report.

The link to the paper is [here](Report.pdf). I have also written a (potentially easier to read) summary [here](https://rorycreedon.github.io/projects/2023/portolio-autoencoders/)

## Abstract

Whilst mean-variance optimisation finds the optimal in-sample portfolio, it typically exhibits poor out-of-sample performance. This paper proposes a novel methodology to incorporate multiple factors (such as historical profit ratios) into constructing a covariance matrix for mean-variance optimisation, inspired by Autowarp [[1](https://arxiv.org/pdf/1810.10107.pdf)]. An autoencoder is first trained on features relevant to each asset. Secondly, an automatically selected distance metric is used to measure the distance between the latent representations for each pair of assets. Finally, this distance matrix is standardised and multiplied by the standard deviations of returns to create a covariance matrix. Using a selection of the constituent stocks in the S&P 500, this method can lead to much improved out-of-sample performance compared to other covariance matrices, such as the sample covariance matrix, covariance shrinkage and the exponentially weighted covariance matrix.

## Project Structure

The repository is structured as follows:

- `data/` contains the data used for the project.
  - `orbis_yf_data.py` downloads the relevant price data from Yahoo! Finance and merges in other data listed below 
  - `EFFR.xlsx` contains data on the Effective Fed Funds Rate, assumed to be the risk-free rate
  - `F-F_Research_Data_Factors_daily.csv` contains data on the Fama-French factors in the Fama-French three-factor model. This file can also be downloaded from Kenneth French's [website](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#Research).
  - `S&P Ratios.xlsx` contains a number of financial ratios downloaded from Orbis for stocks within the S&P 500.
  - `S&P Tickers.csv` contains the tickers in the S&P 500 by day.
  - `sp500_data_{train_date}.pkl` files are created by running `data/orbis_yf_data.py`.

- `models/` contains the code for the models used in the project, which are listed as follows:
  - `autoencoders.py` contains classes for each of the autoencoders used in the report.
  - `autowarp.py` is an implementation of the [Autowarp](https://arxiv.org/abs/1810.10107) algorithm.
  - `fama_french.py` contains a class to produce a covariance matrix using the Fama-French three-factor model.
  - `mean_variance_optimisation.py` contains a class to conduct mean variance optimisation.
- `params/` contains optimsied hyperparameters
- `plots/` contains plots produced and included in the report
- `results/` contains results, showing the Sharpe Ratio, max drawdown and number of assets in the portfolio in the test set
- `demo.ipynb` contains a demo of the model produced and produces some plots for all results
- `environment.yml` is the YAML file for the Conda environment
- `hyperparam_optimisation.py` contains code to conduct Bayesian Optimisation for hyperparameter tuning.
- `Report.pdf` is the paper.
- `run.py` contains code to re-produce all the results in the paper
- `utils.py` contains helper functions used in the repository

## Notes
- This repository downloads data using the [`yfinance`](https://github.com/ranaroussi/yfinance) package. As data is scraped from Yahoo Finance, this can occassionally cause issues as Yahoo has begun encrypting the data that is scraped (see News section on the [`yfinance`](https://github.com/ranaroussi/yfinance) GitHub page).
