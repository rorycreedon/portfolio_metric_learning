import numpy as np
from scipy.linalg import eigh
import pandas as pd
import warnings
from datetime import datetime
import yfinance as yf
from dateutil.relativedelta import relativedelta


def convert_to_numpy(df):
    variables = []
    for v in df.columns.levels[0]:
        variables.append(df[v].to_numpy().T)
    return np.stack(variables, axis=2)


def split_data(start_date, valid_date, train_date, end_date, train_valid_split=2 / 3, returns=False, momentum=False):
    """
    Split Orbis and price data into train, validation and test sets
    :param start_date: start date of data
    :param valid_date: validation start date
    :param train_date: train start date
    :param end_date: end of test set
    :param train_valid_split: train/valid split
    :param returns: Whether to include returns
    :param momentum: Whether to include momentum
    :return: data, a list of data_train, data_valid_train and prices, a list of price_train, price_valid_train, price_valid_valid, price_test
    """
    # Clean data
    data = clean_orbis_data(start_date=start_date, train_date=train_date, returns=returns, momentum=momentum)

    # Train number
    train_number = round(train_valid_split * np.unique(data.columns.get_level_values(1)).shape[0])

    # Check if all price data needed is included in data
    if data.index[-1] >= datetime.strptime(end_date, "%Y-%m-%d"):
        price_data = pd.read_pickle(f'data/sp500_data_{train_date}.pkl')['Price']
    else:
        # Download price data for test set
        start_price_date = datetime.strptime(start_date, "%Y-%m-%d")
        tickers = np.unique(data.columns.get_level_values(1))
        while True:
            try:
                price_data = yf.download(list(tickers), start=start_price_date, end=end_date, progress=False)['Adj Close']
                break
            except:
                start_price_date += relativedelta(days=1)

        # Check if any tickers are missing
        if len(np.unique(data.columns.get_level_values(1))) != price_data.shape[1]:
            # Drop missing tickers from data
            missing_tickers = list(set(np.unique(data.columns.get_level_values(1))) - set(price_data.columns))
            data = data.drop(missing_tickers, axis=1, level=1)

    # Split into train and validation
    data_train = convert_to_numpy(data[data.index < train_date])[:train_number]
    data_valid_train = convert_to_numpy(data[data.index < valid_date])[train_number:]

    # Clean up price data
    price_train, price_valid_train, price_valid_valid, price_test = split_prices(data=price_data, start_date=start_date, valid_date=valid_date, train_date=train_date, end_date=end_date, train_valid_split=train_valid_split)

    data = [data_train, data_valid_train]
    prices = [price_train, price_valid_train, price_valid_valid, price_test]

    return data, prices


def clean_orbis_data(start_date, train_date, returns=False, momentum=False):
    """
    Split Orbis data into train and validation
    :param start_date: start date of the data
    :param valid_date: start date of the validation set
    :param train_date: end date of the training set
    :param train_valid_split: Train/valid split
    :param returns: Whether prices or returns should be used
    :param momentum: Whether momentum features should be used
    :return: train, valid_train, valid_valid
    """
    data = pd.read_pickle(f'data/sp500_data_{train_date}.pkl')

    # 100 indexing price and volume
    data = data[data.index >= start_date]
    data['Price'] = data['Price'].div(data['Price'].iloc[0]).mul(100)
    data['Volume'] = data['Volume'].div(data['Volume'].iloc[0]).mul(100)

    if returns is True:
        returns = data.xs('Price', axis=1, level=0).pct_change()
        returns.columns = pd.MultiIndex.from_product([['Returns'], returns.columns])
        data = pd.concat([data, returns], axis=1)
        data = data.dropna(axis=0)

    if momentum is False:
        # Drop columns that contain the word momentum
        data = data.drop([c for c in data.columns.levels[0] if 'Momentum' in c], axis=1, level=0)
        data.columns = data.columns.remove_unused_levels()

    # Minmax scale each first level index column
    standardise = lambda group: (group - group.min()) / (group.max() - group.min()) if (group.max() - group.min()) != 0 else group * 0
    data = data.groupby(level=0, axis=1).transform(standardise)

    assert data.isna().sum().sum() == 0, 'NaNs in data after standardisation'

    return data


def split_prices(data, start_date, valid_date, train_date, end_date, train_valid_split=2 / 3):
    """
    Split YahooFinance price data into train, validation and test sets
    :param data: YahooFinance price data
    :param start_date: start date of the data
    :param valid_date: start date of the validation set
    :param train_date: end date of the training set
    :param end_date: end date of the test set
    :param train_valid_split: Train/valid split
    :return:train, valid_train, valid_valid and test
    """

    # Train number
    train_number = round(train_valid_split * data.shape[-1])

    # 100 indexing price and volume
    data = data[data.index >= start_date]
    data = data.div(data.iloc[0]).mul(100)

    train = data[data.index < train_date].iloc[:, :train_number]
    valid_train = data[data.index < valid_date].iloc[:, train_number:]
    valid_valid = data[(data.index >= valid_date) & (data.index < train_date)].iloc[:, train_number:]
    valid_valid = valid_valid.div(valid_valid.iloc[0]).mul(100)
    test = data[(data.index >= train_date) & (data.index <= end_date)].iloc[:, :train_number]
    test = test.div(test.iloc[0]).mul(100)

    return train, valid_train, valid_valid, test


def average_rf_rate(prices, days):
    """
    Calculates the average risk free rate from the EFFR data
    :param prices: daily prices
    :param days: number of days assumed in a year
    :return: average risk free rate
    """
    # Load EFFR (risk free rate)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        effr = pd.read_excel('data/EFFR.xlsx', sheet_name='Results', index_col=0, usecols='A:C', engine="openpyxl")
    effr = effr.drop(columns='Rate Type', axis=1)
    effr.index = pd.to_datetime(effr.index)
    effr['Rate (%)'] = effr['Rate (%)'] / (100 * days)

    # Merge with prices to get relevant days
    avg_rf = pd.merge(prices.iloc[:, 0].rename('prices'), effr, how='left', left_index=True, right_index=True)
    avg_rf['Rate (%)'] = avg_rf['Rate (%)'].fillna(method='ffill')

    return avg_rf['Rate (%)'].mean() / 100


def calculate_sharpe_ratio(weights, prices, days=252):
    """
    Calculates Sharpe Ratio from prices and weights
    :param weights: portfolio weights
    :param prices: daily prices
    :param days: number of days assumed in a year
    :return: sharpe ratio
    """

    # Load EFFR (risk free rate)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        effr = pd.read_excel('data/EFFR.xlsx', sheet_name='Results', index_col=0, usecols='A:C', engine="openpyxl")
    effr = effr.drop(columns='Rate Type', axis=1)
    effr.index = pd.to_datetime(effr.index)
    effr['Rate (%)'] = effr['Rate (%)'] / (100 * 252)

    # Calculate returns
    returns = (prices @ weights) / (prices @ weights).shift(1) - 1
    returns = returns.dropna(axis=0)

    # Calculate excess returns
    returns = pd.merge(returns, effr, how='left', left_index=True, right_index=True)
    returns['Rate (%)'] = returns['Rate (%)'].fillna(method='ffill')
    assert returns.isnull().sum().sum() == 0
    returns['excess_returns'] = returns['weights'] - returns['Rate (%)']

    # Calculate Sharpe Ratio
    avg_return = returns['excess_returns'].mean() * days
    std = returns['excess_returns'].std() * np.sqrt(days)
    sharpe = avg_return / std
    return sharpe


def calculate_max_drawdown(prices, weights):
    """
    Calculates the maximum drawdown of a portfolio
    :param prices: Price time series
    :param weights: Weights for each stock
    :return: Maximum drawdown (%)
    """
    portfolio = prices @ weights
    portfolio['running_max'] = portfolio['weights'].cummax()
    portfolio['drawdown'] = portfolio['running_max'] - portfolio['weights']
    max_drawdown = portfolio['drawdown'].max()
    max_drawdown_pct = max_drawdown / portfolio.loc[portfolio['drawdown'].idxmax(), 'running_max']
    return max_drawdown_pct


def calculate_sp500_sharpe(train_date, end_date):
    sp500 = yf.download("^GSPC", start=train_date, end=end_date, period="1d", progress=False)['Adj Close']
    sp500 = sp500.pct_change().dropna()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        effr = pd.read_excel('data/EFFR.xlsx', sheet_name='Results', index_col=0, usecols='A:C', engine="openpyxl")
    effr = effr.drop(columns='Rate Type', axis=1)
    effr.index = pd.to_datetime(effr.index)
    effr['Rate (%)'] = effr['Rate (%)'] / (100 * 252)

    returns = pd.merge(sp500, effr, left_index=True, right_index=True, how='left')
    returns['excess_returns'] = returns['Adj Close'] - returns['Rate (%)']
    assert returns['Rate (%)'].isna().sum()
    sharpe_ratio = returns['excess_returns'].mean() / returns['excess_returns'].std()

    return sharpe_ratio


def calculate_sp500_drawdown(train_date, end_date):
    sp500 = pd.DataFrame(yf.download("^GSPC", start=train_date, end=end_date, period="1d", progress=False)['Adj Close'])
    sp500['running_max'] = sp500['Adj Close'].cummax()
    sp500['drawdown'] = sp500['running_max'] - sp500['Adj Close']
    max_drawdown = sp500['drawdown'].max()
    max_drawdown_pct = max_drawdown / sp500.loc[sp500['drawdown'].idxmax(), 'running_max']

    return max_drawdown_pct


def calculate_sd(weights, prices, days=252):
    """
    Calculates variance from prices and weights
    :param weights: portfolio weights
    :param prices: daily prices
    :param days: number of days assumed in a year
    :return: standard deviation
    """
    returns = ((prices @ weights) / (prices @ weights).shift(1)) - 1
    std = returns.std() * np.sqrt(days)
    return std[0]


def is_psd(matrix):
    """
    Check if a matrix is positive semi-definite
    :param matrix: a matrix (numpy.array)
    :return: True/False if matrix is PSD
    """
    # Compute eigenvalues of the matrix
    eigvals = np.linalg.eigvals(matrix)

    # Check if all eigenvalues are non-negative
    if np.all(eigvals >= 0):
        return True
    else:
        return False


def get_near_psd(matrix):
    """
    Find nearest PSD matrix if matrix if not PSD, else return original matrix
    :param matrix: a matrix (numpy.array)
    :return: A PSD matrix (numpy.array)
    """
    if is_psd(matrix):
        return matrix
    else:
        # Get the symmetric part of the distance matrix
        sym_dist_matrix = 0.5 * (matrix + matrix.T)

        # Compute the eigenvalues and eigenvectors of the symmetric distance matrix
        eig_vals, eig_vecs = eigh(sym_dist_matrix)

        # Set negative eigenvalues to zero
        eig_vals[eig_vals < 0] = 0

        # Construct the nearest semi-positive definite matrix
        nearest_spd_matrix = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T

        # Ensure that the matrix does not contain complex numbers
        nearest_spd_matrix = np.real_if_close(nearest_spd_matrix)

        return nearest_spd_matrix
