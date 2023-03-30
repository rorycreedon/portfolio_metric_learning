import numpy as np
from scipy.linalg import eigh
import pandas as pd
import warnings
import numpy as np


def convert_to_numpy(df):
    variables = []
    for v in df.columns.levels[0]:
        variables.append(df[v].to_numpy().T)
    return np.stack(variables, axis=2)


def split_orbis_data(start_date, valid_date, train_date, train_valid_split=2/3, returns=False, momentum=False):
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
    data = pd.read_pickle('data/sp500_data.pkl')

    # Train number
    train_number = round(train_valid_split*np.unique(data.columns.get_level_values(1)).shape[0])

    # 100 indexing price and volume
    data = data[data.index >= start_date]
    data['Price'] = data['Price'].div(data['Price'].iloc[0]).mul(100)
    data['Volume'] = data['Volume'].div(data['Volume'].iloc[0]).mul(100)

    if returns is True:
        returns = data.xs('Price', axis=1, level=0).pct_change()
        returns.columns = pd.MultiIndex.from_product([['Returns'], returns.columns])
        data = pd.concat([data, returns], axis=1)
        # data['Price Returns'] = (data['Price'] / data['Price'].shift(1)) - 1
        data = data.dropna(axis=0)

    if momentum is False:
        # Drop columns that contain the word momentum
        data = data.drop([c for c in data.columns.levels[0] if 'Momentum' in c], axis=1, level=0)
        data.columns = data.columns.remove_unused_levels()

    # Minmax scale each first level index column
    standardise = lambda group: (group - group.min()) / (group.max() - group.min())
    data = data.groupby(level=0, axis=1).transform(standardise)

    # Split into train and validation
    train = convert_to_numpy(data[data.index < train_date])[:train_number]
    valid_train = convert_to_numpy(data[data.index < valid_date])[train_number:]

    return train, valid_train


def split_prices(start_date, valid_date, train_date, end_date, train_valid_split=2/3):
    """
    Split YahooFinance price data into train, validation and test sets
    :param start_date: start date of the data
    :param valid_date: start date of the validation set
    :param train_date: end date of the training set
    :param end_date: end date of the test set
    :param train_valid_split: Train/valid split
    :return:train, valid_train, valid_valid and test
    """
    data = pd.read_pickle('data/sp500_data.pkl')['Price']

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
    effr['Rate (%)'] = effr['Rate (%)'] / (100*days)

    # Merge with prices to get relevant days
    avg_rf = pd.merge(prices.iloc[:, 0].rename('prices'), effr, how='left', left_index=True, right_index=True)
    avg_rf['Rate (%)'] = avg_rf['Rate (%)'].fillna(method='ffill')

    return avg_rf['Rate (%)'].mean()/100


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
    effr['Rate (%)'] = effr['Rate (%)'] / (100*252)

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
