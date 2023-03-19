import numpy as np
from scipy.linalg import eigh
import pandas as pd
import warnings


def convert_to_numpy(df):
    variables = []
    for v in df.columns.levels[0]:
        variables.append(df[v].to_numpy().T)
    return np.stack(variables, axis=2)


# %%
def split_orbis_data(train_date, valid_date, train_number=200, returns=False, momentum=False):
    data = pd.read_pickle('data/sp500_data.pkl')

    if returns is True:
        data['Price'] = (data['Price'] / data['Price'].shift(1)) - 1
        data = data.dropna(axis=0)

    if momentum is False:
        # Drop columns that contain the word momentum
        data = data.drop([c for c in data.columns.levels[0] if 'Momentum' in c], axis=1, level=0)
        data.columns = data.columns.remove_unused_levels()

    # Minmax scale each first level index column
    standardise = lambda group: (group - group.min()) / (group.max() - group.min())
    data = data.groupby(level=0, axis=1).transform(standardise)

    # Split into test, train, valid
    train = convert_to_numpy(data[data.index < train_date])[:train_number]
    valid_train = convert_to_numpy(data[data.index < valid_date])[train_number:]
    valid_valid = convert_to_numpy(data[(data.index >= valid_date) & (data.index < train_date)])[train_number:]
    test = convert_to_numpy(data[data.index >= train_date])[:train_number]

    return train, valid_train, valid_valid, test


def split_prices(train_date, valid_date, train_number=200):
    data = pd.read_pickle('data/sp500_data.pkl')['Price']
    data = data.div(data.iloc[0]).mul(100)

    train = data[data.index < train_date].iloc[:, :train_number]
    valid_train = data[data.index < valid_date].iloc[:, train_number:]
    valid_valid = data[(data.index >= valid_date) & (data.index < train_date)].iloc[:, train_number:]
    valid_valid = valid_valid.div(valid_valid.iloc[0]).mul(100)
    test = data[data.index >= train_date].iloc[:, :train_number]
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
        # C = (matrix + matrix.T) / 2
        # eigval, eigvec = np.linalg.eig(C)
        # eigval[eigval < 0] = 0
        # print("matrix fixed for PSD")
        #
        # return np.real(eigvec.dot(np.diag(eigval)).dot(eigvec.T))

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
