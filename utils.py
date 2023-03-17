import numpy as np
from scipy.linalg import eigh
import pandas as pd


def convert_to_numpy(df):
    variables = []
    for v in df.columns.levels[0]:
        variables.append(df[v].to_numpy().T)
    return np.stack(variables, axis=2)


# %%
def split_orbis_data(train_date, valid_date, train_number=200, returns=False, momentum=False):
    data = pd.read_pickle('data/data.pkl')

    if returns is True:
        data['Price'] = (data['Price'] / data['Price'].shift(1)) - 1
        data = data.dropna(axis=0)

    if momentum is False:
        # Drop columns that contain the word momentum
        data = data.drop([c for c in data.columns.levels[0] if 'Momentum' in c], axis=1, level=0)
        data.columns = data.columns.remove_unused_levels()

    # Minmax scale each first level index column
    for v in data.columns.levels[0]:
        data[v] = (data[v] - data[v].min()) / (data[v].max() - data[v].min())

    train = convert_to_numpy(data[data.index < train_date])[:train_number]
    valid_train = convert_to_numpy(data[data.index < valid_date])[train_number:]
    valid_valid = convert_to_numpy(data[(data.index >= valid_date) & (data.index < train_date)])[train_number:]
    test = convert_to_numpy(data[data.index >= train_date])[:train_number]

    return train, valid_train, valid_valid, test


def split_prices(train_date, valid_date, train_number=200):
    data = pd.read_pickle('data/data.pkl')['Price']
    data = data.div(data.iloc[0]).mul(100)

    train = data[data.index < train_date].iloc[:, :train_number]
    valid_train = data[data.index < valid_date].iloc[:, train_number:]
    valid_valid = data[(data.index >= valid_date) & (data.index < train_date)].iloc[:, train_number:]
    valid_valid = valid_valid.div(valid_valid.iloc[0]).mul(100)
    test = data[data.index >= train_date].iloc[:, :train_number]
    test = test.div(test.iloc[0]).mul(100)

    return train, valid_train, valid_valid, test


def calculate_sharpe_ratio(weights, prices, rf_rate, days=252):
    """
    Calculates Sharpe Ratio from prices and weights
    :param weights: portfolio weights
    :param prices: daily prices
    :param rf_rate: risk-free weight (annual)
    :param days: number of days assumed in a year
    :return: sharpe ratio
    """
    returns = (prices @ weights) / (prices @ weights).shift(1) - 1
    avg_return = returns.mean() * days
    std = returns.std() * np.sqrt(days)
    sharpe = (avg_return - rf_rate) / std
    return sharpe.values[0]


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
