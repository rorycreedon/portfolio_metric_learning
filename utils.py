import numpy as np


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


def is_PSD(matrix):
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
    if is_PSD(matrix):
        return matrix
    else:
        C = (matrix + matrix.T) / 2
        eigval, eigvec = np.linalg.eig(C)
        eigval[eigval < 0] = 0
        print("matrix fixed for PSD")

        return eigvec.dot(np.diag(eigval)).dot(eigvec.T)
