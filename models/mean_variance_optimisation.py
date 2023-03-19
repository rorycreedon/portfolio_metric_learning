import numpy as np
import pandas as pd
import utils

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import sample_cov, CovarianceShrinkage, exp_cov
from pypfopt import objective_functions


class MeanVarianceOptimisation:

    def __init__(self, expected_returns, prices, solver="OSQP", weight_bounds=(0, 1)):
        """
        Initialise class.
        :param expected_returns: expected returns for each asset (pandas.DataFrame), shape n_assets x 1
        :param solver: Solver that can be found with `cvxpy.installed_solvers()`, for example: ['ECOS', 'ECOS_BB', 'OSQP', 'SCIPY', 'SCS']
        :param prices: prices for each asset (pandas.DataFrame)
        :param weight_bounds: (min_pct_portfolio, max_pct_portfolio). Usually (0,1), but set min_pct_portfolio<0 to allow short positions
        """
        super(MeanVarianceOptimisation, self).__init__()
        self.solver = solver
        self.prices = prices
        self.weight_bounds = weight_bounds
        self.expected_returns = expected_returns

    def make_risk_matrix(self, dist_matrix, C=-0.5):
        """
        Make risk matrix from distance matrix and standard deviation of prices.
        :param dist_matrix: distance matrix (numpy.array)
        :param C: min correlation (float)
        :return: risk matrix (numpy.array)
        """

        # Standardise distance matrix
        # dist_matrix = np.log(dist_matrix + 1e-8)
        dist_matrix = (dist_matrix - dist_matrix.min()) / (dist_matrix.max() - dist_matrix.min())
        dist_matrix = dist_matrix * (1 - C) - 1
        dist_matrix = - dist_matrix
        assert np.all(dist_matrix >= -1) and np.all(dist_matrix <= 1)

        # Calculate std matrix
        std = self.prices.pct_change().dropna().std() * np.sqrt(252)
        std_matrix = np.outer(std, std)

        # Make risk matrix
        risk_matrix = dist_matrix * std_matrix

        # Make symmetric
        risk_matrix = (risk_matrix + risk_matrix.T) / 2

        # Make PSD if necessary
        risk_matrix = utils.get_near_psd(risk_matrix)

        return risk_matrix

    def benchmark_matrix(self, benchmark_model):
        """
        Calculates the benchmark risk matrix
        :param benchmark_model: the benchmark model being used. Currently have ["Covariance", "Covariance Shrinkage", "Exponentially weighted Covariance"].
        :return: risk matrix
        """
        if benchmark_model == "Covariance":
            risk_matrix = sample_cov(self.prices)
        elif benchmark_model == "Covariance Shrinkage":
            risk_matrix = CovarianceShrinkage(self.prices).ledoit_wolf()
        elif benchmark_model == "Exponentially weighted Covariance":
            risk_matrix = exp_cov(self.prices)
        else:
            raise ValueError("Not in list of benchmarks")

        return risk_matrix

    def setup_efficient_frontier(self, risk_matrix):
        """
        Sets up efficient frontier for mean variance optimisation
        :param risk_matrix: distance matrix (numpy.array)

        :return: Efficient Frontier object
        """
        ef = EfficientFrontier(self.expected_returns, risk_matrix, weight_bounds=self.weight_bounds, solver=self.solver)
        return ef

    def max_sharpe_ratio(self, risk_matrix, days=252, l2_reg=0.5):
        """
        Calculates weights that optimise the Sharpe Ratio and the best Sharpe Ratio
        :param risk_matrix: risk matrix (numpy.array)
        :param days: number of days in a year
        :param l2_reg: L2 regularisation parameter
        :return: optimised weights, sharpe ratio
        """

        ef = self.setup_efficient_frontier(risk_matrix)
        if l2_reg > 0:
            ef.add_objective(objective_functions.L2_reg, gamma=l2_reg)
        ef.max_sharpe(risk_free_rate=utils.average_rf_rate(self.prices, days))
        weights = pd.DataFrame.from_dict(ef.clean_weights(), orient='index', columns=['weights'])
        sharpe_ratio = utils.calculate_sharpe_ratio(weights=weights, prices=self.prices, days=days)
        return weights, sharpe_ratio

    def min_volatility(self, risk_matrix, l2_reg=0.5, days=252):
        """
        Calculates weights that optimise the volatility and the best volatility
        :param risk_matrix: risk matrix (numpy.array)
        :param l2_reg: L2 regularisation parameter
        :return: optimised weights
        """

        ef = self.setup_efficient_frontier(risk_matrix)
        if l2_reg > 0:
            ef.add_objective(objective_functions.L2_reg, gamma=l2_reg)
        ef.min_volatility()
        weights = pd.DataFrame.from_dict(ef.clean_weights(), orient='index', columns=['weights'])
        std = utils.calculate_sd(weights=weights, prices=self.prices, days=days)
        return weights, std
