import numpy as np
import pandas as pd
import utils

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import sample_cov, CovarianceShrinkage
from pypfopt import objective_functions # leaving this here for when implementing transaction costs


class MeanVarianceOptimisation:

    def __init__(self, expected_returns, prices, solver="OSQP", weight_bounds=(0, 1)):
        """
        Initialise class.
        :param expected_returns: expected returns for each asset (pandas.DataFrame), shape n_assets x 1
        :param risk_matrix: risk matrix for each asset, shape n_assets x n_assets
        :param solver: Solver that can be found with `cvxpy.installed_solvers()`, for example: ['ECOS', 'ECOS_BB', 'OSQP', 'SCIPY', 'SCS']
        :param prices: prices for each asset (pandas.DataFrame)
        :param weight_bounds: (min_pct_portfolio, max_pct_portfolio). Usually (0,1), but set min_pct_portfolio<0 to allow short positions
        """
        super(MeanVarianceOptimisation, self).__init__()
        self.solver = solver
        self.prices = prices
        self.weight_bounds = weight_bounds
        self.expected_returns = expected_returns

    def make_risk_matrix(self, dist_matrix, C=1):
        """
        Make risk matrix from distance matrix and standard deviation of prices (WOULD PRICES BE BETTER?).
        :param dist_matrix: distance matrix (numpy.array)
        :param C: regularisation parameter (int)
        :return: risk matrix (numpy.array)
        """

        # Log distance_matrix and scale to [0,1]
        dist_matrix = np.log(dist_matrix + 1e-8)
        dist_matrix = (dist_matrix - np.min(dist_matrix)) / (np.max(dist_matrix) - np.min(dist_matrix))

        # Invert (as risk matrix should show dissimilarity)
        dist_matrix = 1 - dist_matrix

        # Calculate std matrix
        std = self.prices.std(axis=0)
        std = np.sqrt(std)
        std_matrix = np.outer(std, std)
        std_matrix = (std_matrix - np.min(std_matrix)) / (np.max(std_matrix) - np.min(std_matrix))

        # Make risk matrix
        risk_matrix = (C * dist_matrix) + std_matrix

        # Make symmetric
        risk_matrix = (risk_matrix + risk_matrix.T) / 2

        # Make PSD if necessary
        risk_matrix = utils.get_near_psd(risk_matrix)

        return risk_matrix

    def setup_efficient_frontier(self, is_benchmark, benchmark_model, dist_matrix=None, C=1):
        """
        Sets up efficient frontier for mean variance optimisation
        :param dist_matrix: distance matrix (numpy.array)
        :param C: regularisation parameter to make risk matrix from dist matrix
        :param benchmark: whether a benchmark is being used
        :param benchmark_model: the benchmark model being used. Currently have ["Covariance", "Covaraince Shrinkage"].
        :return: Efficient Frontier object
        """
        if is_benchmark:
            if benchmark_model == "Covariance":
                risk_matrix = sample_cov(self.prices)
            elif benchmark_model == "Covariance Shrinkage":
                risk_matrix = CovarianceShrinkage(self.prices).ledoit_wolf()
            else:
                return "Not in list of benchmarks"

        else:
            risk_matrix = self.make_risk_matrix(dist_matrix, C)

        ef = EfficientFrontier(self.expected_returns, risk_matrix, weight_bounds=self.weight_bounds, solver=self.solver)
        return ef

    def max_sharpe_ratio(self, is_benchmark = False, benchmark_model = "Covariance", dist_matrix=None, rf_rate=0.02, days=252, C=1):
        """
        Calculates weights that optimise the Sharpe Ratio and the best Sharpe Ratio
        :param dist_matrix: distance matrix (numpy.array)
        :param benchmark: whether a benchmark is being used
        :param benchmark_model: the benchmark model being used. Currently have ["Covariance", "Covariance Shrinkage"].
        :param rf_rate: risk-free rate
        :param days: number of days in a year
        :return: optimised weights, sharpe ratio
        """

        ef = self.setup_efficient_frontier(C=C, is_benchmark=is_benchmark, benchmark_model=benchmark_model, dist_matrix=dist_matrix)
        ef.max_sharpe(risk_free_rate=rf_rate)
        weights = pd.DataFrame.from_dict(ef.clean_weights(), orient='index', columns=['weights'])
        sharpe_ratio = utils.calculate_sharpe_ratio(weights=weights, prices=self.prices, rf_rate=rf_rate, days=days)
        return weights, sharpe_ratio

