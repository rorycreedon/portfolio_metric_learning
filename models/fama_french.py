import pandas as pd
import statsmodels.api as sm
import numpy as np
from numba import jit


class FamaFrench:

    def __init__(self, prices, file_path='data/F-F_Research_Data_Factors_daily.CSV', n_rows=25419):
        self.prices = prices
        self.file_path = file_path
        self.n_rows = n_rows

    def _prepare_data(self):
        """
        Import data for Fama-French model and merge with prices dataframe
        :return: Merged dataframe of returns and factor data
        """
        # Import factor data
        factor_data = pd.read_csv(self.file_path, index_col=0, parse_dates=True, skiprows=4,
                                  date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d'), nrows=self.n_rows)

        # Merge in returns for each stock
        returns = self.prices.pct_change().dropna()

        # Merge returns_train and factor_data on index
        returns_merged = returns.merge(factor_data.drop(columns=['RF']), left_index=True, right_index=True, how='left')

        return returns_merged

    def _estimate_params_vars(self, returns_merged):
        """
        Estimate factor loadings and idiosyncratic variances for each stock
        :param returns_merged: Merged dataframe of returns and factor data
        :return: Factor loadings and idiosyncratic variances for each stock
        """
        # Empty dict for factor loadings
        factor_loadings = {}
        # Empty dict for idiosyncratic variances
        idiosyncratic_vars = {}

        # Loop through stocks to estimate factor loadings and idiosyncratic variances
        for ticker in self.prices.columns:
            y = returns_merged[ticker]
            X = returns_merged[['Mkt-RF', 'SMB', 'HML']]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            factor_loadings[ticker] = model.params
            residuals = model.resid
            idiosyncratic_vars[ticker] = residuals.var()

        return factor_loadings, idiosyncratic_vars

    @staticmethod
    @jit(nopython=True)
    def make_covariance_matrix(factor_loadings_arr, idiosyncratic_vars_arr, factor_cov_matrix_arr, n_stocks):
        """
        Make covariance matrix for each stock
        :param factor_loadings_arr: A np array of factor loadings for each stock
        :param idiosyncratic_vars_arr: A np array of idiosyncratic variances for each stock
        :param factor_cov_matrix_arr: A np array of the factor covariance matrix
        :param n_stocks: The number of stocks
        :return: Multi-factor covariance matrix
        """
        multi_factor_cov_matrix = np.zeros((n_stocks, n_stocks))

        for i in range(n_stocks):
            for j in range(n_stocks):
                multi_factor_cov_matrix[i, j] = (
                        factor_loadings_arr[i, 1:] * factor_loadings_arr[j, 1:] * factor_cov_matrix_arr).sum()
                if i == j:
                    multi_factor_cov_matrix[i, j] += idiosyncratic_vars_arr[i]

        return multi_factor_cov_matrix

    def get_covariance_matrix(self):
        """
        Get covariance matrix for each stock
        :return: Multi-factor covariance matrix
        """
        returns_merged = self._prepare_data()
        factor_loadings, idiosyncratic_vars = self._estimate_params_vars(returns_merged)
        returns = returns_merged[['Mkt-RF', 'SMB', 'HML']].values
        factor_cov_matrix = np.cov(returns, rowvar=False)

        tickers = self.prices.columns
        n_stocks = len(tickers)

        factor_loadings_arr = np.array([factor_loadings[ticker].values for ticker in tickers])
        idiosyncratic_vars_arr = np.array([idiosyncratic_vars[ticker] for ticker in tickers])

        multi_factor_cov_matrix = self.make_covariance_matrix(
            factor_loadings_arr, idiosyncratic_vars_arr, factor_cov_matrix, n_stocks
        )

        # Convert the result back to a DataFrame
        multi_factor_cov_matrix = pd.DataFrame(multi_factor_cov_matrix, index=tickers, columns=tickers)

        return multi_factor_cov_matrix
