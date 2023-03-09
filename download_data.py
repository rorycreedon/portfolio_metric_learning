# Import packages
import pandas as pd
import simfin as sf
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import yfinance as yf

class SimFinData:

    def __init__(self, API_KEY, data_dir):
        super(SimFinData, self).__init__()
        self.API_KEY = API_KEY
        self.data_dir = data_dir

    def setup(self):
        sf.set_api_key(self.API_KEY)
        sf.set_data_dir(self.data_dir)

    def download_income(self):
        """Download income statement data from SimFin API and return a pandas dataframe"""
        income = sf.load_income(variant='quarterly', market='us')
        # Drop index
        income = income.reset_index()
        # drop columns from income
        income = income.drop(columns=['Publish Date', 'Restated Date', 'Currency', 'Net Extraordinary Gains (Losses)', 'Abnormal Gains (Losses)', 'Report Date', 'SimFinId'])
        # Rename Fiscal Year and Fiscal Period to year and quarter
        income = income.rename(columns={'Fiscal Year': 'Year', 'Fiscal Period': 'Quarter'})

        return income
    
    def download_balance(self):
        """Download balance sheet data from SimFin API and return a pandas dataframe"""
        balance_sheet = sf.load_balance(variant='quarterly', market='us')
        balance_sheet = balance_sheet.reset_index()
        balance_sheet = balance_sheet.rename(columns={'Fiscal Year': 'Year', 'Fiscal Period': 'Quarter'})
        # Drop columns
        balance_sheet = balance_sheet.drop(columns=['SimFinId', 'Currency', 'Restated Date', 'Report Date', 'Publish Date'])
        
        return balance_sheet
    
    def download_cashflow(self):
        """Download cash flow data from SimFin API and return a pandas dataframe"""
        cash_flow = sf.load_cashflow(variant='quarterly', market='us')
        cash_flow = cash_flow.reset_index()
        cash_flow = cash_flow.drop(cash_flow.filter(like='Change in').columns, axis=1)
        cash_flow = cash_flow.rename(columns={'Fiscal Year': 'Year', 'Fiscal Period': 'Quarter'})
        cash_flow = cash_flow.drop(columns=['SimFinId', 'Currency', 'Restated Date', 'Publish Date', 'Report Date'])
        return cash_flow
    
    def download_companies(self):
        """Download company data from SimFin API and return a pandas dataframe"""
        companies = sf.load_companies(market='us')
        companies = companies.reset_index()
        companies = companies.drop(columns=['SimFinId', 'Company Name'])

        industries = sf.load_industries()
        industries = industries.reset_index()
        industries = industries.drop(columns=['Industry'])
        # One hot encode sectors
        industries = pd.get_dummies(industries, columns=['Sector'], prefix='', prefix_sep='')


        companies = companies.merge(industries, on='IndustryId', how='left')

        return companies
    
    def download_hist_prices(self):
        """Download historical share prices from SimFin API and return a pandas dataframe"""
        prices = sf.load_shareprices(variant='daily', market='us')
        prices = prices.reset_index()
        prices = prices.drop(columns=['SimFinId', 'Open', 'Low', 'High', 'Adj. Close', 'Shares Outstanding'])
        prices['Year'] = prices['Date'].dt.year
        prices['Quarter'] = prices['Date'].dt.quarter
        # Convert Quarter to string
        prices['Quarter'] = "Q" + prices['Quarter'].astype(str)
        prices = prices.rename(columns={'Close': 'Price'})
        return prices
    
    def download_all_data(self, start_date=None):
        """Download all data from SimFin API, merge and return a pandas dataframe"""
        
        # Download data
        income = self.download_income()
        balance_sheet = self.download_balance()
        cash_flow = self.download_cashflow()
        companies = self.download_companies()
        prices = self.download_hist_prices()
        
        # Merge together
        data = prices.merge(balance_sheet, on=['Ticker', 'Year', 'Quarter'], how='inner', suffixes=('', '_duplicate'))
        data = data.merge(income, on=['Ticker', 'Year', 'Quarter'], how='inner', suffixes=('', '_duplicate'))
        data = data.merge(cash_flow, on=['Ticker', 'Year', 'Quarter'], how='inner', suffixes=('', '_duplicate'))
        data = data.merge(companies, on=['Ticker'], how='inner', suffixes=('', '_duplicate'))
        data.drop(data.filter(regex='_duplicate$').columns, axis=1, inplace=True)

        # Drop index columns
        data = data.drop(columns=['Year', 'Quarter'])
        data.set_index(['Date'], inplace=True)

        # S&P 500 companies only
        sp500 = pd.read_csv('data/sp500.csv')

        if start_date is None:
            start_date = str(data.index[-1].date())

        # Find a date that we know which companies are in the S&P 500
        i=1
        while start_date not in sp500['date'].values:
            i+=1
            start_date = str(data.index[-i].date())

        # Include only those in the S&P 500
        tickers = pd.DataFrame(sp500[sp500['date'] == start_date]['tickers'].values[0].split(','), columns=['Ticker'])
        data = data.reset_index().merge(tickers, on=['Ticker'], how='inner').set_index('Date')

        # Reshape
        data = data.pivot_table(index = 'Date', columns = 'Ticker', values = data.columns[1:], aggfunc = 'sum')

        # Swap column levels
        data.columns = data.columns.swaplevel(0, 1)

        return data
    
    def clean_normalise(self):

        # Do something about NaNs

        raise NotImplementedError()


    def download_prices(self, shape='wide', output_package = 'numpy'):
        # Download data from API
        prices_df = sf.load_shareprices(variant='daily', market='us')
        
        if shape == 'wide':
            # Reshape data
            prices_df = prices_df.pivot_table(index='Date', columns='Ticker', values='Close')

            # Dates
            first_date = prices_df.index[0]
            last_date = prices_df.index[-1]

            # Drop columns that where the first or last date are missing
            cols_to_remove = prices_df.loc[last_date][prices_df.loc[first_date].isna()].index
            prices_df = prices_df.drop(columns=cols_to_remove)
            cols_to_remove = prices_df.loc[last_date][prices_df.loc[first_date].isna()].index
            prices_df = prices_df.drop(columns=cols_to_remove)

            # Fill in missing values with the average of the previous and next value
            prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')

            # Drop columns where the max value is more than 10 times the min value
            cols_to_remove = prices_df.max()[prices_df.max() > 10*prices_df.min()].index
            prices_df = prices_df.drop(columns=cols_to_remove)

            # Index prices starting at 100
            prices_df = prices_df.div(prices_df.iloc[0]).mul(100)

            # Drop first row
            prices_df = prices_df.iloc[1:]

            if output_package == 'numpy':
                return np.expand_dims(prices_df.to_numpy().T, axis=2)
            
            if output_package == 'pandas':
                return prices_df
            
            if output_package == 'torch':
                return torch.tensor(np.expand_dims(prices_df.to_numpy().T, axis=2))
            
        if shape == 'long':
            prices_df = pd.DataFrame(prices_df['Close'])
            prices_df = prices_df.rename(columns={'Close': 'Price'})
            prices_df

            if output_package == 'numpy':
                return prices_df.to_numpy()
            
            if output_package == 'pandas':
                return prices_df
            
            if output_package == 'torch':
                return torch.tensor(prices_df.to_numpy())

def download_orbis_ratios():

    # Import data
    ratios = pd.read_excel('data/S&P Ratios.xlsx', index_col=0, sheet_name="Results", usecols='C:DU')

    # Replace "n.a." and "n.s." with NaN
    ratios = ratios.replace('n.a.', np.nan)
    ratios = ratios.replace('n.s.', np.nan)

    # Get stub names
    cols = []
    for c in ratios.columns:
        if c[-1] in ['1','2','3','4']:
            cols.append(c[:-len('\n2022 Quarter 4')])
    cols = list(np.unique(cols))

    # Reshape
    ratios = ratios.reset_index()
    reshaped = pd.wide_to_long(ratios, stubnames=cols, i=["Ticker symbol"], j="quarter_year", sep='\n', suffix = '.+').reset_index()

    # Extract year and quarter
    reshaped['Year'] = reshaped['quarter_year'].str.split(' ').str[0]
    reshaped['Quarter'] = reshaped['quarter_year'].str.split(' ').str[2]

    # Drop quarter_year
    reshaped = reshaped.drop(columns=['quarter_year'])

    # Drop where year is 2022 and quarter is 4
    reshaped = reshaped[~((reshaped['Year'] == '2022') & (reshaped['Quarter'] == '4'))]

    # Drop rows with NaNs
    reshaped = reshaped.dropna(axis=0, how='any')

    # Count number of rows by Ticker symbol
    counts = pd.DataFrame(reshaped['Ticker symbol'].value_counts())
    counts.rename(columns={'Ticker symbol': 'count'}, inplace=True)
    counts = counts.reset_index()
    counts.rename(columns={'index': 'Ticker symbol'}, inplace=True)
    counts = counts[counts['count']==counts['count'].max()]
    counts.drop(columns=['count'], inplace=True)

    # Only include tickers that we have data for all quarters
    reshaped = reshaped.merge(counts, on=['Ticker symbol'], how='inner')

    # Clean up column types
    reshaped['Year'] = reshaped['Year'].astype(int)
    reshaped['Quarter'] = reshaped['Quarter'].astype(int)
    reshaped['Cash flow / Operating revenue (%)'] = reshaped['Cash flow / Operating revenue (%)'].astype(float)
    reshaped['Current ratio'] = reshaped['Current ratio'].astype(float)
    reshaped['Net assets turnover'] = reshaped['Net assets turnover'].astype(float)
    reshaped['Profit margin (%)'] = reshaped['Profit margin (%)'].astype(float)
    reshaped['Solvency ratio (Asset based) (%)'] = reshaped['Solvency ratio (Asset based) (%)'].astype(float)

    return reshaped

if __name__ == '__main__':

    data = download_yfinance_data()
    data.to_csv("data/prices.csv")

#     # Example usage
#     downloader = SimFinData(API_KEY=os.environ.get("SIMFIN_API_KEY"), data_dir="~/simfin_data/")
#     downloader.setup()
#     prices = downloader.download_prices(shape = 'wide', output_package = 'numpy')
