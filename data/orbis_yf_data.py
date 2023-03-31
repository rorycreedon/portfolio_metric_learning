import pandas as pd
import numpy as np
import yfinance as yf
import datetime


def clean_orbis_ratios(ratios):
    """
    Clean Orbis ratios from downloaded Excel file
    :param ratios: a pandas DataFrame containing Orbis ratios, read from an Excel file
    :return: Cleaned pandas DataFrame
    """

    # Replace "n.a." and "n.s." with NaN
    ratios = ratios.replace('n.a.', np.nan)
    ratios = ratios.replace('n.s.', np.nan)

    # Get stub names
    cols = []
    for c in ratios.columns:
        if c[-1] in ['1', '2', '3', '4']:
            cols.append(c[:-len('\n2022 Quarter 4')])
    cols = list(np.unique(cols))

    # Reshape
    ratios = ratios.reset_index()
    ratios = pd.wide_to_long(ratios, stubnames=cols, i=["Ticker symbol"], j="quarter_year", sep='\n',
                             suffix='.+').reset_index()

    # Extract year and quarter
    ratios['Year'] = ratios['quarter_year'].str.split(' ').str[0]
    ratios['Quarter'] = ratios['quarter_year'].str.split(' ').str[2]

    # Drop quarter_year
    ratios = ratios.drop(columns=['quarter_year'])

    # Drop where year is 2022 and quarter is 4
    ratios = ratios[~((ratios['Year'] == '2022') & (ratios['Quarter'] == '4'))]

    # Drop rows with NaNs
    ratios = ratios.dropna(axis=0, how='any')

    # Count number of rows by Ticker symbol
    counts = pd.DataFrame(ratios['Ticker symbol'].value_counts())
    counts.rename(columns={'Ticker symbol': 'count'}, inplace=True)
    counts = counts.reset_index()
    counts.rename(columns={'index': 'Ticker symbol'}, inplace=True)
    counts = counts[counts['count'] == counts['count'].max()]
    counts.drop(columns=['count'], inplace=True)

    # Only include tickers that we have data for all quarters
    ratios = ratios.merge(counts, on=['Ticker symbol'], how='inner')

    # Clean up column types
    ratios['Year'] = ratios['Year'].astype(int)
    ratios['Quarter'] = ratios['Quarter'].astype(int)
    ratios['Cash flow / Operating revenue (%)'] = ratios['Cash flow / Operating revenue (%)'].astype(float)
    ratios['Current ratio'] = ratios['Current ratio'].astype(float)
    ratios['Net assets turnover'] = ratios['Net assets turnover'].astype(float)
    ratios['Profit margin (%)'] = ratios['Profit margin (%)'].astype(float)

    return ratios


def sp500_tickers(start_date='2019-12-23'):
    """
    Get S&P 500 tickers from a given date
    :param start_date: The date to get the tickers from
    :return: A list containing the tickers
    """
    # S&P 500 companies only
    sp500 = pd.read_csv('S&P Tickers.csv')
    ticker_list = sp500[sp500['date'] == start_date]['tickers'].iloc[0].split(',')  # last change before 2020-01-01

    return ticker_list


def get_tickers_with_retry(start_date, tickers_function):
    date_format = "%Y-%m-%d"
    current_date = datetime.datetime.strptime(start_date, date_format)

    while True:
        try:
            tickers = tickers_function(current_date.strftime(date_format))
            return tickers
        except Exception as e:
            print(f"Error occurred for date {current_date.strftime(date_format)}: {e}")
            current_date -= datetime.timedelta(days=1)


def yf_data(tickers):
    """
    Download price, price momentum, volume and volume momentum data from Yahoo Finance
    :param tickers: The tickers to download data for
    :return: A cleaned dataframe containing price, price momentum, volume and volume momentum data from Yahoo Finance
    """
    # Download YF data
    data = yf.download(tickers, start="2017-01-01", end="2023-01-01")

    # Reshape
    data = data.reset_index()
    data = data.melt(id_vars='Date', var_name=['Attribute', 'Ticker symbol'])
    data = data.pivot_table(index=['Date', 'Ticker symbol'], columns='Attribute', values='value')
    data = data.reset_index()[['Date', 'Ticker symbol', 'Adj Close', 'Volume']]

    # rename column Adj Close to Price
    data.rename(columns={"Adj Close": "Price"}, inplace=True)

    # 100 indexing
    data['Price'] = data.groupby('Ticker symbol', group_keys=False)['Price'].apply(lambda x: x / x.iloc[0] * 100)
    data['Volume'] = data.groupby('Ticker symbol', group_keys=False)['Volume'].apply(lambda x: x / x.iloc[0] * 100)

    # Calculate momentum for each ticker
    data['Price Momentum'] = data.groupby('Ticker symbol')['Price'].pct_change(30)
    data['Price Momentum'] = data['Price Momentum'].replace([np.inf, -np.inf], 0)
    data['Volume Momentum'] = data.groupby('Ticker symbol')['Volume'].pct_change(30)
    data['Volume Momentum'] = data['Volume Momentum'].replace([np.inf, -np.inf], 0)

    # Drop if Price Momentum is NaN
    data = data.dropna(subset=['Price Momentum'])

    # Drop tickers where volume is 0
    data = data[data['Volume'] != 0]

    # 100 indexing again
    data['Price'] = data.groupby('Ticker symbol', group_keys=False)['Price'].apply(lambda x: x / x.iloc[0] * 100)
    data['Volume'] = data.groupby('Ticker symbol', group_keys=False)['Volume'].apply(lambda x: x / x.iloc[0] * 100)

    # Extract year and quarter
    data['Year'] = data['Date'].dt.year
    data['Quarter'] = data['Date'].dt.quarter

    return data


def merge_reshape_data(orbis_ratios, yf_prices):
    """
    Merge and reshape the Orbis and Yahoo Finance data
    :param orbis_ratios: Cleaned Orbis dataframe
    :param yf_prices: Cleaned Yahoo Finance dataframe
    :return: Merged and reshaped combined dataframe
    """
    # Merge
    data = orbis_ratios.merge(yf_prices, on=['Ticker symbol', 'Year', 'Quarter'], how='inner')

    # Drop year and quarter
    data.drop(columns=['Year', 'Quarter'], inplace=True)

    # Reshape
    data = data.pivot_table(index='Date', columns='Ticker symbol', values=data.columns[1:])

    # Drop NaNs
    data = data.transpose().dropna(how='any').transpose()

    return data


def download_all_data(sp500_ratios, date):
    # Download S&P 500 data
    orbis_ratios = clean_orbis_ratios(sp500_ratios)
    yf_all_data = yf_data(get_tickers_with_retry(date, sp500_tickers))

    # Merge
    data = merge_reshape_data(orbis_ratios, yf_all_data)

    # Save
    data.to_pickle(f'data/sp500_data.pkl')


if __name__ == "__main__":

    sp500_ratios = pd.read_excel('data/S&P Ratios.xlsx', index_col=0, sheet_name="Results", usecols='C:CU')
    download_all_data(sp500_ratios, '2019-12-23')
