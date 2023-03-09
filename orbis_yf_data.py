import pandas as pd
import numpy as np
import yfinance as yf

def download_orbis_ratios():

    # Import data
    ratios = pd.read_excel('data/S&P Ratios.xlsx', index_col=0, sheet_name="Results", usecols='C:EQ')

    # Replace "n.a." and "n.s." with NaN
    ratios = ratios.replace('n.a.', np.nan)
    ratios = ratios.replace('n.s.', np.nan)

    # Get stub names
    cols = []
    for c in ratios.columns:
        if c[-1] in ['1','2','3','4']:
            cols.append(c[:-len('\n2022 Quarter 4')])
    cols = list(np.unique(cols))
    cols

    # Reshape
    ratios = ratios.reset_index()
    ratios = pd.wide_to_long(ratios, stubnames=cols, i=["Ticker symbol"], j="quarter_year", sep='\n', suffix = '.+').reset_index()

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
    counts = counts[counts['count']==counts['count'].max()]
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
    ratios['Solvency ratio (Asset based) (%)'] = ratios['Solvency ratio (Asset based) (%)'].astype(float)

    return ratios

def sp500_tickers(start_date = '2019-12-23'):

    # S&P 500 companies only
    sp500 = pd.read_csv('data/S&P Tickers.csv')
    ticker_list = sp500[sp500['date'] == start_date]['tickers'].iloc[0].split(',') # last change before 2020-01-01

    return ticker_list

def yf_data(tickers):

    # Download YF data
    data = yf.download(tickers, start="2017-01-01", end="2022-12-31")['Close']
    # 100 indexing prices
    data = data.div(data.iloc[0]).mul(100)
    # Drop tickers with NaNs
    data.dropna(axis=1, how='all', inplace=True)

    # reshape wide to long
    data = data.reset_index()
    data = pd.melt(data, id_vars='Date', value_vars=list(data.columns[1:]), var_name='Ticker symbol', value_name='Price')

    # Extract year and quarter
    data['Year'] = data['Date'].dt.year
    data['Quarter'] = data['Date'].dt.quarter

    return data

def merge_reshape_data(orbis_ratios, yf_prices):

    # Merge
    data = orbis_ratios.merge(yf_prices, on=['Ticker symbol', 'Year', 'Quarter'], how='inner')

    # Drop year and quarter
    data.drop(columns=['Year', 'Quarter'], inplace=True)

    # Reshape
    data = data.pivot_table(index = 'Date', columns = 'Ticker symbol', values = data.columns[1:])

    return data


if __name__ == "__main__":
    # Download data
    orbis_ratios = download_orbis_ratios()
    yf_prices = yf_data(sp500_tickers('2019-12-23'))

    # Merge
    data = merge_reshape_data(orbis_ratios, yf_prices)

    # Save
    data.to_pickle('data/data.pkl')
