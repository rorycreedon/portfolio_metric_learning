# Import packages
import pandas as pd
import simfin as sf

# API key for downloading data from SimFin server.
API_KEY = "ly74Rpz46DQxyI1hESLEf7P2M3Idbp6j"

# Set your SimFin+ API-key for downloading data.
sf.set_api_key(API_KEY)

# Set the local directory where data-files are stored.
# The directory will be created if it does not already exist.
sf.set_data_dir("~/simfin_data/")

# Download the data from the SimFin server and load into a Pandas DataFrame.
companies = sf.load_companies(market="us")
income_statement = sf.load_income(variant="quarterly", market="us")
balance_sheet = sf.load_balance(variant="quarterly", market="us")
cash_flow = sf.load_cashflow(variant="quarterly", market="us")
industry = sf.load_industries()

# Merge the dataframes
df = income_statement.merge(companies, how="outer", left_index=True, right_index=True)
df = df.merge(balance_sheet, how="outer", left_index=True, right_index=True, validate="1:1")
df = df.merge(cash_flow, how="outer", left_index=True, right_index=True, validate="1:1")
df = df.merge(
    industry, how="left", left_index=False, left_on='IndustryId', right_index=True,
)
