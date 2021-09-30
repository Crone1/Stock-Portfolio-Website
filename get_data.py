
# general packages
from tqdm.auto import tqdm
import pandas as pd

# For loading the config
import yaml

# For scraping the ticker data
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# For querying Yahoo Finance
import yfinance as yf

# For getting currency exchange rates
import requests
from datetime import datetime, timedelta

# For storing the data
import json

# Read in config of my chromedriver location
with open("configs/config.yaml", "r") as config:
    c = yaml.load(config, Loader=yaml.FullLoader)
chromedriver_location = c["chromedriver_location"]



def read_in_raw_transactions_data():

    # Read in the transactions table
    transactions = pd.read_csv("data/transactions.csv")

    # Ensure the date column is a datetime oject
    transactions['date'] = pd.to_datetime(transactions['date'])

    return transactions


def create_company_name_col_using_ticker(df):

    # if the company_name column already exists then drop it
    df.drop(columns=["company_name"], inplace=True, errors="ignore")

    # Add a company name column to the datafram using the defined json file
    with open('data/stock_ticker_to_name.json', 'r') as file:
        comp_name_map = json.load(file)
        df["company_name"] = [comp_name_map[ticker] if ticker in comp_name_map else "" for ticker in df["stock_ticker"]]

    # Check if any rows didn't have an associated company name in thie json file
    rows_with_no_company_name = df[df["company_name"] == ""]
    if len(rows_with_no_company_name) > 0:
        print("The following stocks did not have an associated company name:")
        print(rows_with_no_company_name[["stock_ticker", "exchange_ticker"]].drop_duplicates().to_markdown())

    return df


def create_exchange_name_nd_curency_cols_using_ticker(df):

    # if the exchange_name column already exists then drop it
    df.drop(columns=["exchange_name", "currency"], inplace=True, errors="ignore")

    # Add an exchange name column to the datafram using the defined CSV
    exch_tab = pd.read_csv('data/exchange_name_and_currency.csv')
    df_with_name = df.merge(exch_tab, how='left', on="exchange_ticker")

    # Check if any rows didn't have an associated exchange name in thie CSV
    rows_with_no_exchange_name = df_with_name[df_with_name["exchange_name"] == None]
    if len(rows_with_no_exchange_name) > 0:
        print("The following rows did not have an associated exchange name:")
        print(rows_with_no_exchange_name.to_markdown())

    return df_with_name


def create_company_ticker_to_name_map():
    """
	Scrape the Stock Analysis website to create a map from the ticker names of thousands of stocks to their respective company names

	Exports this data to a Json file for further use
    """

    # define the website URl
    url = 'https://stockanalysis.com/stocks/'

    # open the browser
    driver = webdriver.Chrome(chromedriver_location)
    driver.get(url)
    time.sleep(2)

    # check how many pages there are
    pages_text = driver.find_element(By.XPATH, '/html/body/div/div/main/div/div/nav/div/span').text
    num_pages = int(pages_text.split(" ")[-1])

    # define a dictionary to store the scraped data
    ticker_to_company_map = {}

    # iterate through each page
    for pg_num in tqdm(range(num_pages)):
        # iteratively scrape each ticker and company name
        num_tickers_on_page = len(driver.find_elements(By.XPATH, "/html/body/div/div/main/div/div/div[2]/table/tbody/tr"))
        for i in tqdm(range(1, num_tickers_on_page), leave=False):
            ticker = driver.find_element(By.XPATH, '/html/body/div/div/main/div/div/div[2]/table/tbody/tr[{}]/td[1]/a'.format(i)).text
            company_name = driver.find_element(By.XPATH, '/html/body/div/div/main/div/div/div[2]/table/tbody/tr[{}]/td[2]'.format(i)).text
            ticker_to_company_map[ticker] = company_name

        if pg_num != num_pages - 1:
            # Change to the next page
            next_page_button = driver.find_element(By.XPATH, '/html/body/div/div/main/div/div/nav/button[2]')
            next_page_button.click()

    # store this scraped data as json file
    with open('data/ticker_to_name.json', 'w') as file:
        json.dump(ticker_to_company_map, file)


def create_table_of_historic_stock_prices(ticker_list, start_date_list):

    stock_prices_df = pd.DataFrame()
    stock_dividend_df = pd.DataFrame()
    for ticker, start_date in zip(ticker_list, start_date_list):
        
        try:
            # get the close price & dividend info for each ticker
            ticker_yf_object = yf.Ticker(ticker)
            ticker_hist = ticker_yf_object.history(start=start_date)
            price_df = ticker_hist[["Close"]].rename(columns={"Close": ticker})
            price_df.index.rename("date", inplace=True)
            dividend_df = ticker_hist[["Dividends"]].rename(columns={"Dividends": ticker})
            dividend_df.index.rename("date", inplace=True)

            # add this price & dividend data to a dataframe for all tickers
            stock_prices_df = pd.concat([stock_prices_df, price_df], join="outer", axis=1)
            stock_dividend_df = pd.concat([stock_dividend_df, dividend_df], join="outer", axis=1)

        except:
            print(ticker, "not found")

    # store this data as a CSV
    stock_prices_df.to_csv("data/stock_price_data.csv", index=True)
    stock_dividend_df.to_csv("data/stock_dividend_data.csv", index=True)


def create_exchange_rate_to_date_map(base_currency, exchange_currency_list, start_date):

    # Define the main variables needed to scrape the exchange rates
    base_url = 'https://api.exchangerate.host/timeseries?'
    target_currecies = ','.join(exchange_currency_list)
    end_date = datetime.now().date()
    currency_rates_df = pd.DataFrame()

    while True:
        # define the URL to query
        query = base_url + f'start_date={start_date}&end_date={end_date}&base={base_currency}&symbols={target_currecies}'

        # Query the URL for the currency data
        url_response = requests.get(query).json()

        # Store the currency exchange rates in a dataframe
        if url_response["success"]:
            response_df = pd.DataFrame(url_response["rates"]).T
            response_df.index = pd.to_datetime(response_df.index)
            currency_rates_df = pd.concat([currency_rates_df, response_df], axis=0)
        else:
            print("ERROR when scraping exchange rates:\n", url_response)

        # Exit the loop if the last day scraped is the same as the specified end_date
        last_date_scraped = max(currency_rates_df.index)
        if last_date_scraped == end_date:
            break
        else:
            start_date = last_date_scraped + timedelta(days=1)

    # store this data as a CSV
    currency_rates_df.index.rename("date", inplace=True)
    currency_rates_df.to_csv("data/{}_currency_exchange_data.csv".format(base_currency), index=True)

