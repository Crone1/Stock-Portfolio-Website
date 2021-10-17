
# general packages
from tqdm.auto import tqdm
import pandas as pd
import os

# For loading the config
import yaml

# For scraping the ticker data
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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

    # Close the cookies pop-up
    custom_choices_button = driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div/div[3]/div[2]/div[1]').click()
    legit_interests_button = driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div/div[1]/div[2]/ul/li[3]').click()
    object_button = driver.execute_script("arguments[0].click();", WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[2]/div/div/div/div[2]/div[3]/div[1]/div[1]/div/label/input'))))
    save_button = driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div/div[3]/div[2]/div[1]').click()

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

    # close the chrome window
    driver.close()


def create_exchange_ticker_to_name_map():
    pass


def update_unique_currency_list(transactions, currency_list):
    
    # Get a list of the unique currencies
    unique_currencies = list(transactions["currency"].drop_duplicates())

    # Iterate through these unique currency and update the list
    for currency in unique_currencies:
        if currency not in currency_list:
            currency_list.append(currency)

    return currency_list


def update_ticker_date_map(transactions, ticker_date_map):

    # Get a list of the unique tickers
    unique_tickers = transactions[["stock_ticker", "exchange_ticker"]].drop_duplicates()
    unique_stock_tickers = list(unique_tickers["stock_ticker"])
    unique_exchange_tickers = list(unique_tickers["exchange_ticker"])

    # Get a list of the date when this ticker first appeared
    ticker_start_dates = [str(min(transactions[(transactions["stock_ticker"] == stock) & (transactions["exchange_ticker"] == exchange)]["date"]))[:10] for stock, exchange in zip(unique_stock_tickers, unique_exchange_tickers)]

    # Iterate through these unique value and update the ticker map
    for date, stock, exchange in zip(ticker_start_dates, unique_stock_tickers, unique_exchange_tickers):

        # Update the map of tickers to date by looking for the minimum date it appeared
        ticker_key = (stock, exchange)
        if ticker_key in ticker_date_map:
            if date < ticker_date_map[ticker_key]:
                ticker_date_map[ticker_key] = date
        else:
            ticker_date_map[ticker_key] = date

    return ticker_date_map


def extract_unique_stock_and_currency_values_from_all_transactions():

    # Iterate through each transactions data
    transactions_dir = "data/user_transactions"
    ticker_date_map, currency_list = {}, []
    for username in [filename[:-len("_transactions.csv")] for filename in os.listdir(transactions_dir)]:
        # read in the transactions data
        raw_transactions = read_in_raw_transactions_data(username)

        # Add the currency column to this data
        transactions = create_exchange_name_nd_curency_cols_using_ticker(raw_transactions)

        # Update the dictionaries to find the minimum date when the ticker/currency appeared
        ticker_date_map = update_ticker_date_map(transactions, ticker_date_map)
        currency_list = update_unique_currency_list(transactions, currency_list)

    return ticker_date_map, currency_list


def read_in_raw_transactions_data(username):

    # Read in the transactions table
    transactions = pd.read_csv("data/user_transactions/{}_transactions.csv".format(username))

    # Ensure the date column is a datetime oject
    transactions['date'] = pd.to_datetime(transactions['date'])

    return transactions



def create_table_of_historic_stock_prices(ticker_to_date_map):

    stock_prices_df = pd.DataFrame()
    stock_dividend_df = pd.DataFrame()
    for (stock_ticker, exchange_ticker), start_date in ticker_to_date_map.items():
        
        try:
            # get the close price & dividend info for each stock_ticker
            ticker_yf_object = yf.Ticker(stock_ticker)
            ticker_hist = ticker_yf_object.history(start=start_date)
            price_df = ticker_hist[["Close"]].rename(columns={"Close": "{} ({})".format(stock_ticker, exchange_ticker)})
            price_df.index.rename("date", inplace=True)
            dividend_df = ticker_hist[["Dividends"]].rename(columns={"Dividends": stock_ticker})
            dividend_df.index.rename("date", inplace=True)

            # add this price & dividend data to a dataframe for all tickers
            stock_prices_df = pd.concat([stock_prices_df, price_df], join="outer", axis=1)
            stock_dividend_df = pd.concat([stock_dividend_df, dividend_df], join="outer", axis=1)

        except:
            print(stock_ticker, "not found")

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


def create_company_name_col_using_ticker(df):

    # if the company_name column already exists then drop it
    df.drop(columns=["company_name"], inplace=True, errors="ignore")

    # Add a company name column to the datafram using the defined json file
    with open('../data/stock_ticker_to_name.json', 'r') as file:
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


def add_columns_to_data(df):#, exchange_currencies):
    #if exchange_currencies_currencies["exchange_ticker"] = "USD":
    #    p_curr = "$"
    
    df["share_cost_in_local_currency"] = df["num_shares"] * df["share_price"]
    df["currency_exchange_fee"] = [abs(v) for v in df["share_cost_in_local_currency"] * 0.001]/df["exchange_rate"]
    df["fixed_transaction_fee"] = 0.5
    df["variable_transaction_fee"] = round(((0.004/df["exchange_rate"]) *abs(df["num_shares"])), 2)
    df["total_outgoing_in_eur"] = (df["share_cost_in_local_currency"]/df["exchange_rate"]) + df["fixed_transaction_fee"] + df["variable_transaction_fee"]
    df["share_cost_in_eur"] = df["total_outgoing_in_eur"]-  df["fixed_transaction_fee"] - df["variable_transaction_fee"] - df["currency_exchange_fee"]

    return df


def add_post_transaction_summary(df):
    num_shares_to_date = df["num_shares"].cumsum()
    avg_exchange_rate_to_date = df[["num_shares","exchange_rate"]].cumprod(axis=1)["exchange_rate"].cumsum() / num_shares_to_date

    fees_paid_to_date = (df["currency_exchange_fee"] + df["fixed_transaction_fee"] + df["variable_transaction_fee"]).cumsum()
    avg_fees_per_share = (fees_paid_to_date * avg_exchange_rate_to_date) / num_shares_to_date

    total_share_cost_to_date = df[["num_shares","share_price"]].cumprod(axis=1)["share_price"].cumsum()
    avg_share_cost_to_date = total_share_cost_to_date / num_shares_to_date

    bep = avg_share_cost_to_date + avg_fees_per_share

    df["total_shares_to_date"] = num_shares_to_date
    df["break_even_price"] = bep

    return df


def get_specific_stock(all_data, stock_ticker, exchange_ticker):
    """
    Extract from the dataframe only the data which are associated with the given ticker

    Returns:
        Pandas Dataframe
    """

    return all_data[(all_data["stock_ticker"] == stock_ticker) & (all_data["exchange_ticker"] == exchange_ticker)].reset_index(drop=True)

    
def add_aggregate_price_columns(df):

    # Ensure the table is sorted by ticker and then by date
    df.sort_values(["stock_ticker", "date"], inplace=True, ignore_index=True)

    # Get a list of the stock tickers and their respective exchanges in this df
    unique_tickers_list = list(df["stock_ticker"].drop_duplicates())
    exchanges_list = [str(min(df[df["stock_ticker"] == ticker]["exchange_ticker"]))[:10] for ticker in unique_tickers_list]
    
    # iterate through each stock ticker and add the columns to this stocks data
    df_with_agg_cols = pd.DataFrame()
    for stock_ticker, exchange_ticker in zip(unique_tickers_list, exchanges_list):
        # Get this stocks data from the df
        stocks_data = get_specific_stock(df, stock_ticker, exchange_ticker)
        # Add the relevant columns
        stocks_data_w_cols = add_columns_to_data(stocks_data)
        stocks_data_w_cols = add_post_transaction_summary(stocks_data_w_cols)

        # Append this new df with the aggregate columns to a df of all the stocks data with these cols
        df_with_agg_cols = pd.concat([df_with_agg_cols, stocks_data_w_cols], axis=0, ignore_index=True)
        
    return df_with_agg_cols


def calculate_amount_in_cumulative_account_each_day(username):

    # read in the table of when money was deposited and create a table showing the balance changes
    deposites_df = pd.read_csv("data/user_deposites/{}_deposites.csv".format(username), index_col="date")
    balance_changes_df = deposites_df["amount"].cumsum()

    # ensure there is only one value for each day
    balance_changes_df = balance_changes_df.reset_index().drop_duplicates(subset="date", keep="last").set_index("date")

    # ensure the date index is stored as a date
    balance_changes_df.index = pd.to_datetime(balance_changes_df.index)

    # create a dataframe with a row for each day
    daily_rows_df = pd.DataFrame({"date": pd.date_range(balance_changes_df.index.min(), datetime.now().date(), freq='D')}).set_index("date")

    # expand out our balance dataframe to have a row for each day
    daily_balance_df = pd.merge(daily_rows_df, balance_changes_df, how="left", left_index=True, right_index=True).ffill(axis=0)

    return daily_balance_df


def calculate_cumulative_amount_spent_each_day(transactions_df):

    # Subset the columns to get the amount spent and the date
    transaction_cost_df = transactions_df[["date", "total_outgoing_in_eur"]]

    # group the dta by the date and sum the values for each date
    total_spent_df = transaction_cost_df.groupby(by="date").sum().rename(columns={"total_outgoing_in_eur": "amount"})

    # create a cumulative amount spent
    cumulative_total_spent_df = total_spent_df["amount"].cumsum()

    # create a dataframe with a row for each day
    daily_rows_df = pd.DataFrame({"date": pd.date_range(cumulative_total_spent_df.index.min(), datetime.now().date(), freq='D')}).set_index("date")

    # expand out our daily spend dataframe to have a row for each day
    daily_spend_df = pd.merge(daily_rows_df, cumulative_total_spent_df, how="left", left_index=True, right_index=True).ffill(axis=0)

    return daily_spend_df

