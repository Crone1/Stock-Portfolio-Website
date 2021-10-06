
# general packages
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from dateutil.parser import parse

# For selecting period of valuation table
from dateutil.relativedelta import relativedelta


def col_to_date(col):
    return [d.date() for d in pd.to_datetime(col)]


def get_exchange_rate(date, base_currency, currency):

    # read in the exchange rate data
    scraped_rates = pd.read_csv("data/{}_currency_exchange_data.csv".format(base_currency))

    # Turn the date column to a date and set it as the index
    scraped_rates["date"] = col_to_date(scraped_rates["date"])
    scraped_rates.set_index("date", inplace=True)

    # Fill in weekend values if they are missing
    df_with_all_days = scraped_rates.reindex([d.date() for d in pd.date_range(start=min(scraped_rates.index), end=max(scraped_rates.index))])
    all_rates = df_with_all_days.fillna(method='ffill')

    # Return the rate for the currency on the date we're looking for
    return float(all_rates.loc[date, currency])


def plot_exchange_rates(base_currency):

    # read in the exchange data
    currency_rates_df = pd.read_csv("data/{}_currency_exchange_data.csv".format(base_currency))

    # ensure the date column is stored as a date
    currency_rates_df["date"] = col_to_date(currency_rates_df["date"])

    # plot the data
    num_plots = len(currency_rates_df.columns) - 1
    currency_rates_df.plot(x="date", subplots=True, figsize=(16, num_plots * 3), layout=(num_plots+1//2, 2), xlabel="Date", title="Exchange Rate to {} Over Time".format(base_currency), rot=90)


def get_price(date, ticker):

    # read in the stock price data
    scraped_prices = pd.read_csv("data/stock_price_data.csv")

    # Turn the date column to a date and set it as the index
    scraped_prices["date"] = col_to_date(scraped_prices["date"])
    scraped_prices.set_index("date", inplace=True)

    # Fill in weekend values if they are missing
    df_with_all_days = scraped_prices.reindex([d.date() for d in pd.date_range(start=min(scraped_prices.index), end=max(scraped_prices.index))])
    all_prices = df_with_all_days.fillna(method='ffill')

    # Return the price for the stock on the date we're looking for
    return float(all_prices.loc[date, ticker])


def plot_stock_prices():

    # read in the exchange data
    stock_price_df = pd.read_csv("data/stock_price_data.csv")

    # ensure the date column is stored as a date
    stock_price_df["date"] = col_to_date(stock_price_df["date"])

    # plot the data
    num_plots = len(stock_price_df.columns) - 1
    stock_price_df.plot(x="date", subplots=True, figsize=(16, num_plots * 3), layout=(num_plots+1//2, 2), xlabel="Date", title="Stock Prices Over Time", rot=90)


def get_period(period_str):

    period_dict = {"daily": relativedelta(days=1),
                   "weekly": relativedelta(weeks=1),
                   "monthly": relativedelta(months=1),
                   "quarterly": relativedelta(months=3),
                   "yearly": relativedelta(years=1),
                  }
    return period_dict[period_str]


def create_valuation_table_for_specific_stock(stock_df, selected_period_str, base_currency):

    # Get the valuation period
    selected_period = get_period(selected_period_str)

    # Verify that there is only one stock ticker and exchange present in this data
    tickers_in_df = stock_df["stock_ticker"].drop_duplicates()
    currencies_in_df = stock_df["currency"].drop_duplicates()
    if len(tickers_in_df) == 1 and len(currencies_in_df) == 1:
        ticker = tickers_in_df[0]
        currency = currencies_in_df[0]
    else:
        raise Exception('Error! Got multiple ticker symbols in the passed table')

    # initialise the dates
    current_date = datetime.now().date()
    prev_val_date = parse("1900-01-01").date()
    valuation_date = parse(str(stock_df.loc[0, "date"])).date()
    stock_df["date"] = col_to_date(stock_df["date"])

    # initialise the other variables
    shares_owned, total_spent_in_eur, total_fees_paid = 0, 0, 0

    # iteratively valuate the shares
    valuation_df = pd.DataFrame(columns=["date", "share_price", "exchange_rate", "num_shares_owned", "current_valuation_{}".format(base_currency), "price_paid_{}".format(base_currency), "adjusted_bep", "total_fees_paid", "absolute_profit", "percent_profit", "percent_fees"])
    utcnow = datetime.utcnow()
    pbar = tqdm(total=(current_date - valuation_date)//(utcnow + selected_period - utcnow), leave=False)
    while valuation_date <= current_date:

        try:
            # scrape the values for this stock associated with this exact date
            share_price = get_price(valuation_date, ticker)
            exchange_rate = get_exchange_rate(valuation_date, base_currency, currency)

        except KeyError:
            # remove the progress bar and continue to raise the error
            pbar.reset(total=0)
            pbar.close()
            raise KeyError

        # define the main valuation table columns
        df_up_to_date = stock_df[(stock_df["date"] > prev_val_date) & (stock_df["date"] <= valuation_date)]

        shares_owned += sum(df_up_to_date["num_shares"])
        total_spent_in_eur += sum(df_up_to_date["total_outgoing_in_eur"])
        total_fees_paid += sum(df_up_to_date["currency_exchange_fee"]) + sum(df_up_to_date["fixed_transaction_fee"]) + sum(df_up_to_date["variable_transaction_fee"])

        # define the aggregate columns based on these values
        value_at_date = (share_price * shares_owned) / exchange_rate
        exchange_rate_adjusted_bep = (total_spent_in_eur / shares_owned) * exchange_rate

        # Summarise the asset performance
        absolute_profit = value_at_date - total_spent_in_eur
        perc_profit = absolute_profit / total_spent_in_eur
        perc_fees = total_fees_paid / total_spent_in_eur

        # store these column values in a dataframe row
        valuation_df = valuation_df.append({"date": valuation_date,
                                            "share_price": share_price,
                                            "exchange_rate": exchange_rate,
                                            "num_shares_owned": shares_owned,
                                            "current_valuation_{}".format(base_currency): value_at_date,
                                            "price_paid_{}".format(base_currency): total_spent_in_eur,
                                            "adjusted_bep": exchange_rate_adjusted_bep,
                                            "total_fees_paid": total_fees_paid,
                                            "absolute_profit": absolute_profit,
                                            "percent_profit": perc_profit,
                                            "percent_fees":perc_fees
                                           }, ignore_index=True).reset_index(drop=True)

        # increment the dates
        prev_val_date = valuation_date
        valuation_date += selected_period
        pbar.update(1)
    pbar.close()

    return valuation_df


def get_specific_stock(all_data, stock_ticker, exchange_ticker):
    """
    Extract from the dataframe only the data which are associated with the given ticker

    Returns:
        Pandas Dataframe
    """

    return all_data[(all_data["stock_ticker"] == stock_ticker) & (all_data["exchange_ticker"] == exchange_ticker)].reset_index(drop=True)


def create_all_valuation_tables(transactions, base_currency, val_period):

    # Define a list of the primary keys to each stock
    stock_primary_keys = transactions[["stock_ticker", "exchange_ticker"]].drop_duplicates()

    # Iterate through each stock and create a valuation table for it
    val_tables_map = {}
    pbar = tqdm(total=len(stock_primary_keys))
    for index, (stock_ticker, exchange_ticker) in stock_primary_keys.iterrows():
        try:
            # Retrieve this stocks transaction data
            stock_data = get_specific_stock(transactions, stock_ticker, exchange_ticker)
            
            # Create a valuation table for this data
            val_table = create_valuation_table_for_specific_stock(stock_data, val_period, base_currency)

        except KeyError:
            print("Error with ticker: '{}' ({}). No valuation table calculated for this stock!".format(stock_ticker, exchange_ticker))
            val_table = None

        # Store a dictionary mapping the keys to the stocks valuation tables
        val_tables_map[(stock_ticker, exchange_ticker)] = val_table
        pbar.update(1)

    return val_tables_map

