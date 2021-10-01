
# general packages
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse


def get_exchange_rate(date, base_currency, currency):
    all_rates = pd.read_csv("data/{}_currency_exchange_data.csv".format(base_currency))
    all_rates["date"] = [d.date() for d in pd.to_datetime(all_rates["date"])]
    return float(all_rates[all_rates["date"] == date][currency])


def get_price(date, ticker):
    all_prices = pd.read_csv("data/stock_price_data.csv")
    all_prices["date"] = [d.date() for d in pd.to_datetime(all_prices["date"])]
    return float(all_prices[all_prices["date"] == date][ticker])


def create_valuation_table(df, selected_period, base_currency):
    """
    """

    tickers_in_df = df["stock_ticker"].drop_duplicates()
    currencies_in_df = df["currency"].drop_duplicates()
    if len(tickers_in_df) == 1 and len(currencies_in_df) == 1:
        ticker = tickers_in_df[0]
        currency = currencies_in_df[0]
    else:
        raise Exception('Error! Got multiple ticker symbols in the passed table')

    # initialise the dates
    current_date = datetime.now().date()
    prev_val_date = parse("1900-01-01").date()
    valuation_date = parse(str(df.loc[0, "date"])).date()
    df["date"] = [d.date() for d in pd.to_datetime(df["date"])]

    # initialise the other variables
    shares_owned, total_spent_in_eur, total_fees_paid = 0, 0, 0

    # iteratively valuate the shares
    valuation_df = pd.DataFrame(columns=["date", "share_price", "exchange_rate", "num_shares_owned", "valuation", "price_paid", "adjusted_bep", "total_fees_paid", "absolute_profit", "percent_profit", "percent_fees"])
    while valuation_date <= current_date:

        # scrape the values associated with this exact date
        share_price = get_price(valuation_date, ticker)
        exchange_rate = get_exchange_rate(valuation_date, base_currency, currency)

        # define the main valuation table columns
        df_up_to_date = df[(df["date"] > prev_val_date) & (df["date"] <= valuation_date)]
        shares_owned += sum(df_up_to_date["num_shares"])
        total_spent_in_eur += sum(df_up_to_date["total_outgoing_in_eur"])
        total_fees_paid += sum(df_up_to_date[["currency_exchange_fee", "fixed_transaction_fee", "variable_transaction_fee"]])

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
                                             "valuation": value_at_date,
                                             "price_paid": total_spent_in_eur,
                                             "adjusted_bep": exchange_rate_adjusted_bep,
                                             "total_fees_paid": total_fees_paid,
                                             "absolute_profit": absolute_profit,
                                             "percent_profit": perc_profit,
                                             "percent_fees":perc_fees
                                            }, ignore_index=True).reset_index(drop=True)

        # increment the valuation date
        valuation_date += selected_period

    return valuation_df


def calculate_daily_account_balance(transactions_df):

    # read in the table of when money was deposited and create a table showing the balance changes
    deposites_df = pd.read_csv("data/deposites.csv", index_col="date")
    balance_changes_df = deposites_df["amount"].cumsum()

    # create a dataframe with a row for each day
    daily_rows_df = pd.DataFrame({"date": pd.date_range(balance_changes_df.index.min(), datetime.now().date(), freq='D')}).set_index("date")

    # expand out our balance dataframe to have a row for each day
    daily_balance_df = pd.merge(daily_rows_df, balance_changes_df, how="left", left_index=True, right_index=True).ffill(axis=0)

    return daily_balance_df.reset_index()


def calculate_debit_money(daily_account_balance_df, transactions_df):

    # create a dataframe with a row for each day
    daily_rows_df = pd.DataFrame({"date": pd.date_range(transactions_df["date"].min(), datetime.now().date(), freq='D')}).set_index("date")

    # expand out our balance dataframe to have a row for each day
    daily_transactions_cost_df = pd.merge(daily_rows_df, transactions_df.set_index("date"), how="left", left_index=True, right_index=True).ffill(axis=0)

    daily_transactions_cost_df["share_cost_in_euro"]

