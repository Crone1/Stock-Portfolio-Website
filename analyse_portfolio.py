
# general packages
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse


def add_columns_to_data(df):#, exchange_currencies):
    #if exchange_currencies_currencies["exchange_ticker"] = "USD":
    #    p_curr = "$"
    
    df["share_cost_in_$"] = df["num_shares"] * df["share_price"]
    df["currency_exchange_fee"] = [abs(v) for v in df["share_cost_in_$"] * 0.001]/df["exchange_rate"]
    df["fixed_transaction_fee"] = 0.5
    df["variable_transaction_fee"] = round(((0.004/df["exchange_rate"]) *abs(df["num_shares"])), 2)
    df["total_outgoing_in_eur"] = (df["share_cost_in_$"]/df["exchange_rate"]) + df["fixed_transaction_fee"] + df["variable_transaction_fee"]
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
    Extract from the transactions dataframe only the transactions which are associated with the given ticker

    Returns:
        Pandas Dataframe
    """

    stock_data = all_data[(all_data["stock_ticker"] == stock_ticker) & (all_data["exchange_ticker"] == exchange_ticker)].reset_index(drop=True)
    stock_data = add_columns_to_data(stock_data)
    stock_data = add_post_transaction_summary(stock_data)

    return stock_data


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
