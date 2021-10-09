
# general packages
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import datetime as dt
from datetime import datetime
from dateutil.parser import parse

# For selecting period of valuation table
from dateutil.relativedelta import relativedelta

# For plottimng the data
import matplotlib.pyplot as plt


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
    num_rows = (num_plots+1)//2
    axes = currency_rates_df.plot(x="date", subplots=True, figsize=(16, num_plots * 2), layout=(num_rows, 2), xlabel="Date", rot=90)

    # Set figure title
    fig = plt.gcf()
    fig.suptitle("Exchange Rate to {} Over Time".format(base_currency), fontsize=25)
    fig.subplots_adjust(top=0.836 + (num_rows * 0.014))


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
    num_rows = (num_plots+1)//2
    axes = stock_price_df.plot(x="date", subplots=True, figsize=(16, num_plots * 2), layout=(num_rows, 2), xlabel="Date", rot=90)

    # Set figure title
    fig = plt.gcf()
    fig.suptitle("Stock Prices Over Time", fontsize=25)
    fig.subplots_adjust(top=0.836 + (num_rows * 0.014))


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
    now_datetime = datetime.now()
    max_valuation_date = now_datetime.date() if now_datetime.time() > dt.time(18,0,0) else now_datetime.date() - relativedelta(days=1)
    prev_val_date = parse("1900-01-01").date()
    valuation_date = parse(str(stock_df.loc[0, "date"])).date()

    # Ensure the date column is stored as a date tyoe
    stock_df["date"] = col_to_date(stock_df["date"])

    # initialise the other variables
    shares_owned, total_spent_in_eur, total_fees_paid = 0, 0, 0

    # iteratively valuate the shares
    valuation_df = pd.DataFrame(columns=["date", "share_price", "exchange_rate", "num_shares_owned", "current_valuation_{}".format(base_currency), "price_paid_{}".format(base_currency), "adjusted_bep", "total_fees_paid", "absolute_profit", "percent_profit", "percent_fees"])
    pbar = tqdm(total=(max_valuation_date - valuation_date)//(now_datetime + selected_period - now_datetime), leave=False)
    while valuation_date <= max_valuation_date:

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
            print("Error retrieving stock price for ticker: '{}' ({}). No valuation table calculated for this stock!".format(stock_ticker, exchange_ticker))
            val_table = pd.DataFrame()

        # Store a dictionary mapping the keys to the stocks valuation tables
        val_tables_map[(stock_ticker, exchange_ticker)] = val_table
        pbar.update(1)

    return val_tables_map


def visualise_profit_over_time(map_stock_to_val_table):

    # Find how many stocks there are to visualise
    num_tabs = 0
    for k,v in map_stock_to_val_table.items():
        num_tabs += 1 if not v.empty else 0

    # Define the figure for plotting
    num_rows = (num_tabs+1)//2
    fig, ax = plt.subplots(figsize=(16, num_tabs * 3), nrows=num_rows, ncols=2, gridspec_kw={"hspace":0.35})

    # Iterate through these valuation tables and plot them
    i = 0
    for (stock_ticker, exchange_ticker), val_table in map_stock_to_val_table.items():

        # Remove stocks without a valuation table
        if val_table.empty:
            continue

        # Plot the stocks profit over time
        col = i % 2
        row = i//2
        ax[row, col].plot(val_table["date"], val_table["absolute_profit"])
        ax[row, col].set_title("{} ({})".format(stock_ticker, exchange_ticker))
        ax[row, col].set_ylabel("Profit", fontsize=10)
        ax[row, col].tick_params(labelrotation=90)
        i += 1
    
    # Add a title to the figure
    fig.suptitle("Profit Over Time for Stocks", fontsize=20)
    fig.subplots_adjust(top=0.836 + (num_rows * 0.014))


def get_portflio_on_date(map_stock_to_val_table, date_str=str(datetime.now().date())):
    
    # turn the date string to a date
    if str(datetime.now().date()) == date_str:
        print("Analysing portfolio for today -", date_str)
    date = datetime.strptime(date_str, "%Y-%m-%d").date()

    # create a table of the stock values for this day
    portfolio_on_date = pd.DataFrame()
    for (stock_ticker, exchange_ticker), val_tab in map_stock_to_val_table.items():
        # Remove entries with no valuation table at all
        if val_tab.empty:
            continue

        # Find the data associated with the selected date
        val_row = val_tab[val_tab["date"] == date]

        # Remove entries where there was no data for the selected date
        if val_row.empty:
            continue

        # Ensure the table only has one row
        try:
            assert len(val_row) == 1
        except:
            print("The valuation table for stock '{}' ({}) has more than one row for the date '{}'".format(stock_ticker, exchange_ticker, date))
            raise TableSizeError

        # Populate a dict with the values for this stock on the selected date
        old_columns = list(val_row.columns)
        val_row["stock_ticker"] = stock_ticker
        val_row["exchange_ticker"] = exchange_ticker
        ordered_val_row = val_row[["stock_ticker", "exchange_ticker"] + old_columns].reset_index(drop=True)

        # Add this row to the full dataframe
        portfolio_on_date = pd.concat([portfolio_on_date, ordered_val_row], axis=0)

    return portfolio_on_date


def visualise_portfolio_pie_chart(portfolio_df, base_currency):

    # Define the labels to name the components in the pie chart
    labels = portfolio_df[["stock_ticker", "exchange_ticker"]].apply(lambda x: "{} ({})".format(x["stock_ticker"], x["exchange_ticker"]), axis=1)

    # Define a figure to plot on
    fig, ax = plt.subplots(figsize=(16, 16), nrows=1, ncols=2)

    # set the colour cycle to use in the pie charts
    colormap = ['b', 'g', 'r', 'c', 'm', 'y', 'darkorange', 'grey', 'lime', 'cornflowerblue', 'lightcoral', 'khaki', 'violet', 'yellow', 'sandybrown', 'deepskyblue', 'deeppink', 'honeydew', 'lightsteelblue', 'blueviolet']

    ax[0].set_prop_cycle('color', colormap[:len(portfolio_df)]) #plt.cm.Paired(np.linspace(0,1,len(portfolio_df))))
    ax[1].set_prop_cycle('color', colormap[:len(portfolio_df)])

    # Plot a pie chart of the portolio breakdown based on the where I paid the money in
    paid_col = portfolio_df["price_paid_{}".format(base_currency)]
    ax[0].pie(x=paid_col, labels=labels, autopct='%.1f%%', radius=1.1, pctdistance=1.1, labeldistance=1.2, rotatelabels=True)
    ax[0].set_title("Where I Allocated Money", fontsize=20)

    # Plot a pie chart of the portolio breakdown based on the stocks current value
    val_col = portfolio_df["current_valuation_{}".format(base_currency)]
    percent_lost = 1 - (sum(val_col)/sum(paid_col))
    ax[1].pie(x=val_col, labels=labels, autopct='%.1f%%', radius=1.1 * (1 - percent_lost/2), pctdistance=1.1, labeldistance=1.2, rotatelabels=True)
    ax[1].set_title('Current Value ({:.2f}% of amount paid)'.format(1 - percent_lost), fontsize=20)

