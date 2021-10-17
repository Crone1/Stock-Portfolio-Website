
# general packages
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import datetime as dt
from datetime import datetime
from dateutil.parser import parse
import os

# For selecting period of valuation table
from dateutil.relativedelta import relativedelta

# Checking & making directories
from pathlib import Path

# For plotting the data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# For sending automatic email
import smtplib, ssl
from pretty_html_table import build_table
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def col_to_date(col):
    return [d.date() for d in pd.to_datetime(col)]


def plot_stock_prices(username, transactions_df):

    # read in the exchange data
    all_stock_prices_df = pd.read_csv("../data/stock_price_data.csv")

    # ensure the date column is stored as a date
    all_stock_prices_df["date"] = col_to_date(all_stock_prices_df["date"])

    # Get a subset of the stocks specific to the transactions df
    unique_tickers = transactions_df[["stock_ticker", "exchange_ticker"]].drop_duplicates()
    unique_stock_tickers = list(unique_tickers["stock_ticker"])
    unique_exchange_tickers = list(unique_tickers["exchange_ticker"])
    plot_cols = []
    for col in all_stock_prices_df.drop(columns=["date"]).columns:
        col_stock_ticker, col_exchange_ticker = col.rstrip(")").split(" (")
        for unique_stock_ticker, unique_exchange_ticker in zip(unique_stock_tickers, unique_exchange_tickers):
            if (col_stock_ticker == unique_stock_ticker) and (col_exchange_ticker == unique_exchange_ticker):
                plot_cols.append(col)
    subset_stock_price_df = all_stock_prices_df[["date"] + plot_cols]

    # Get a subset of these stock prices to match the dates in the transactions df
    ticker_start_dates = col_to_date([str(min(transactions_df[(transactions_df["stock_ticker"] == stock) & (transactions_df["exchange_ticker"] == exchange)]["date"]))[:10] for stock, exchange in zip(unique_stock_tickers, unique_exchange_tickers)])
    stock_price_df = pd.DataFrame()
    for date, stock_ticker, exchange_ticker in zip(ticker_start_dates, unique_stock_tickers, unique_exchange_tickers):
        col = "{} ({})".format(stock_ticker, exchange_ticker)
        # check if the stock is in this stock price df
        if col not in subset_stock_price_df.columns:
            print("Cannot create plot for '{}' as it is not in the scraped stock price data".format(col))
            continue
        specific_col = subset_stock_price_df[["date", col]]
        subset_cols_rows = specific_col[specific_col["date"] > date].set_index(["date"])
        stock_price_df = pd.concat([stock_price_df, subset_cols_rows], axis=1)
    stock_price_df = stock_price_df.reset_index().rename(columns={"index": "date"})

    # plot the data
    plot_titles = list(stock_price_df.drop(columns=["date"]).columns)
    num_rows = (len(plot_titles)+1)//2
    axes = stock_price_df.plot(x="date", subplots=True, title=plot_titles, figsize=(16, len(plot_titles) * 2), layout=(num_rows, 2), legend=None, xlabel="Date", rot=90)

    # Set figure title
    fig = plt.gcf()
    fig.suptitle("Stock Prices Over Time", fontsize=25)
    fig.subplots_adjust(top=0.836 + (num_rows * 0.014))

    # Save the figure as a png
    Path("../saved_figures/{}".format(username)).mkdir(parents=True, exist_ok=True)
    plt.savefig("../saved_figures/{}/all_stock_prices.png".format(username))


def plot_exchange_rates(username, transactions_df, base_currency):

    # read in the exchange data
    all_currency_rates_df = pd.read_csv("../data/{}_currency_exchange_data.csv".format(base_currency))

    # ensure the date column is stored as a date
    all_currency_rates_df["date"] = col_to_date(all_currency_rates_df["date"])

    # Get a subset of the stocks specific to the transactions df
    unique_currencies = list(transactions_df["currency"].drop_duplicates())
    subset_currency_rates_df = all_currency_rates_df[["date"] + unique_currencies]

    # Get a subset of these stock prices to match the dates in the transactions df
    currency_start_dates = col_to_date([str(min(transactions_df[transactions_df["currency"] == currency]["date"]))[:10] for currency in unique_currencies])
    currency_rates_df = pd.DataFrame()
    for date, currency in zip(currency_start_dates, unique_currencies):
        # check if the stock is in this stock price df
        if currency not in subset_currency_rates_df.columns:
            print("Cannot create plot for '{}' as it is not in the scraped exchange rate data".format(currency))
            continue
        specific_curr = subset_currency_rates_df[["date", currency]]
        subset_curr_rows = specific_curr[specific_curr["date"] > date].set_index(["date"])
        currency_rates_df = pd.concat([currency_rates_df, subset_curr_rows], axis=1)
    currency_rates_df = currency_rates_df.reset_index().rename(columns={"index": "date"})

    # plot the data
    num_plots = len(currency_rates_df.columns) - 1
    num_rows = (num_plots+1)//2
    axes = currency_rates_df.plot(x="date", subplots=True, figsize=(16, num_plots * 2), layout=(num_rows, 2), xlabel="Date", rot=90)

    # Set figure title
    fig = plt.gcf()
    fig.suptitle("Exchange Rate to {} Over Time".format(base_currency), fontsize=25)
    fig.subplots_adjust(top=0.836 + (num_rows * 0.014))

    # Save the figure as a png
    Path("../saved_figures/{}".format(username)).mkdir(parents=True, exist_ok=True)
    plt.savefig("../saved_figures/{}/all_exchange_rates.png".format(username))


def get_stock_price(date, stock_ticker, exchange_ticker):

    # read in the stock price data
    scraped_prices = pd.read_csv("../data/stock_price_data.csv")

    # Turn the date column to a date and set it as the index
    scraped_prices["date"] = col_to_date(scraped_prices["date"])
    scraped_prices.set_index("date", inplace=True)

    # Fill in weekend values if they are missing

    df_with_all_days = scraped_prices.reindex([d.date() for d in pd.date_range(start=min(scraped_prices.index), end=datetime.today().date())])
    all_prices = df_with_all_days.fillna(method='ffill')

    # Get combined ticker
    ticker = "{} ({})".format(stock_ticker, exchange_ticker)

    # Return the price for the stock on the date we're looking for
    return float(all_prices.loc[date, ticker])


def get_exchange_rate(date, base_currency, currency):

    # read in the exchange rate data
    scraped_rates = pd.read_csv("../data/{}_currency_exchange_data.csv".format(base_currency))

    # Turn the date column to a date and set it as the index
    scraped_rates["date"] = col_to_date(scraped_rates["date"])
    scraped_rates.set_index("date", inplace=True)

    # Fill in weekend values if they are missing
    df_with_all_days = scraped_rates.reindex([d.date() for d in pd.date_range(start=min(scraped_rates.index), end=max(scraped_rates.index))])
    all_rates = df_with_all_days.fillna(method='ffill')

    # Return the rate for the currency on the date we're looking for
    return float(all_rates.loc[date, currency])


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
    tickers_in_df = stock_df[["stock_ticker", "exchange_ticker"]].drop_duplicates()
    currencies_in_df = stock_df["currency"].drop_duplicates()
    if len(tickers_in_df) == 1 and len(currencies_in_df) == 1:
        stock_ticker = tickers_in_df.loc[0, "stock_ticker"]
        exchange_ticker = tickers_in_df.loc[0, "exchange_ticker"]
        currency = currencies_in_df[0]
    else:
        raise ValueError('Error! Got multiple ticker symbols in the passed table')

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
            share_price = get_stock_price(valuation_date, stock_ticker, exchange_ticker)
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
                                            "percent_fees": perc_fees,
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


def create_all_valuation_tables(transactions, base_currency, val_period="daily"):

    # Define a list of the primary keys to each stock
    stock_primary_keys = transactions[["stock_ticker", "exchange_ticker"]].drop_duplicates()

    # Iterate through each stock and create a valuation table for it
    val_tables_map = {}
    pbar = tqdm(total=len(stock_primary_keys))
    for _, (stock_ticker, exchange_ticker) in stock_primary_keys.iterrows():
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


def plot_background_colours(axis, min_val, max_val):

    axis_buffer = (max_val - min_val) * 0.05
    min_to_colour, max_to_colour = min_val-axis_buffer-1, max_val+axis_buffer+1
    abs_max = max(abs(max_val), abs(min_val))
    profit_top_cutoff, profit_mid_cutoff, loss_mid_cutoff, loss_bottom_cutoff = abs_max * 0.67, abs_max * 0.33, -abs_max * 0.33, -abs_max * 0.67

    # Plot the different colour bands
    if min_to_colour < loss_bottom_cutoff:
        axis.axhspan(min_to_colour, loss_bottom_cutoff, facecolor="darkred", alpha=0.5)
        axis.axhspan(loss_bottom_cutoff, loss_mid_cutoff, facecolor="red", alpha=0.5)
        axis.axhspan(loss_mid_cutoff, 0, facecolor="lightcoral", alpha=0.5)
    elif min_to_colour < loss_mid_cutoff:
        axis.axhspan(min_to_colour, loss_mid_cutoff, facecolor="red", alpha=0.5)
        axis.axhspan(loss_mid_cutoff, 0, facecolor="lightcoral", alpha=0.5)
    elif min_to_colour < 0:
        axis.axhspan(min_to_colour, 0, facecolor="lightcoral", alpha=0.5)
    if max_to_colour > profit_top_cutoff:
        axis.axhspan(profit_top_cutoff, max_to_colour, facecolor="darkgreen", alpha=0.5)
        axis.axhspan(profit_mid_cutoff, profit_top_cutoff, facecolor="green", alpha=0.5)
        axis.axhspan(0, profit_mid_cutoff, facecolor="palegreen", alpha=0.5)
    elif max_to_colour > profit_mid_cutoff:
        axis.axhspan(profit_mid_cutoff, max_to_colour, facecolor="green", alpha=0.5)
        axis.axhspan(0, profit_mid_cutoff, facecolor="palegreen", alpha=0.5)
    elif max_to_colour > 0:
        axis.axhspan(0, max_to_colour, facecolor="palegreen", alpha=0.5)

    # plot a line at 0 to distinguish profit VS loss
    axis.axhline(0, color="white", linewidth=3)

    # Set the Y-axis limits
    axis.set_ylim(bottom=min_val-axis_buffer, top=max_val+axis_buffer)


def visualise_profit_over_time(username, map_stock_to_val_table):

    # Find how many stocks there are to visualise
    num_tabs = 0
    for k,v in map_stock_to_val_table.items():
        num_tabs += 1 if not v.empty else 0

    # Define the figure for plotting
    num_rows = (num_tabs+1)//2
    fig, ax = plt.subplots(figsize=(16, num_tabs * 2), nrows=num_rows, ncols=2, gridspec_kw={"hspace":0.5})

    # Define the format for the y-ticks
    @ticker.FuncFormatter
    def major_formatter(val, pos):
        add_euro = "€{}".format(int(abs(val)))
        return "-" + add_euro if val < 0 else add_euro

    # Iterate through these valuation tables and plot them
    i = 0
    for (stock_ticker, exchange_ticker), val_table in map_stock_to_val_table.items():

        # Remove stocks without a valuation table
        if val_table.empty:
            continue

        # Define the axes to plot on 
        col = i % 2
        row = i//2
        i += 1

        # Plot the background colors
        min_val, max_val = min(0, min(val_table["absolute_profit"])), max(0, max(val_table["absolute_profit"]))
        plot_background_colours(ax[row, col], min_val, max_val)

        # Plot the stocks profit over time
        ax[row, col].plot(val_table["date"], val_table["absolute_profit"], color="black")
        ax[row, col].set_title("{} ({})".format(stock_ticker, exchange_ticker), fontsize=20)
        ax[row, col].set_ylabel("Profit", fontsize=15)
        ax[row, col].tick_params(axis='x', labelrotation=45)
        ax[row, col].yaxis.set_major_formatter(major_formatter)
        ax[row, col].set_xlim(left=min(val_table["date"]), right=max(val_table["date"]))
    
    # Add a title to the figure
    fig.suptitle("Profit Over Time for Stocks", fontsize=30)
    fig.subplots_adjust(top=0.836 + (num_rows * 0.014))

    # Save the figure as a png
    Path("../saved_figures/{}".format(username)).mkdir(parents=True, exist_ok=True)
    plt.savefig("../saved_figures/{}/all_stock_profit_over_time.png".format(username))


def get_portflio_on_date(map_stock_to_val_table, date_str=str(datetime.now().date() - relativedelta(days=1))):

    # turn the date string to a date
    if str(datetime.now().date() - relativedelta(days=1)) == date_str:
        print("Analysing portfolio for yesterday -", date_str)
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
            raise Exception("The valuation table for stock '{}' ({}) has more than one row for the date '{}'".format(stock_ticker, exchange_ticker, date))

        # Populate a dict with the values for this stock on the selected date
        old_columns = list(val_row.columns)
        val_row["stock_ticker"] = stock_ticker
        val_row["exchange_ticker"] = exchange_ticker
        ordered_val_row = val_row[["stock_ticker", "exchange_ticker"] + old_columns].reset_index(drop=True)

        # Add this row to the full dataframe
        portfolio_on_date = pd.concat([portfolio_on_date, ordered_val_row], axis=0)

    if portfolio_on_date.empty:
        raise ValueError("Choose a new date - No valuation was calculated for the given date '{}'".format(date))

    return portfolio_on_date.reset_index(drop=True)


def visualise_portfolio_pie_chart(username, portfolio_df, base_currency):

    # Define the labels to name the components in the pie chart
    labels = portfolio_df[["stock_ticker", "exchange_ticker"]].apply(lambda x: "{} ({})".format(x["stock_ticker"], x["exchange_ticker"]), axis=1)

    # Define a figure to plot on
    fig, ax = plt.subplots(figsize=(16, 16), nrows=1, ncols=2, gridspec_kw={"wspace":0.45})

    # set the colour cycle to use in the pie charts
    colormap = ['b', 'g', 'r', 'c', 'm', 'y', 'darkorange', 'grey', 'lime', 'cornflowerblue', 'lightcoral', 'khaki', 'violet', 'yellow', 'sandybrown', 'deepskyblue', 'deeppink', 'honeydew', 'lightsteelblue', 'blueviolet']

    ax[0].set_prop_cycle('color', colormap[:len(portfolio_df)]) #plt.cm.Paired(np.linspace(0,1,len(portfolio_df))))
    ax[1].set_prop_cycle('color', colormap[:len(portfolio_df)])

    # Plot a pie chart of the portolio breakdown based on the where I paid the money in
    paid_col = portfolio_df["price_paid_{}".format(base_currency)]
    ax[0].pie(x=paid_col, labels=labels, autopct='%.1f%%', radius=1.1, pctdistance=1.1, labeldistance=1.2, rotatelabels=True)
    ax[0].set_title("Where I Allocated Money", fontsize=20, pad=75)

    # Plot a pie chart of the portolio breakdown based on the stocks current value
    val_col = portfolio_df["current_valuation_{}".format(base_currency)]
    percent_lost = 1 - (sum(val_col)/sum(paid_col))
    ax[1].pie(x=val_col, labels=labels, autopct='%.1f%%', radius=1.1 * (1 - percent_lost/2), pctdistance=1.1, labeldistance=1.2, rotatelabels=True)
    ax[1].set_title('Current Value ({:.2f}% of amount paid)'.format(100*(1 - percent_lost)), fontsize=20, pad=75)

    # Save the figure as a png
    Path("../saved_figures/{}".format(username)).mkdir(parents=True, exist_ok=True)
    plt.savefig("../saved_figures/{}/portfolio_pie_chart.png".format(username))


def format_portfolio_df(portfolio_df, base_currency):

    return portfolio_df


def create_portfolio_image_attachments_for_email(username):

    # Define the loctions of where the attachments are
    exchange_rate_filename = "../saved_figures/{}/all_exchange_rates.png".format(username)
    stock_prices_filename = "../saved_figures/{}/all_stock_prices.png".format(username)
    stock_profit_over_time_filename = "../saved_figures/{}/all_stock_profit_over_time.png".format(username)
    portfolio_pie_chart_filename = "../saved_figures/{}/portfolio_pie_chart.png".format(username)

    # Iterate through each and create a valid attachment
    filename_list = [exchange_rate_filename, stock_prices_filename, stock_profit_over_time_filename, portfolio_pie_chart_filename]
    attachment_list = []
    for filename in filename_list:

        # Remove the directories from the filename
        name = filename.split("/")[-1]

        # Open PDF file in binary mode
        with open(filename, "rb") as file:
            # Add file as application/octet-stream - Email client can download this automatically as an attachment
            attachment = MIMEApplication(file.read())

        # Add header as key/value pair to attachment part
        attachment.add_header("Content-Disposition", "attachment", filename=name)

        # Add the attachment to the list of all attachments
        attachment_list.append(attachment)

    return attachment_list


def send_portfolio_update_from_gmail_account(username, reciever_email, portfolio_df):

    # Define the variables to connect to gmail's servers
    smtp_server = 'smtp.gmail.com'
    port_no = 465
    context = ssl.create_default_context()

    # Define the sender's email address and the get it's password
    sender_email = 'portfolio.updater@gmail.com'
    sender_email_password = os.environ.get('PORTFOLIO_UPDATE_EMAIL_PASSWORD')

    # Setup the structure of the email
    message = MIMEMultipart("mixed")
    message["Subject"] = "Portfolio Update - {date}".format(date=datetime.today().date() - relativedelta(days=1))
    message["From"] = sender_email
    message["To"] = reciever_email

    # Define the HTML message
    html_message_p1 = """\
    <html>
      <body>
        <p>Hi,<br>
           <br>
           Just a quick email to give you another regular portfolio update.<br>
           <br>
           We have attached images showing where your portfolio is currently valued and how it has performed over time.<br>
           The following table contains a summary of each stock in more detail:<br>
           """
    html_table = build_table(portfolio_df, font_size="medium", text_align="center", index=False, color='blue_light')
    html_message_p2 = """\
           <br>
           Regards,<br>
           Portman<br>
        </p>
      </body>
    </html>
    """
    full_html_message = html_message_p1 + html_table + html_message_p2

    # Attach this HTML message to the email
    text = MIMEText(full_html_message, "html")
    message.attach(text)

    # Attach the png portfolio images to the email
    attachment_list = create_portfolio_image_attachments_for_email(username)
    for attachment in attachment_list:
        message.attach(attachment)

    # Login to the email and send the message
    with smtplib.SMTP_SSL(smtp_server, port_no, context=context) as server:
        server.login(sender_email, sender_email_password)
        server.sendmail(sender_email, reciever_email, message.as_string())
    
    return "Email Sent Successfully"

