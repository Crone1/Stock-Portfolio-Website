# For scraping the data
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# For storing the data
import json

# Read in config of my chromedriver location
with open("configs/configs.yaml", "r") as config:
    c = yaml.load(config, Loader=yaml.FullLoader)
chromedriver_location = c["chromedriver_location"]


def update_company_tickers():
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
    for _ in tqdm(range(num_pages)):
        # iteratively scrape each ticker and company name
        num_tickers_on_page = len(driver.find_elements(By.XPATH, "/html/body/div/div/main/div/div/div[2]/table/tbody/tr"))
        for i in tqdm(range(1, num_tickers_on_page), leave=False):
            ticker = driver.find_element(By.XPATH, '/html/body/div/div/main/div/div/div[2]/table/tbody/tr[{}]/td[1]/a'.format(i)).text
            company_name = driver.find_element(By.XPATH, '/html/body/div/div/main/div/div/div[2]/table/tbody/tr[{}]/td[2]'.format(i)).text
            ticker_to_company_map[ticker] = company_name

        # Change to the next page
        next_page_button = driver.find_element(By.XPATH, '/html/body/div/div/main/div/div/nav/button[2]')
        next_page_button.click()

    # store this scraped data as json file
    with open('ticker_to_name.json', 'w') as file:
        json.dump(ticker_to_company_map, file)