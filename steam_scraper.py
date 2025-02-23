import time
from datetime import datetime
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

class SteamScraper:
    def __init__(self, output_path='games_fresh.csv'):
        self.output_path = output_path
        self.url = 'https://store.steampowered.com/charts/mostplayed'
        self.options = Options()
        self.options.add_argument('--headless=new')

        self.column_names = ['name', 'price', 'current_players', 'peak_players_today']
        self.class_names = ['_1n_4-zvf0n4aqGEksbgW9N', '_3j4dI1yA7cRfCvK8h406OB', '_3L0CDDIUaOKTGfqdpqmjcy', 'yJB7DYKsuTG2AYhJdWTIk']

    def scrape(self):
        driver = webdriver.Chrome(options=self.options)
        driver.get(self.url)
        time.sleep(2)

        games = driver.find_elements(By.CLASS_NAME, "_2-RN6nWOY56sNmcDHu069P")

        data = {col: [] for col in self.column_names}
        data['date'] = datetime.now()

        for game in games:
            for count, column in enumerate(self.column_names):
                try:
                    value = game.find_element(By.CLASS_NAME, self.class_names[count]).text
                    data[column].append(value)
                except NoSuchElementException:
                    data[column].append(np.nan)

        driver.quit()
        self.save_to_csv(data)

    def save_to_csv(self, data):
        df = pd.DataFrame(data)
        df['price'] = np.where(
            df['price'].str.contains('Free To Play|Coming Soon', na=True),
            0.0,
            pd.to_numeric(df['price'].str.extract(r'(\d+,\d+|\d+\.\d+)')[0]
                          .str.replace(',', '.'), errors='coerce').fillna(0.0)
        )

        df[['current_players', 'peak_players_today']] = df[['current_players', 'peak_players_today']].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')

        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df.to_csv(self.output_path, index=False)