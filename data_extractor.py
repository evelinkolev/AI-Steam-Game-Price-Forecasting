from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import time
from datetime import datetime
import pandas as pd
import numpy as np

options = Options()
options.add_argument('--headless=new')
driver = webdriver.Chrome(options=options)
driver.get('https://store.steampowered.com/charts/mostplayed')
time.sleep(2)

games = driver.find_elements(By.CLASS_NAME, "_2-RN6nWOY56sNmcDHu069P")

data = {'name': [],
        'price': [],
        'current_players': [],
        'peak_players_today': [],
        'date': datetime.now()}

column_names = ['name', 'price', 'current_players', 'peak_players_today']
class_names = ['_1n_4-zvf0n4aqGEksbgW9N', '_3j4dI1yA7cRfCvK8h406OB', '_3L0CDDIUaOKTGfqdpqmjcy', 'yJB7DYKsuTG2AYhJdWTIk']

for game in games:
    for count in range(len(column_names)):
        try:
            name = game.find_element(By.CLASS_NAME, class_names[count]).text
            data[column_names[count]].append(name)
        except NoSuchElementException:
            data[column_names[count]].append(np.nan)

df = pd.DataFrame(data)
current_date = datetime.now().strftime('%Y-%m-%d')
df.to_csv(f'{current_date}_Steam_top100_most_played_games_raw.csv', index=False)

driver.quit()