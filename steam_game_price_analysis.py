from datetime import datetime
import pandas as pd

df = pd.read_csv('2024-11-02_Steam_top100_most_played_games_raw.csv')

def clean_price(price):
    if not isinstance(price, str):
        return 0.0
    if 'Free To Play' in price or 'Coming Soon' in price:
        return 0.0
    try:

        price = price.split('\n')[-1].replace('€', '').replace(',', '.')
        return float(price)
    except:
        return 0.0


df['price'] = df['price'].apply(clean_price)


df['current_players'] = pd.to_numeric(df['current_players'].str.replace(',', ''), errors='coerce')
df['peak_players_today'] = pd.to_numeric(df['peak_players_today'].str.replace(',', ''), errors='coerce')


df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')


print(df.head())

paid_games = df[df['price'] > 0]

current_date = datetime.now().strftime('%Y-%m-%d')

paid_games.to_csv(f'{current_date}_Steam_top100_most_played_games_clean.csv', index=False)

print(f"Found {len(paid_games)} paid games out of {len(df)} total games")

print(paid_games.head())