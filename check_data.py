import pandas as pd

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f'数据总条数: {len(df)}')
print(f'最新数据日期: {df.iloc[-1]["date"]}')
print(f'最新号码: {df.iloc[-1]["number"]}')
