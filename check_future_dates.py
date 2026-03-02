#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查数据中是否有未来日期"""

import pandas as pd
from datetime import datetime

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv')
df['date_obj'] = pd.to_datetime(df['date'])

# 当前日期
today = datetime(2026, 3, 2)

print(f"当前日期: {today.strftime('%Y/%m/%d')}")
print(f"数据最后日期: {df['date'].iloc[-1]}")
print(f"数据总条数: {len(df)}")

# 检查未来日期
future = df[df['date_obj'] > today]
print(f"\n未来日期数据: {len(future)}条")

if len(future) > 0:
    print("\n未来日期详情:")
    print(future[['date', 'number', 'animal', 'element']])

# 检查2026/2/26前后的数据
print("\n\n2026/2/26前后数据:")
target_date = datetime(2026, 2, 26)
nearby = df[(df['date_obj'] >= datetime(2026, 2, 24)) & (df['date_obj'] <= datetime(2026, 2, 28))]
print(nearby[['date', 'number', 'animal', 'element']])
