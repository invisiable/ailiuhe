"""分析当前生肖TOP4策略的性能"""
import pandas as pd
import numpy as np

# 读取300期回测数据
df = pd.read_csv('zodiac_top4_backtest_300periods.csv', encoding='utf-8-sig')

print(f"{'='*70}")
print("当前生肖TOP4策略性能分析（300期）")
print(f"{'='*70}\n")

# 基本统计
total_periods = len(df)
# is_hit列是中文字符串："是" 或 "否"
hits = (df['is_hit'] == '是').sum()
misses = total_periods - hits
hit_rate = hits / total_periods

print(f"总期数: {total_periods}")
print(f"命中次数: {hits}")
print(f"未中次数: {misses}")
print(f"命中率: {hit_rate*100:.2f}%\n")

# 计算最大连续不中
max_consecutive_misses = 0
current_consecutive_misses = 0
consecutive_miss_sequences = []

for i, is_hit in enumerate(df['is_hit']):
    if is_hit == '否':
        current_consecutive_misses += 1
    else:
        if current_consecutive_misses > 0:
            consecutive_miss_sequences.append({
                'length': current_consecutive_misses,
                'end_period': i
            })
        max_consecutive_misses = max(max_consecutive_misses, current_consecutive_misses)
        current_consecutive_misses = 0

# 检查最后是否仍在连续不中
if current_consecutive_misses > 0:
    consecutive_miss_sequences.append({
        'length': current_consecutive_misses,
        'end_period': len(df)
    })
    max_consecutive_misses = max(max_consecutive_misses, current_consecutive_misses)

print(f"【连续不中统计】")
print(f"最大连续不中: {max_consecutive_misses}期")
print(f"连续不中>=5期次数: {sum(1 for s in consecutive_miss_sequences if s['length'] >= 5)}")
print(f"连续不中>=7期次数: {sum(1 for s in consecutive_miss_sequences if s['length'] >= 7)}")
print(f"连续不中>=10期次数: {sum(1 for s in consecutive_miss_sequences if s['length'] >= 10)}\n")

# 显示所有>=5期的连续不中
print("【连续不中>=5期的详细情况】")
long_misses = [s for s in consecutive_miss_sequences if s['length'] >= 5]
for seq in sorted(long_misses, key=lambda x: x['length'], reverse=True):
    end_idx = seq['end_period'] - 1
    start_idx = end_idx - seq['length'] + 1
    print(f"  {seq['length']}期不中: 第{start_idx+1}期 到 第{end_idx+1}期")

# 财务统计
if 'cumulative_profit' in df.columns:
    final_profit = df['cumulative_profit'].iloc[-1]
    total_bet = df['bet_amount'].sum()
    roi = (final_profit / total_bet) * 100
    
    print(f"\n【财务统计】")
    print(f"总投注: {total_bet:.2f}元")
    print(f"最终利润: {final_profit:+.2f}元")
    print(f"ROI: {roi:+.2f}%")

print(f"\n{'='*70}")
print(f"优化目标: 将最大连续不中从{max_consecutive_misses}期降低到4期")
print(f"{'='*70}")
