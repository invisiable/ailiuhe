# -*- coding: utf-8 -*-
"""
原来动态投注基准策略 - 300期详细回测列表
参数：lookback=12, good_thresh=0.35, bad_thresh=0.20, boost=1.2, reduce=0.8, max=10
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.insert(0, '.')

# 加载数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
animals = df['animal'].tolist()
dates = df['date'].tolist()

# 使用Top15预测器
from top15_statistical_predictor import Top15StatisticalPredictor
predictor = Top15StatisticalPredictor()

# Fibonacci序列
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def get_fib_mult(losses, max_mult=10):
    if losses >= len(FIB):
        return min(FIB[-1], max_mult)
    return min(FIB[losses], max_mult)

# 配置参数（原来的动态投注基准策略）
LOOKBACK = 12
GOOD_THRESH = 0.35
BAD_THRESH = 0.20
BOOST_MULT = 1.2
REDUCE_MULT = 0.8
MAX_MULT = 10
BASE_BET = 15

test_periods = 300
test_start = len(numbers) - test_periods

print("=" * 80)
print("原来动态投注基准策略 - 300期详细回测")
print("=" * 80)
print(f"配置参数:")
print(f"  回看窗口: {LOOKBACK}期")
print(f"  增强阈值: {GOOD_THRESH*100:.0f}% (命中率>=35%时增强)")
print(f"  降低阈值: {BAD_THRESH*100:.0f}% (命中率<=20%时降低)")
print(f"  增强倍数: {BOOST_MULT}x")
print(f"  降低倍数: {REDUCE_MULT}x")
print(f"  最大倍数: {MAX_MULT}x")
print()
print(f"测试期数: {test_periods}期")
print(f"时间范围: {dates[test_start]} ~ {dates[-1]}")
print()

# 运行回测
results = []
consecutive_losses = 0
recent_results = []
cumulative_profit = 0
peak = 0
max_drawdown = 0

for i in range(test_start, len(numbers)):
    period_idx = i - test_start + 1
    history = numbers[:i]
    actual = numbers[i]
    date = dates[i]
    animal = animals[i]
    
    # 预测
    prediction = predictor.predict(history)
    hit = actual in prediction
    
    # 获取基础Fibonacci倍数
    base_mult = get_fib_mult(consecutive_losses, MAX_MULT)
    fib_idx = consecutive_losses if consecutive_losses < len(FIB) else len(FIB) - 1
    
    # 动态调整（先计算倍数，再更新历史）
    if len(recent_results) >= LOOKBACK:
        recent_window = recent_results[-LOOKBACK:]
        recent_hit_rate = sum(recent_window) / len(recent_window)
        
        if recent_hit_rate >= GOOD_THRESH:
            final_mult = min(base_mult * BOOST_MULT, MAX_MULT)
            strategy = "增强"
        elif recent_hit_rate <= BAD_THRESH:
            final_mult = max(base_mult * REDUCE_MULT, 1.0)
            strategy = "降低"
        else:
            final_mult = base_mult
            strategy = "正常"
    else:
        final_mult = base_mult
        recent_hit_rate = sum(recent_results) / len(recent_results) if recent_results else 0
        strategy = "初始"
    
    # 计算盈亏
    bet = BASE_BET * final_mult
    if hit:
        profit = 32 * final_mult  # 净盈利 (47元奖励 - 15元投注 = 32元)
        consecutive_losses = 0
    else:
        profit = -bet
        consecutive_losses += 1
    
    cumulative_profit += profit
    peak = max(peak, cumulative_profit)
    current_drawdown = peak - cumulative_profit
    max_drawdown = max(max_drawdown, current_drawdown)
    
    # 记录结果
    results.append({
        '期号': period_idx,
        '日期': date,
        '开奖号码': actual,
        '生肖': animal,
        '预测TOP15': str(prediction[:5]) + '...' if len(prediction) > 5 else str(prediction),
        'Fib基础倍数': base_mult,
        '最近命中率': f"{recent_hit_rate*100:.1f}%",
        '策略': strategy,
        '投注倍数': final_mult,
        '投注金额': bet,
        '是否命中': '✓' if hit else '✗',
        '当期盈亏': profit,
        '累计盈亏': cumulative_profit,
        '当前回撤': current_drawdown,
        '连续未中': consecutive_losses if not hit else 0,
        'Fib索引': fib_idx,
        '触及10倍': '是' if final_mult >= 10 else ''
    })
    
    # 更新历史（在计算后）
    recent_results.append(1 if hit else 0)

# 转换为DataFrame
df_results = pd.DataFrame(results)

# 计算汇总统计
total_hits = sum(1 for r in results if r['是否命中'] == '✓')
hit_rate = total_hits / len(results) * 100
total_bet = sum(r['投注金额'] for r in results)
total_profit = cumulative_profit
roi = total_profit / total_bet * 100 if total_bet > 0 else 0
times_10x = sum(1 for r in results if r['触及10倍'] == '是')
hits_10x = sum(1 for r in results if r['触及10倍'] == '是' and r['是否命中'] == '✓')

print("=" * 80)
print("【回测结果汇总】")
print("=" * 80)
print(f"总期数: {len(results)}期")
print(f"命中次数: {total_hits}次")
print(f"命中率: {hit_rate:.1f}%")
print(f"总投注: {total_bet:.0f}元")
print(f"净收益: {total_profit:+.0f}元")
print(f"ROI: {roi:.2f}%")
print(f"最大回撤: {max_drawdown:.0f}元")
print(f"10倍投注: {times_10x}次，命中{hits_10x}次")
print()

# 保存到CSV
output_file = 'original_dynamic_300periods_detail.csv'
df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"详细数据已保存到: {output_file}")
print()

# 显示前10期和后10期
print("【前10期详情】")
print("-" * 120)
print(df_results.head(10).to_string(index=False))
print()

print("【后10期详情】")
print("-" * 120)
print(df_results.tail(10).to_string(index=False))
print()

# 关键期数分析
print("=" * 80)
print("【关键期数分析】")
print("=" * 80)

# 最大盈利期
max_profit_idx = df_results['当期盈亏'].idxmax()
max_profit = df_results.loc[max_profit_idx]
print(f"最大盈利期: 第{max_profit['期号']}期 ({max_profit['日期']})")
print(f"  投注{max_profit['投注倍数']}倍，{max_profit['是否命中']}，盈亏{max_profit['当期盈亏']:+.0f}元")
print()

# 最大亏损期
max_loss_idx = df_results['当期盈亏'].idxmin()
max_loss = df_results.loc[max_loss_idx]
print(f"最大亏损期: 第{max_loss['期号']}期 ({max_loss['日期']})")
print(f"  投注{max_loss['投注倍数']}倍，{max_loss['是否命中']}，盈亏{max_loss['当期盈亏']:+.0f}元")
print()

# 最大回撤发生时
max_dd_idx = df_results['当前回撤'].idxmax()
max_dd_period = df_results.loc[max_dd_idx]
print(f"最大回撤期: 第{max_dd_period['期号']}期 ({max_dd_period['日期']})")
print(f"  回撤{max_dd_period['当前回撤']:.0f}元，累计盈亏{max_dd_period['累计盈亏']:+.0f}元")
print()

# 高倍投注分析
high_mult = df_results[df_results['投注倍数'] >= 5.0]
if len(high_mult) > 0:
    print(f"≥5倍投注: {len(high_mult)}次")
    high_mult_hits = sum(1 for _, r in high_mult.iterrows() if r['是否命中'] == '✓')
    print(f"  命中{high_mult_hits}次 ({high_mult_hits/len(high_mult)*100:.1f}%)")
    print(f"  日期: {', '.join(high_mult['日期'].tolist()[:5])}{'...' if len(high_mult) > 5 else ''}")
    print()

# 10倍投注详情
if times_10x > 0:
    mult_10x = df_results[df_results['触及10倍'] == '是']
    print(f"10倍投注详情: ({times_10x}次)")
    print("-" * 80)
    for _, r in mult_10x.iterrows():
        print(f"  第{r['期号']}期 ({r['日期']}): {r['是否命中']} 号码{r['开奖号码']}, 盈亏{r['当期盈亏']:+.0f}元")
    print()

# 连续未中分析
max_consecutive_loss = df_results['连续未中'].max()
print(f"最长连续未中: {max_consecutive_loss}期")
max_consec_idx = df_results['连续未中'].idxmax()
max_consec_period = df_results.loc[max_consec_idx]
print(f"  发生在第{max_consec_period['期号']}期 ({max_consec_period['日期']})")
print()

print("=" * 80)
print(f"完整详情请查看: {output_file}")
print("=" * 80)
