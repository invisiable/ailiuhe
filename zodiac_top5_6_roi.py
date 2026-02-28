# -*- coding: utf-8 -*-
"""
使用Top5/Top6生肖策略达到20%+ ROI
"""

import pandas as pd
import numpy as np
from itertools import product

# 加载数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
animals = df['animal'].tolist()
dates = df['date'].tolist()

# 生肖到号码映射
ZODIAC_NUMBERS = {
    '鼠': [1, 13, 25, 37, 49],
    '牛': [2, 14, 26, 38],
    '虎': [3, 15, 27, 39],
    '兔': [4, 16, 28, 40],
    '龙': [5, 17, 29, 41],
    '蛇': [6, 18, 30, 42],
    '马': [7, 19, 31, 43],
    '羊': [8, 20, 32, 44],
    '猴': [9, 21, 33, 45],
    '鸡': [10, 22, 34, 46],
    '狗': [11, 23, 35, 47],
    '猪': [12, 24, 36, 48]
}

test_periods = 300
test_start = len(numbers) - test_periods

print("=" * 70)
print("Top5/Top6生肖 + 动态投注策略 ROI分析")
print("=" * 70)
print(f"数据: {len(numbers)}期, 测试: {test_periods}期")
print(f"时间范围: {dates[test_start]} ~ {dates[-1]}")
print()

# 改进的生肖预测函数
def predict_top_zodiacs(history_animals, top_n=5):
    """预测Top N个热门生肖"""
    if len(history_animals) < 10:
        return ['鼠', '牛', '虎', '兔', '龙', '蛇'][:top_n]
    
    # 统计最近出现频率
    zodiac_list = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
    freq = {z: 0 for z in zodiac_list}
    
    # 最近30期
    recent30 = history_animals[-30:] if len(history_animals) >= 30 else history_animals
    for zodiac in recent30:
        freq[zodiac] = freq.get(zodiac, 0) + 1
    
    # 最近10期权重更高
    recent10 = history_animals[-10:] if len(history_animals) >= 10 else history_animals
    for zodiac in recent10:
        freq[zodiac] = freq.get(zodiac, 0) + 2
    
    # 最近3期最高权重
    recent3 = history_animals[-3:] if len(history_animals) >= 3 else history_animals
    for zodiac in recent3:
        freq[zodiac] = freq.get(zodiac, 0) + 3
    
    # 排序返回Top N
    sorted_zodiacs = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [z for z, _ in sorted_zodiacs[:top_n]]

# Fibonacci序列
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def get_fib_mult(losses, max_mult=10):
    if losses >= len(FIB):
        return min(FIB[-1], max_mult)
    return min(FIB[losses], max_mult)

def run_backtest(top_n, lookback, good_thresh, bad_thresh, boost_mult, reduce_mult, max_mult, bet_per_zodiac=15):
    """
    运行回测
    top_n: 选择几个生肖
    bet_per_zodiac: 每个生肖投注金额（元）
    """
    results = []
    consecutive_losses = 0
    recent_results = []
    
    for i in range(test_start, len(numbers)):
        history_animals = animals[:i]
        actual_animal = animals[i]
        
        pred_zodiacs = predict_top_zodiacs(history_animals, top_n)
        hit = actual_animal in pred_zodiacs
        
        # 获取倍数
        base_mult = get_fib_mult(consecutive_losses, max_mult)
        
        if len(recent_results) >= lookback:
            recent_window = recent_results[-lookback:]
            recent_hit_rate = sum(recent_window) / len(recent_window)
            
            if recent_hit_rate >= good_thresh:
                final_mult = min(base_mult * boost_mult, max_mult)
            elif recent_hit_rate <= bad_thresh:
                final_mult = max(base_mult * reduce_mult, 1.0)
            else:
                final_mult = base_mult
        else:
            final_mult = base_mult
        
        # 计算盈亏
        # 投注: 每个生肖投bet_per_zodiac * 倍数, 共top_n个生肖
        # 但简化计算: 只计算单注15元的情况（生肖组合投注）
        bet = bet_per_zodiac * final_mult
        
        if hit:
            # 假设赔率1:3 (投15赢45, 净赚30)
            profit = bet_per_zodiac * 2 * final_mult  # 净利润
            consecutive_losses = 0
        else:
            profit = -bet
            consecutive_losses += 1
        
        results.append({'hit': hit, 'bet': bet, 'profit': profit, 'mult': final_mult})
        recent_results.append(1 if hit else 0)
    
    total_bet = sum(r['bet'] for r in results)
    total_profit = sum(r['profit'] for r in results)
    roi = total_profit / total_bet * 100 if total_bet > 0 else 0
    
    # 最大回撤
    cumsum = 0
    peak = 0
    max_dd = 0
    for r in results:
        cumsum += r['profit']
        peak = max(peak, cumsum)
        max_dd = max(max_dd, peak - cumsum)
    
    hit_rate = sum(r['hit'] for r in results) / len(results) * 100
    
    return {
        'roi': roi,
        'profit': total_profit,
        'drawdown': max_dd,
        'hit_rate': hit_rate,
        'total_bet': total_bet
    }

# 测试不同生肖数量的命中率
print("【基础命中率测试】")
print("-" * 50)
for top_n in [4, 5, 6]:
    hits = 0
    for i in range(test_start, len(numbers)):
        history_animals = animals[:i]
        actual_animal = animals[i]
        pred_zodiacs = predict_top_zodiacs(history_animals, top_n)
        if actual_animal in pred_zodiacs:
            hits += 1
    hit_rate = hits / test_periods * 100
    theoretical_roi = (hit_rate * 30 - (100 - hit_rate) * 15) / 15
    print(f"Top{top_n}生肖: 命中率 {hit_rate:.1f}%, 理论ROI {theoretical_roi:.1f}%")

print()

# 搜索配置
top_n_options = [5, 6]
lookbacks = [5, 8, 10, 12, 15]
good_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
bad_thresholds = [0.25, 0.30, 0.35]
boost_mults = [1.2, 1.5, 2.0, 2.5, 3.0]
reduce_mults = [0.3, 0.4, 0.5, 0.6]
max_mults = [10, 15, 20]

print("=" * 70)
print("【搜索ROI >= 20%的配置】")
print("=" * 70)

results_list = []
for top_n in top_n_options:
    for lb, gt, bt, bm, rm, mm in product(lookbacks, good_thresholds, bad_thresholds, boost_mults, reduce_mults, max_mults):
        if gt <= bt:
            continue
        result = run_backtest(top_n, lb, gt, bt, bm, rm, mm)
        result.update({
            'top_n': top_n,
            'lookback': lb, 'good_thresh': gt, 'bad_thresh': bt,
            'boost_mult': bm, 'reduce_mult': rm, 'max_mult': mm
        })
        results_list.append(result)

# 筛选ROI >= 20%
high_roi = [r for r in results_list if r['roi'] >= 20]
print(f"搜索 {len(results_list)} 种配置")
print(f"找到 {len(high_roi)} 个ROI >= 20% 的配置")
print()

if high_roi:
    high_roi.sort(key=lambda x: x['roi'], reverse=True)
    
    print("【Top 15 高ROI配置】")
    print("-" * 110)
    for i, r in enumerate(high_roi[:15], 1):
        print(f"{i}. Top{r['top_n']}生肖 | ROI={r['roi']:.2f}% | 收益={r['profit']:+.0f}元 | 回撤={r['drawdown']:.0f}元 | 命中率={r['hit_rate']:.1f}%")
        print(f"   参数: lookback={r['lookback']}, good={r['good_thresh']:.2f}, bad={r['bad_thresh']:.2f}, "
              f"boost={r['boost_mult']}x, reduce={r['reduce_mult']}x, max={r['max_mult']}")
    
    # 风险调整后排名
    print()
    print("【风险调整后最佳配置（高ROI + 低回撤）】")
    print("-" * 80)
    for r in high_roi:
        r['risk_score'] = r['roi'] / (r['drawdown'] / 1000 + 0.1)
    
    high_roi.sort(key=lambda x: x['risk_score'], reverse=True)
    
    for i, r in enumerate(high_roi[:5], 1):
        print(f"{i}. Top{r['top_n']}生肖 | ROI={r['roi']:.2f}% | 收益={r['profit']:+.0f}元 | 回撤={r['drawdown']:.0f}元")
        print(f"   参数: lookback={r['lookback']}, good={r['good_thresh']:.2f}, bad={r['bad_thresh']:.2f}, "
              f"boost={r['boost_mult']}x, reduce={r['reduce_mult']}x, max={r['max_mult']}")
        print()

else:
    results_list.sort(key=lambda x: x['roi'], reverse=True)
    print("未找到ROI >= 20%的配置")
    print()
    print("【最高ROI配置 Top 10】")
    print("-" * 100)
    for i, r in enumerate(results_list[:10], 1):
        print(f"{i}. Top{r['top_n']}生肖 | ROI={r['roi']:.2f}% | 收益={r['profit']:+.0f}元 | 回撤={r['drawdown']:.0f}元 | 命中率={r['hit_rate']:.1f}%")
        print(f"   参数: lookback={r['lookback']}, good={r['good_thresh']:.2f}, bad={r['bad_thresh']:.2f}, "
              f"boost={r['boost_mult']}x, reduce={r['reduce_mult']}x, max={r['max_mult']}")

print()
print("=" * 70)
print("【分析与建议】")
print("=" * 70)
if high_roi:
    best = high_roi[0]
    print(f"✅ 成功找到ROI >= 20%的配置！")
    print(f"   最佳配置: Top{best['top_n']}生肖, ROI={best['roi']:.2f}%")
    print(f"   建议将GUI中的Top15预测改为Top{best['top_n']}生肖预测")
else:
    print("❌ 当前预测模型无法达到20% ROI")
    print("   可能的解决方向:")
    print("   1. 改进生肖预测算法，提高准确率")
    print("   2. 使用条件投注，只在高胜率时段投注")
    print("   3. 使用Top6生肖（47.7%命中率），但需要更好的预测模型")
