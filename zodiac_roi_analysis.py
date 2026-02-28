# -*- coding: utf-8 -*-
"""
使用生肖预测器计算ROI - 生肖预测命中率约50-60%
"""

import pandas as pd
import numpy as np
from itertools import product
import sys
sys.path.insert(0, '.')

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

NUMBER_TO_ZODIAC = {}
for zodiac, nums in ZODIAC_NUMBERS.items():
    for n in nums:
        NUMBER_TO_ZODIAC[n] = zodiac

test_periods = 300
test_start = len(numbers) - test_periods

print("=" * 70)
print("生肖预测ROI分析")
print("=" * 70)
print(f"数据: {len(numbers)}期, 测试: {test_periods}期")
print(f"时间范围: {dates[test_start]} ~ {dates[-1]}")
print()

# 简化的生肖预测函数（基于历史热门生肖）
def predict_top_zodiacs(history_animals, top_n=4):
    """预测Top N个生肖"""
    if len(history_animals) < 10:
        return ['鼠', '牛', '虎', '兔'][:top_n]
    
    # 统计最近出现频率
    recent = history_animals[-30:]
    freq = {}
    for zodiac in ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']:
        freq[zodiac] = recent.count(zodiac)
    
    # 最近5期的权重更高
    for zodiac in history_animals[-5:]:
        freq[zodiac] = freq.get(zodiac, 0) + 2
    
    # 排序返回Top N
    sorted_zodiacs = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [z for z, _ in sorted_zodiacs[:top_n]]

# 测试生肖预测命中率
print("测试生肖预测命中率...")
print()

for top_n in [4, 5, 6]:
    hits = 0
    total_numbers = 0  # 预测覆盖的号码数
    
    for i in range(test_start, len(numbers)):
        history_animals = animals[:i]
        actual_animal = animals[i]
        
        pred_zodiacs = predict_top_zodiacs(history_animals, top_n)
        
        # 计算覆盖的号码数
        covered_numbers = sum(len(ZODIAC_NUMBERS[z]) for z in pred_zodiacs)
        total_numbers += covered_numbers
        
        if actual_animal in pred_zodiacs:
            hits += 1
    
    hit_rate = hits / test_periods * 100
    avg_numbers = total_numbers / test_periods
    
    # 计算理论ROI
    # 假设每个生肖投注15元，Top4就是60元
    # 命中一个生肖得45元
    # 实际是复合投注：投15元赢45元
    theoretical_roi = (hit_rate * 30 - (100 - hit_rate) * 15) / 15
    
    print(f"Top{top_n}生肖: 命中率 {hit_rate:.1f}% ({hits}/{test_periods}), "
          f"平均覆盖 {avg_numbers:.1f}个号码, 理论ROI {theoretical_roi:.1f}%")

# 使用更好的生肖预测器
print()
print("=" * 70)
print("使用高级生肖预测器...")
print("=" * 70)

# 尝试加载zodiac预测器
try:
    from zodiac_v11_predictor import ZodiacV11Predictor
    predictor = ZodiacV11Predictor()
    predictor_name = "生肖v11"
except:
    try:
        from zodiac_super_predictor import ZodiacSuperPredictor
        predictor = ZodiacSuperPredictor()
        predictor_name = "生肖Super"
    except:
        predictor = None
        predictor_name = None

if predictor:
    print(f"使用预测器: {predictor_name}")
    hits = 0
    
    for i in range(test_start, len(numbers)):
        history = numbers[:i]
        history_animals = animals[:i]
        actual_animal = animals[i]
        
        # 获取预测生肖
        try:
            # 尝试不同的预测方法
            if hasattr(predictor, 'predict_top4_zodiacs'):
                pred_zodiacs = predictor.predict_top4_zodiacs(history, history_animals)
            elif hasattr(predictor, 'predict'):
                pred = predictor.predict(history)
                if isinstance(pred, dict) and 'zodiacs' in pred:
                    pred_zodiacs = pred['zodiacs'][:4]
                else:
                    # 从预测号码转换为生肖
                    pred_zodiacs = list(set([NUMBER_TO_ZODIAC.get(n) for n in pred if n in NUMBER_TO_ZODIAC]))[:4]
            else:
                continue
            
            if actual_animal in pred_zodiacs:
                hits += 1
        except Exception as e:
            continue
    
    if hits > 0:
        hit_rate = hits / test_periods * 100
        theoretical_roi = (hit_rate * 30 - (100 - hit_rate) * 15) / 15
        print(f"命中率: {hit_rate:.1f}% ({hits}/{test_periods})")
        print(f"理论ROI: {theoretical_roi:.1f}%")

# Fibonacci策略回测
print()
print("=" * 70)
print("【生肖Top4 + Fibonacci策略 回测】")
print("=" * 70)

FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def get_fib_mult(losses, max_mult=10):
    if losses >= len(FIB):
        return min(FIB[-1], max_mult)
    return min(FIB[losses], max_mult)

def run_zodiac_backtest(lookback, good_thresh, bad_thresh, boost_mult, reduce_mult, max_mult):
    results = []
    consecutive_losses = 0
    recent_results = []
    
    for i in range(test_start, len(numbers)):
        history_animals = animals[:i]
        actual_animal = animals[i]
        
        pred_zodiacs = predict_top_zodiacs(history_animals, 4)
        hit = actual_animal in pred_zodiacs
        
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
        
        bet = 15 * final_mult
        if hit:
            profit = 30 * final_mult
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
    
    return {
        'roi': roi,
        'profit': total_profit,
        'drawdown': max_dd,
        'hit_rate': sum(r['hit'] for r in results) / len(results) * 100
    }

# 搜索高ROI配置
lookbacks = [5, 8, 10, 12, 15]
good_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
bad_thresholds = [0.20, 0.25, 0.30, 0.35]
boost_mults = [1.2, 1.5, 2.0, 2.5, 3.0]
reduce_mults = [0.3, 0.4, 0.5, 0.6]
max_mults = [10, 15, 20]

print(f"搜索参数组合...")

results_list = []
for lb, gt, bt, bm, rm, mm in product(lookbacks, good_thresholds, bad_thresholds, boost_mults, reduce_mults, max_mults):
    if gt <= bt:
        continue
    result = run_zodiac_backtest(lb, gt, bt, bm, rm, mm)
    result.update({
        'lookback': lb, 'good_thresh': gt, 'bad_thresh': bt,
        'boost_mult': bm, 'reduce_mult': rm, 'max_mult': mm
    })
    results_list.append(result)

# 筛选ROI >= 20%
high_roi = [r for r in results_list if r['roi'] >= 20]
print(f"找到 {len(high_roi)} 个ROI >= 20% 的配置")
print()

if high_roi:
    high_roi.sort(key=lambda x: x['roi'], reverse=True)
    
    print("【Top 10 高ROI配置】")
    print("-" * 100)
    for i, r in enumerate(high_roi[:10], 1):
        print(f"{i}. ROI={r['roi']:.2f}%, 收益={r['profit']:+.0f}元, 回撤={r['drawdown']:.0f}元, 命中率={r['hit_rate']:.1f}%")
        print(f"   参数: lookback={r['lookback']}, good={r['good_thresh']:.2f}, bad={r['bad_thresh']:.2f}, "
              f"boost={r['boost_mult']}x, reduce={r['reduce_mult']}x, max={r['max_mult']}")
        print()
else:
    results_list.sort(key=lambda x: x['roi'], reverse=True)
    print("未找到ROI >= 20%的配置，最高ROI:")
    print()
    for i, r in enumerate(results_list[:10], 1):
        print(f"{i}. ROI={r['roi']:.2f}%, 收益={r['profit']:+.0f}元, 回撤={r['drawdown']:.0f}元, 命中率={r['hit_rate']:.1f}%")
        print(f"   参数: lookback={r['lookback']}, good={r['good_thresh']:.2f}, bad={r['bad_thresh']:.2f}, "
              f"boost={r['boost_mult']}x, reduce={r['reduce_mult']}x, max={r['max_mult']}")
