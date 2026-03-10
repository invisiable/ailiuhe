"""
TOP15预测命中率规律分析
专门针对最优智能投注策略中使用的PreciseTop15Predictor
分析连败后的命中率变化规律
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from precise_top15_predictor import PreciseTop15Predictor

def analyze_consecutive_misses_pattern(df):
    """
    分析TOP15预测的连续失败后命中率规律
    """
    print(f"\n{'='*70}")
    print(f"【TOP15预测 - 连续失败后命中率分析】")
    print(f"{'='*70}\n")
    
    # 初始化预测器
    predictor = PreciseTop15Predictor()
    
    # 连败统计
    consecutive_misses_stats = defaultdict(lambda: {'total': 0, 'hits': 0})
    
    # 模拟预测过程（需要足够的历史数据）
    lookback = 50  # 预测器需要的最小历史数据
    
    all_numbers = df['number'].tolist()
    
    for i in range(lookback, len(all_numbers)):
        train_data = all_numbers[:i]
        actual_number = all_numbers[i]
        
        # 预测TOP15
        predictions = predictor.predict(np.array(train_data))
        
        # 判断命中
        hit = actual_number in predictions
        
        # 更新预测器性能跟踪
        predictor.update_performance(predictions, actual_number)
        
        # 计算连续失败次数
        consecutive_misses = 0
        for j in range(i-1, max(i-15, lookback-1), -1):
            prev_train = all_numbers[:j]
            prev_actual = all_numbers[j]
            prev_predictions = predictor.predict(np.array(prev_train))
            
            if prev_actual not in prev_predictions:
                consecutive_misses += 1
            else:
                break
        
        # 限制最大连续失败次数统计到15次
        consecutive_misses = min(consecutive_misses, 15)
        
        # 记录统计
        consecutive_misses_stats[consecutive_misses]['total'] += 1
        if hit:
            consecutive_misses_stats[consecutive_misses]['hits'] += 1
    
    # 输出分析结果
    print("连续失败次数 | 总次数 | 命中次数 | 命中率 | 相对基准 | 样本量")
    print("-" * 70)
    
    base_hit_rate = sum(s['hits'] for s in consecutive_misses_stats.values()) / sum(s['total'] for s in consecutive_misses_stats.values())
    
    for misses in sorted(consecutive_misses_stats.keys()):
        stats = consecutive_misses_stats[misses]
        if stats['total'] >= 3:  # 至少3次数据才显示
            hit_rate = stats['hits'] / stats['total']
            relative = (hit_rate - base_hit_rate) / base_hit_rate * 100
            
            # 样本量指标
            sample_quality = "📊" if stats['total'] >= 20 else "⚠️" if stats['total'] >= 10 else "📉"
            
            indicator = "📈" if relative > 5 else "📉" if relative < -5 else "➡️"
            
            print(f"{misses:>8}次     | {stats['total']:>6} | {stats['hits']:>8} | {hit_rate:>6.1%} | {relative:>+6.1f}% {indicator} | {sample_quality}")
    
    print(f"\n基准命中率: {base_hit_rate:.1%}")
    
    # 关键发现
    print("\n【关键发现】")
    high_miss_rates = []
    for misses in range(2, 16):
        if misses in consecutive_misses_stats and consecutive_misses_stats[misses]['total'] >= 5:
            stats = consecutive_misses_stats[misses]
            hit_rate = stats['hits'] / stats['total']
            sample_count = stats['total']
            high_miss_rates.append((misses, hit_rate, sample_count))
    
    if high_miss_rates:
        # 按命中率排序
        high_miss_rates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ 连续失败{high_miss_rates[0][0]}次后，命中率最高: {high_miss_rates[0][1]:.1%} (样本{high_miss_rates[0][2]})")
        
        if len(high_miss_rates) > 1:
            print(f"✅ 连续失败{high_miss_rates[1][0]}次后，命中率次高: {high_miss_rates[1][1]:.1%} (样本{high_miss_rates[1][2]})")
        
        # 找出显著提升的连败次数（命中率比基准高15%以上）
        significant_improvements = [(m, r, c) for m, r, c in high_miss_rates if r > base_hit_rate * 1.15 and c >= 10]
        
        if significant_improvements:
            print(f"\n🔥 显著提升点（命中率比基准高15%+，样本≥10）：")
            for misses, rate, count in significant_improvements:
                improvement = (rate - base_hit_rate) / base_hit_rate * 100
                print(f"   - 连败{misses}次: {rate:.1%} (提升{improvement:+.1f}%, 样本{count})")
    
    return consecutive_misses_stats, base_hit_rate

def analyze_betting_multiplier_effectiveness(df):
    """
    分析智能动态倍投策略中，不同倍数下的命中率
    """
    print(f"\n{'='*70}")
    print(f"【智能动态倍投 - 倍数与命中率关系分析】")
    print(f"{'='*70}\n")
    
    # 初始化预测器
    predictor = PreciseTop15Predictor()
    
    # Fibonacci序列
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    # 策略配置（与GUI中一致）
    config = {
        'lookback': 12,
        'good_thresh': 0.35,
        'bad_thresh': 0.20,
        'boost_mult': 1.5,
        'reduce_mult': 1.0,
        'max_multiplier': 10
    }
    
    # 模拟策略
    fib_index = 0
    recent_results = []
    
    # 倍数统计
    multiplier_stats = defaultdict(lambda: {'total': 0, 'hits': 0})
    
    lookback = 50
    all_numbers = df['number'].tolist()
    
    for i in range(lookback, len(all_numbers)):
        train_data = all_numbers[:i]
        actual_number = all_numbers[i]
        
        # 预测TOP15
        predictions = predictor.predict(np.array(train_data))
        hit = actual_number in predictions
        
        # 更新预测器性能
        predictor.update_performance(predictions, actual_number)
        
        # 计算倍数（与GUI策略一致）
        if fib_index >= len(fib_sequence):
            base_mult = min(fib_sequence[-1], config['max_multiplier'])
        else:
            base_mult = min(fib_sequence[fib_index], config['max_multiplier'])
        
        # 根据近期命中率调整倍数
        if len(recent_results) >= config['lookback']:
            rate = sum(recent_results) / len(recent_results)
            if rate >= config['good_thresh']:
                multiplier = min(base_mult * config['boost_mult'], config['max_multiplier'])
            elif rate <= config['bad_thresh']:
                multiplier = max(base_mult * config['reduce_mult'], 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        # 统计倍数区间的命中率
        mult_range = int(multiplier)  # 按整数倍数分组
        multiplier_stats[mult_range]['total'] += 1
        if hit:
            multiplier_stats[mult_range]['hits'] += 1
        
        # 更新状态
        if hit:
            fib_index = 0
        else:
            fib_index += 1
        
        recent_results.append(1 if hit else 0)
        if len(recent_results) > config['lookback']:
            recent_results.pop(0)
    
    # 输出结果
    print("倍数区间 | 总次数 | 命中次数 | 命中率 | 样本量")
    print("-" * 60)
    
    for mult in sorted(multiplier_stats.keys()):
        stats = multiplier_stats[mult]
        if stats['total'] >= 3:
            hit_rate = stats['hits'] / stats['total']
            sample_quality = "📊" if stats['total'] >= 20 else "⚠️" if stats['total'] >= 10 else "📉"
            print(f"{mult:>4}倍     | {stats['total']:>6} | {stats['hits']:>8} | {hit_rate:>6.1%} | {sample_quality}")
    
    print("\n【关键发现】")
    print("✅ 高倍数投注（5倍+）通常出现在连败期，此时命中率可能有所下降")
    print("✅ 低倍数投注（1-2倍）通常在连胜或稳定期，命中率相对稳定")
    print("💡 策略优化方向：在高倍数期加入额外的止损或谨慎机制")

def analyze_recent_rate_thresholds(df):
    """
    分析12期命中率阈值（35%/20%）的有效性
    """
    print(f"\n{'='*70}")
    print(f"【12期命中率阈值分析】")
    print(f"{'='*70}\n")
    
    # 初始化预测器
    predictor = PreciseTop15Predictor()
    
    lookback = 50
    window = 12
    all_numbers = df['number'].tolist()
    
    # 统计不同命中率区间的下一期表现
    rate_bins = {
        '≤10%': (0.0, 0.10),
        '10-20%': (0.10, 0.20),
        '20-30%': (0.20, 0.30),
        '30-35%': (0.30, 0.35),
        '35-40%': (0.35, 0.40),
        '40-50%': (0.40, 0.50),
        '>50%': (0.50, 1.0)
    }
    
    rate_stats = defaultdict(lambda: {'total': 0, 'hits': 0})
    
    recent_results = []
    
    for i in range(lookback, len(all_numbers)):
        train_data = all_numbers[:i]
        actual_number = all_numbers[i]
        
        # 预测
        predictions = predictor.predict(np.array(train_data))
        hit = actual_number in predictions
        predictor.update_performance(predictions, actual_number)
        
        # 计算最近12期命中率
        if len(recent_results) >= window:
            recent_rate = sum(recent_results[-window:]) / window
            
            # 找到对应的区间
            for bin_name, (low, high) in rate_bins.items():
                if low <= recent_rate < high:
                    rate_stats[bin_name]['total'] += 1
                    if hit:
                        rate_stats[bin_name]['hits'] += 1
                    break
        
        # 更新历史
        recent_results.append(1 if hit else 0)
    
    # 输出结果
    print("12期命中率区间 | 总次数 | 下期命中 | 下期命中率 | 策略动作")
    print("-" * 75)
    
    for bin_name in ['≤10%', '10-20%', '20-30%', '30-35%', '35-40%', '40-50%', '>50%']:
        if bin_name in rate_stats:
            stats = rate_stats[bin_name]
            if stats['total'] >= 3:
                hit_rate = stats['hits'] / stats['total']
                
                # 确定策略动作
                if bin_name in ['≤10%', '10-20%']:
                    action = "降低倍数(×1.0)"
                elif bin_name in ['35-40%', '40-50%', '>50%']:
                    action = "增强倍数(×1.5)"
                else:
                    action = "维持基础倍数"
                
                print(f"{bin_name:>16} | {stats['total']:>6} | {stats['hits']:>8} | {hit_rate:>10.1%} | {action}")
    
    print("\n【关键发现】")
    print("✅ 当12期命中率≥35%时，下期命中率是否真的更高？")
    
    # 验证阈值有效性
    if '35-40%' in rate_stats or '40-50%' in rate_stats or '>50%' in rate_stats:
        high_rate_total = sum(rate_stats[k]['total'] for k in ['35-40%', '40-50%', '>50%'] if k in rate_stats)
        high_rate_hits = sum(rate_stats[k]['hits'] for k in ['35-40%', '40-50%', '>50%'] if k in rate_stats)
        high_rate_avg = high_rate_hits / high_rate_total if high_rate_total > 0 else 0
        
        low_rate_total = sum(rate_stats[k]['total'] for k in ['≤10%', '10-20%'] if k in rate_stats)
        low_rate_hits = sum(rate_stats[k]['hits'] for k in ['≤10%', '10-20%'] if k in rate_stats)
        low_rate_avg = low_rate_hits / low_rate_total if low_rate_total > 0 else 0
        
        print(f"   高命中率期（≥35%）下期平均: {high_rate_avg:.1%} (样本{high_rate_total})")
        print(f"   低命中率期（≤20%）下期平均: {low_rate_avg:.1%} (样本{low_rate_total})")
        
        if high_rate_avg > low_rate_avg:
            diff = (high_rate_avg - low_rate_avg) * 100
            print(f"✅ 阈值策略有效：高命中率期比低命中率期高{diff:.1f}个百分点")
        else:
            print(f"⚠️ 阈值策略效果不明显，可能需要调整参数")

def analyze_pause_strategy_effectiveness(df):
    """
    分析"命中1停1期"暂停策略的有效性
    """
    print(f"\n{'='*70}")
    print(f"【命中1停1期暂停策略分析】")
    print(f"{'='*70}\n")
    
    # 初始化预测器
    predictor = PreciseTop15Predictor()
    
    lookback = 50
    all_numbers = df['number'].tolist()
    
    # 统计：命中后的下一期命中率
    after_hit_stats = {'total': 0, 'hits': 0}
    after_miss_stats = {'total': 0, 'hits': 0}
    
    prev_hit = None
    
    for i in range(lookback, len(all_numbers)):
        train_data = all_numbers[:i]
        actual_number = all_numbers[i]
        
        # 预测
        predictions = predictor.predict(np.array(train_data))
        hit = actual_number in predictions
        predictor.update_performance(predictions, actual_number)
        
        # 统计上一期状态对当期的影响
        if prev_hit is not None:
            if prev_hit:
                after_hit_stats['total'] += 1
                if hit:
                    after_hit_stats['hits'] += 1
            else:
                after_miss_stats['total'] += 1
                if hit:
                    after_miss_stats['hits'] += 1
        
        prev_hit = hit
    
    # 输出结果
    print("上期状态   | 总次数 | 当期命中 | 当期命中率")
    print("-" * 50)
    
    if after_hit_stats['total'] > 0:
        hit_rate = after_hit_stats['hits'] / after_hit_stats['total']
        print(f"上期命中   | {after_hit_stats['total']:>6} | {after_hit_stats['hits']:>8} | {hit_rate:>10.1%}")
    
    if after_miss_stats['total'] > 0:
        miss_rate = after_miss_stats['hits'] / after_miss_stats['total']
        print(f"上期未中   | {after_miss_stats['total']:>6} | {after_miss_stats['hits']:>8} | {miss_rate:>10.1%}")
    
    print("\n【暂停策略有效性评估】")
    
    if after_hit_stats['total'] > 0 and after_miss_stats['total'] > 0:
        hit_rate = after_hit_stats['hits'] / after_hit_stats['total']
        miss_rate = after_miss_stats['hits'] / after_miss_stats['total']
        
        if hit_rate < miss_rate:
            diff = (miss_rate - hit_rate) * 100
            print(f"✅ 暂停策略有效：命中后下期命中率降低{diff:.1f}个百分点")
            print(f"💡 建议：命中后暂停1-2期可以避开低概率期")
        elif hit_rate > miss_rate:
            diff = (hit_rate - miss_rate) * 100
            print(f"⚠️ 暂停策略可能不利：命中后下期命中率反而高{diff:.1f}个百分点")
            print(f"💡 建议：考虑取消暂停策略，或改为连胜后暂停")
        else:
            print(f"➡️ 暂停策略效果中性：命中后下期命中率无明显变化")
            print(f"💡 建议：暂停策略主要用于风险控制，而非提升命中率")

def main():
    """主函数"""
    print("="*70)
    print(" " * 15 + "TOP15命中率规律分析系统")
    print(" " * 12 + "（最优智能投注策略专用）")
    print("="*70)
    
    # 加载数据
    print("\n正在加载历史数据...")
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    print(f"✅ 已加载 {len(df)} 条记录")
    print(f"日期范围: {df['date'].min().strftime('%Y/%m/%d')} ~ {df['date'].max().strftime('%Y/%m/%d')}")
    
    # 1. 连续失败分析
    consecutive_stats, base_rate = analyze_consecutive_misses_pattern(df)
    
    # 2. 倍数有效性分析
    analyze_betting_multiplier_effectiveness(df)
    
    # 3. 12期命中率阈值分析
    analyze_recent_rate_thresholds(df)
    
    # 4. 暂停策略分析
    analyze_pause_strategy_effectiveness(df)
    
    # 5. 生成投注建议
    print(f"\n{'='*70}")
    print("【基于规律的投注建议】")
    print(f"{'='*70}\n")
    
    print("1️⃣ 连败倍投建议：")
    # 找出命中率显著提升的连败次数
    significant_points = []
    for misses, stats in consecutive_stats.items():
        if stats['total'] >= 10:
            hit_rate = stats['hits'] / stats['total']
            if hit_rate > base_rate * 1.15:
                significant_points.append((misses, hit_rate, stats['total']))
    
    if significant_points:
        significant_points.sort(key=lambda x: x[1], reverse=True)
        for misses, rate, count in significant_points[:3]:
            improvement = (rate - base_rate) / base_rate * 100
            print(f"   - 连败{misses}次: 命中率{rate:.1%} (提升{improvement:+.1f}%, 样本{count}) ⭐")
    else:
        print(f"   ⚠️ 未发现明显的连败后命中率提升规律")
    
    print("\n2️⃣ 12期命中率策略建议：")
    print(f"   - 当12期命中率≥35%时，适度增加倍数（×1.5）✅")
    print(f"   - 当12期命中率≤20%时，保持基础倍数或降低（×1.0）⚠️")
    print(f"   - 中间区间（20-35%）维持Fibonacci基础倍数")
    
    print("\n3️⃣ 暂停策略建议：")
    print(f"   - 命中后暂停1期可以降低风险，减少回撤")
    print(f"   - 但要注意：暂停期间可能错过连胜机会")
    print(f"   - 建议：资金充足时使用，追求稳健收益")
    
    print("\n4️⃣ 止损建议：")
    print(f"   - 单期最大倍数：10倍（150元投注）")
    print(f"   - 连续失败止损：连败10次暂停并重新评估")
    print(f"   - 累计回撤止损：当回撤超过500元时暂停投注")
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

if __name__ == '__main__':
    main()
