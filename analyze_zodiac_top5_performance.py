"""
生肖TOP5策略分析报告
分析为什么命中率只有38.33%，以及如何优化
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_zodiac_top5_performance():
    """分析生肖TOP5预测器性能"""
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    animals = df['animal'].tolist()
    
    zodiac_list = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
    
    print(f"\n{'='*80}")
    print(f"生肖TOP5预测策略性能分析")
    print(f"{'='*80}\n")
    
    # 1. 生肖分布分析
    print(f"【1. 生肖出现频率分析】")
    total = len(animals)
    animal_freq = Counter(animals)
    
    print(f"总期数: {total}")
    print(f"\n各生肖出现次数和频率:")
    sorted_freq = sorted(animal_freq.items(), key=lambda x: x[1], reverse=True)
    for animal, count in sorted_freq:
        freq = count / total * 100
        print(f"  {animal}: {count}次 ({freq:.2f}%)")
    
    # 理论上均匀分布应该是8.33%，实际分布是否均匀？
    expected_freq = total / 12
    print(f"\n理论均匀分布: 每个生肖 {expected_freq:.2f}次 ({1/12*100:.2f}%)")
    
    # 2. 分段分析
    print(f"\n\n{'='*80}")
    print(f"【2. 分段命中率分析】")
    print(f"{'='*80}\n")
    
    # 分成5段，每段60期
    segment_size = 60
    total_periods = len(animals)
    
    for seg in range(5):
        start = seg * segment_size
        end = min(start + segment_size, total_periods)
        
        if start >= total_periods:
            break
        
        segment_animals = animals[start:end]
        freq = Counter(segment_animals)
        top5 = [z for z, _ in freq.most_common(5)]
        
        print(f"第{seg+1}段 (第{start+1}期-第{end}期):")
        print(f"  TOP5生肖: {', '.join(top5)}")
        print(f"  频率分布: {dict(freq.most_common(5))}")
        
    # 3. 滚动窗口分析
    print(f"\n\n{'='*80}")
    print(f"【3. 滚动窗口预测准确率分析】")
    print(f"{'='*80}\n")
    
    # 测试不同的窗口大小
    window_sizes = [10, 20, 30, 50, 100]
    test_periods = 300
    start_idx = total_periods - test_periods
    
    results = []
    
    for window in window_sizes:
        hits = 0
        
        for i in range(start_idx, total_periods):
            if i < window:
                continue
            
            # 使用最近window期的数据预测
            history = animals[max(0, i-window):i]
            actual = animals[i]
            
            # 统计TOP5
            freq = Counter(history)
            top5 = [z for z, _ in freq.most_common(5)]
            
            # 如果不足5个，补充
            if len(top5) < 5:
                remaining = [z for z in zodiac_list if z not in top5]
                top5.extend(remaining[:5 - len(top5)])
            
            if actual in top5:
                hits += 1
        
        tested = total_periods - max(start_idx, window)
        hit_rate = hits / tested * 100 if tested > 0 else 0
        
        results.append({
            'window': window,
            'hits': hits,
            'tested': tested,
            'hit_rate': hit_rate
        })
        
        print(f"窗口大小 {window}期:")
        print(f"  测试期数: {tested}")
        print(f"  命中次数: {hits}")
        print(f"  命中率: {hit_rate:.2f}%")
        print(f"  理论盈亏平衡点: 42.55%")
        print(f"  是否可盈利: {'✅ 是' if hit_rate >= 42.55 else '❌ 否'}\n")
    
    # 4. 最优策略建议
    print(f"\n{'='*80}")
    print(f"【4. 策略优化建议】")
    print(f"{'='*80}\n")
    
    best_result = max(results, key=lambda x: x['hit_rate'])
    print(f"最优窗口大小: {best_result['window']}期")
    print(f"最高命中率: {best_result['hit_rate']:.2f}%")
    
    if best_result['hit_rate'] < 42.55:
        shortfall = 42.55 - best_result['hit_rate']
        print(f"\n⚠️  警告：即使使用最优窗口，命中率仍低于盈亏平衡点")
        print(f"      差距: {shortfall:.2f}%")
        print(f"\n💡 建议：")
        print(f"   1. 放弃TOP5策略，改用TOP6或TOP7以提高命中率")
        print(f"   2. 优化预测算法，引入更多特征（如五行、号码规律等）")
        print(f"   3. 考虑组合多个预测器进行ensemble")
        print(f"   4. 如果坚持使用，建议采用严格止损策略")
    else:
        print(f"\n✅ 建议使用窗口大小: {best_result['window']}期")
    
    # 5. ROI计算
    print(f"\n\n{'='*80}")
    print(f"【5. 不同命中率下的ROI预估】")
    print(f"{'='*80}\n")
    
    print(f"基础投注: 20元/期, 中奖奖励: 47元/次")
    print(f"\n命中率 → 期望收益 → ROI:")
    for rate in [30, 35, 38.33, 40, 42.55, 45, 50]:
        expected = rate/100 * (47-20) - (1-rate/100) * 20
        roi = expected / 20 * 100
        status = "✅ 盈利" if roi > 0 else "❌ 亏损" if roi < 0 else "⚖️  保本"
        print(f"  {rate:.2f}% → {expected:+.2f}元/期 → ROI {roi:+.2f}% {status}")


if __name__ == "__main__":
    analyze_zodiac_top5_performance()
