"""
重新训练生肖预测器 - 基于最新数据优化
分析最近200期数据，调整预测策略
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def analyze_recent_data(csv_file='data/lucky_numbers.csv', window=200):
    """分析最近N期数据的生肖分布特征"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    # 获取最近200期
    recent_df = df.tail(window)
    animals = recent_df['animal'].values
    
    print(f"{'='*80}")
    print(f"分析最近{window}期数据（第{len(df)-window+1}-{len(df)}期）")
    print(f"日期范围: {recent_df.iloc[0]['date']} 至 {recent_df.iloc[-1]['date']}")
    print(f"{'='*80}\n")
    
    # 1. 整体频率分析
    freq = Counter(animals)
    print("【生肖频率统计】")
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    for zodiac, count in sorted_freq:
        percentage = count / window * 100
        bar = '█' * int(percentage * 2)
        print(f"{zodiac:>2}: {count:>3}次 ({percentage:>5.1f}%) {bar}")
    
    # 2. 分析热门生肖（出现≥20次）
    hot_threshold = window * 0.10  # 10%以上
    hot_zodiacs = [z for z, c in sorted_freq if c >= hot_threshold]
    cold_zodiacs = [z for z, c in sorted_freq if c <= window * 0.06]  # 6%以下
    
    print(f"\n【热门生肖】(≥{hot_threshold:.0f}次): {', '.join(hot_zodiacs)}")
    print(f"【冷门生肖】(≤{window*0.06:.0f}次): {', '.join(cold_zodiacs)}")
    
    # 3. 分析间隔模式
    print("\n【间隔分析】")
    gaps = defaultdict(list)
    last_seen = {}
    
    for i, zodiac in enumerate(animals):
        if zodiac in last_seen:
            gap = i - last_seen[zodiac]
            gaps[zodiac].append(gap)
        last_seen[zodiac] = i
    
    gap_stats = {}
    for zodiac in sorted(gaps.keys()):
        if gaps[zodiac]:
            avg_gap = np.mean(gaps[zodiac])
            min_gap = min(gaps[zodiac])
            max_gap = max(gaps[zodiac])
            gap_stats[zodiac] = {'avg': avg_gap, 'min': min_gap, 'max': max_gap}
            print(f"  {zodiac}: 平均间隔{avg_gap:.1f}期 (最小{min_gap}, 最大{max_gap})")
    
    # 4. 分析连续出现模式
    print("\n【连续出现分析】")
    consecutive_count = 0
    prev_zodiac = None
    consecutive_cases = []
    
    for zodiac in animals:
        if zodiac == prev_zodiac:
            consecutive_count += 1
        else:
            if consecutive_count > 0:
                consecutive_cases.append((prev_zodiac, consecutive_count + 1))
            consecutive_count = 0
        prev_zodiac = zodiac
    
    if consecutive_count > 0:
        consecutive_cases.append((prev_zodiac, consecutive_count + 1))
    
    consecutive_stats = Counter([z for z, _ in consecutive_cases])
    print(f"  连续出现的生肖: {dict(consecutive_stats)}")
    print(f"  最长连续: {max(consecutive_cases, key=lambda x: x[1]) if consecutive_cases else '无'}")
    
    # 5. 分析最近30期模式
    print("\n【最近30期特征】")
    recent_30 = animals[-30:]
    freq_30 = Counter(recent_30)
    sorted_30 = sorted(freq_30.items(), key=lambda x: x[1], reverse=True)
    print(f"  前5高频: {', '.join([f'{z}({c})' for z, c in sorted_30[:5]])}")
    
    cold_30 = [z for z in ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪'] if z not in freq_30]
    print(f"  未出现生肖: {', '.join(cold_30) if cold_30 else '全部出现'}")
    
    return {
        'freq': freq,
        'hot_zodiacs': hot_zodiacs,
        'cold_zodiacs': cold_zodiacs,
        'gap_stats': gap_stats,
        'recent_30_freq': freq_30,
        'recent_30_cold': cold_30
    }


def create_optimized_predictor_v2(analysis_result):
    """基于分析结果创建优化预测器 v2.0"""
    
    print(f"\n{'='*80}")
    print("【生成优化预测策略】")
    print(f"{'='*80}\n")
    
    hot_zodiacs = set(analysis_result['hot_zodiacs'])
    cold_zodiacs = set(analysis_result['cold_zodiacs'])
    recent_30_freq = analysis_result['recent_30_freq']
    recent_30_cold = set(analysis_result['recent_30_cold'])
    
    # 策略权重调整建议
    print("【策略权重优化建议】\n")
    
    # 检测当前模式
    hot_ratio = len(hot_zodiacs) / 12
    cold_ratio = len(recent_30_cold) / 12
    diversity = len(recent_30_freq) / 12
    
    print(f"热门生肖占比: {hot_ratio*100:.1f}% ({len(hot_zodiacs)}/12)")
    print(f"最近30期未出现生肖: {cold_ratio*100:.1f}% ({len(recent_30_cold)}/12)")
    print(f"最近30期多样性: {diversity*100:.1f}%\n")
    
    # 根据特征推荐策略
    if cold_ratio >= 0.25:  # 超过25%生肖未出现
        strategy_type = "极致冷门策略"
        weights = {
            'cold_boost': 0.45,      # 大幅提升冷门生肖权重
            'anti_hot': 0.25,        # 强力反热门
            'gap_analysis': 0.15,    # 间隔分析
            'rotation': 0.10,        # 轮转模式
            'diversity': 0.05        # 多样性
        }
    elif hot_ratio >= 0.25:  # 超过25%生肖为热门
        strategy_type = "热门感知策略"
        weights = {
            'cold_boost': 0.25,      # 适度冷门
            'anti_hot': 0.15,        # 适度反热门
            'gap_analysis': 0.25,    # 加强间隔
            'hot_momentum': 0.20,    # 热门惯性
            'rotation': 0.10,        # 轮转
            'diversity': 0.05        # 多样性
        }
    else:  # 均衡状态
        strategy_type = "平衡策略"
        weights = {
            'cold_boost': 0.35,      # 冷门优先
            'anti_hot': 0.20,        # 反热门
            'gap_analysis': 0.20,    # 间隔分析
            'rotation': 0.15,        # 轮转
            'diversity': 0.10        # 多样性
        }
    
    print(f"推荐策略类型: {strategy_type}\n")
    print("权重配置:")
    for name, weight in weights.items():
        bar = '█' * int(weight * 50)
        print(f"  {name:>15}: {weight:.2f} {bar}")
    
    return {
        'strategy_type': strategy_type,
        'weights': weights,
        'hot_zodiacs': list(hot_zodiacs),
        'cold_zodiacs': list(cold_zodiacs),
        'recent_30_cold': list(recent_30_cold)
    }


def save_optimized_config(config, filename='optimized_zodiac_config_v2.txt'):
    """保存优化配置"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("生肖预测器优化配置 v2.0\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"策略类型: {config['strategy_type']}\n\n")
        f.write("权重配置:\n")
        for name, weight in config['weights'].items():
            f.write(f"  {name}: {weight}\n")
        
        f.write(f"\n热门生肖: {', '.join(config['hot_zodiacs'])}\n")
        f.write(f"冷门生肖: {', '.join(config['cold_zodiacs'])}\n")
        f.write(f"最近30期未出现: {', '.join(config['recent_30_cold']) if config['recent_30_cold'] else '全部出现'}\n")
    
    print(f"\n✅ 配置已保存: {filename}")


if __name__ == '__main__':
    print("="*80)
    print("生肖预测器重新训练程序 - 基于最新数据分析")
    print("="*80 + "\n")
    
    # 步骤1: 分析最近200期数据
    analysis = analyze_recent_data(window=200)
    
    # 步骤2: 生成优化配置
    optimized_config = create_optimized_predictor_v2(analysis)
    
    # 步骤3: 保存配置
    save_optimized_config(optimized_config)
    
    print("\n" + "="*80)
    print("下一步: 使用优化配置创建新预测器并验证")
    print("="*80)
