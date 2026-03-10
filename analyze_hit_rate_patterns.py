"""
命中率规律分析
分析历史数据中的命中率规律，特别关注：
1. 连续失败后的命中率
2. 时间周期性规律
3. 热度规律
4. 命中率趋势
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

def load_data(filepath='data/lucky_numbers.csv'):
    """加载历史数据"""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    return df

def analyze_consecutive_misses_pattern(df, predictor_type='zodiac'):
    """
    分析连续失败后的命中率规律
    
    参数:
        predictor_type: 'zodiac' 或 'number'
    """
    print(f"\n{'='*60}")
    print(f"【连续失败后命中率分析 - {predictor_type.upper()}】")
    print(f"{'='*60}\n")
    
    # 模拟预测过程（使用简单的历史高频预测）
    consecutive_misses_stats = defaultdict(lambda: {'total': 0, 'hits': 0})
    
    if predictor_type == 'zodiac':
        # 生肖预测：使用近期高频生肖
        lookback = 20
        for i in range(lookback, len(df)):
            recent_data = df.iloc[i-lookback:i]
            actual_animal = df.iloc[i]['animal']
            
            # 获取高频生肖作为预测
            animal_counts = recent_data['animal'].value_counts()
            top_animals = set(animal_counts.head(5).index)
            
            # 判断是否命中
            hit = actual_animal in top_animals
            
            # 计算连续失败次数
            consecutive_misses = 0
            for j in range(i-1, max(i-10, lookback-1), -1):
                prev_actual = df.iloc[j]['animal']
                prev_recent = df.iloc[j-lookback:j]
                prev_top = set(prev_recent['animal'].value_counts().head(5).index)
                if prev_actual not in prev_top:
                    consecutive_misses += 1
                else:
                    break
            
            # 限制最大连续失败次数统计到10次
            consecutive_misses = min(consecutive_misses, 10)
            
            # 记录统计
            consecutive_misses_stats[consecutive_misses]['total'] += 1
            if hit:
                consecutive_misses_stats[consecutive_misses]['hits'] += 1
    
    else:  # number prediction
        # 号码预测：使用近期高频号码
        lookback = 30
        for i in range(lookback, len(df)):
            recent_data = df.iloc[i-lookback:i]
            actual_number = df.iloc[i]['number']
            
            # 获取高频号码作为预测
            number_counts = recent_data['number'].value_counts()
            top_numbers = set(number_counts.head(15).index)
            
            # 判断是否命中
            hit = actual_number in top_numbers
            
            # 计算连续失败次数
            consecutive_misses = 0
            for j in range(i-1, max(i-10, lookback-1), -1):
                prev_actual = df.iloc[j]['number']
                prev_recent = df.iloc[j-lookback:j]
                prev_top = set(prev_recent['number'].value_counts().head(15).index)
                if prev_actual not in prev_top:
                    consecutive_misses += 1
                else:
                    break
            
            consecutive_misses = min(consecutive_misses, 10)
            
            consecutive_misses_stats[consecutive_misses]['total'] += 1
            if hit:
                consecutive_misses_stats[consecutive_misses]['hits'] += 1
    
    # 输出分析结果
    print("连续失败次数 | 总次数 | 命中次数 | 命中率 | 相对基准")
    print("-" * 60)
    
    base_hit_rate = sum(s['hits'] for s in consecutive_misses_stats.values()) / sum(s['total'] for s in consecutive_misses_stats.values())
    
    for misses in sorted(consecutive_misses_stats.keys()):
        stats = consecutive_misses_stats[misses]
        if stats['total'] >= 5:  # 至少5次数据才有统计意义
            hit_rate = stats['hits'] / stats['total']
            relative = (hit_rate - base_hit_rate) / base_hit_rate * 100
            indicator = "📈" if relative > 5 else "📉" if relative < -5 else "➡️"
            print(f"{misses:>8}次     | {stats['total']:>6} | {stats['hits']:>8} | {hit_rate:>6.1%} | {relative:>+6.1f}% {indicator}")
    
    print(f"\n基准命中率: {base_hit_rate:.1%}")
    
    # 关键发现
    print("\n【关键发现】")
    high_miss_rates = []
    for misses in range(3, 11):
        if misses in consecutive_misses_stats and consecutive_misses_stats[misses]['total'] >= 5:
            stats = consecutive_misses_stats[misses]
            hit_rate = stats['hits'] / stats['total']
            high_miss_rates.append((misses, hit_rate))
    
    if high_miss_rates:
        high_miss_rates.sort(key=lambda x: x[1], reverse=True)
        print(f"✅ 连续失败{high_miss_rates[0][0]}次后，命中率最高: {high_miss_rates[0][1]:.1%}")
        if len(high_miss_rates) > 1:
            print(f"✅ 连续失败{high_miss_rates[1][0]}次后，命中率次高: {high_miss_rates[1][1]:.1%}")
    
    return consecutive_misses_stats, base_hit_rate

def analyze_time_patterns(df):
    """分析时间周期性规律"""
    print(f"\n{'='*60}")
    print("【时间周期性规律分析】")
    print(f"{'='*60}\n")
    
    df['weekday'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # 使用近期高频生肖作为预测基准
    lookback = 20
    df['predicted_hit'] = False
    
    for i in range(lookback, len(df)):
        recent_data = df.iloc[i-lookback:i]
        actual_animal = df.iloc[i]['animal']
        top_animals = set(recent_data['animal'].value_counts().head(5).index)
        df.loc[df.index[i], 'predicted_hit'] = actual_animal in top_animals
    
    # 分析星期几的规律
    print("【星期规律】")
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    weekday_stats = df[df.index >= lookback].groupby('weekday')['predicted_hit'].agg(['sum', 'count', 'mean'])
    
    print("星期   | 总次数 | 命中次数 | 命中率")
    print("-" * 45)
    for idx, row in weekday_stats.iterrows():
        if row['count'] >= 5:
            print(f"{weekday_names[idx]:>6} | {int(row['count']):>6} | {int(row['sum']):>8} | {row['mean']:>6.1%}")
    
    # 分析月份规律
    print("\n【月份规律】")
    month_stats = df[df.index >= lookback].groupby('month')['predicted_hit'].agg(['sum', 'count', 'mean'])
    
    print("月份 | 总次数 | 命中次数 | 命中率")
    print("-" * 40)
    for idx, row in month_stats.iterrows():
        if row['count'] >= 10:
            print(f"{int(idx):>4} | {int(row['count']):>6} | {int(row['sum']):>8} | {row['mean']:>6.1%}")

def analyze_hot_cold_patterns(df):
    """分析数字/生肖热度规律"""
    print(f"\n{'='*60}")
    print("【热度规律分析】")
    print(f"{'='*60}\n")
    
    # 分析最近100期的数据
    recent_df = df.tail(100)
    
    # 生肖热度
    print("【生肖热度 TOP5（最近100期）】")
    animal_counts = recent_df['animal'].value_counts().head(10)
    for animal, count in animal_counts.items():
        percentage = count / len(recent_df) * 100
        heat = "🔥" if percentage > 10 else "🌡️" if percentage > 7 else "❄️"
        print(f"{animal}: {count}次 ({percentage:.1f}%) {heat}")
    
    # 号码热度
    print("\n【号码热度 TOP15（最近100期）】")
    number_counts = recent_df['number'].value_counts().head(15)
    for number, count in number_counts.items():
        percentage = count / len(recent_df) * 100
        heat = "🔥" if percentage > 3 else "🌡️" if percentage > 2 else "❄️"
        print(f"{number:>2}: {count}次 ({percentage:.1f}%) {heat}")
    
    # 冷号分析（最近100期从未出现）
    all_numbers = set(range(1, 50))
    appeared_numbers = set(recent_df['number'].unique())
    cold_numbers = all_numbers - appeared_numbers
    
    if cold_numbers:
        print(f"\n【冷号（最近100期未出现）】")
        print(f"冷号数量: {len(cold_numbers)}个")
        print(f"冷号列表: {sorted(cold_numbers)}")

def analyze_streak_patterns(df):
    """分析连胜连败规律"""
    print(f"\n{'='*60}")
    print("【连胜连败规律分析】")
    print(f"{'='*60}\n")
    
    # 模拟预测
    lookback = 20
    streak_stats = {'current_streak': 0, 'max_win_streak': 0, 'max_lose_streak': 0, 
                    'win_streaks': [], 'lose_streaks': []}
    
    current_streak = 0
    for i in range(lookback, len(df)):
        recent_data = df.iloc[i-lookback:i]
        actual_animal = df.iloc[i]['animal']
        top_animals = set(recent_data['animal'].value_counts().head(5).index)
        hit = actual_animal in top_animals
        
        if hit:
            if current_streak >= 0:
                current_streak += 1
            else:
                streak_stats['lose_streaks'].append(abs(current_streak))
                current_streak = 1
        else:
            if current_streak <= 0:
                current_streak -= 1
            else:
                streak_stats['win_streaks'].append(current_streak)
                current_streak = -1
        
        if current_streak > 0:
            streak_stats['max_win_streak'] = max(streak_stats['max_win_streak'], current_streak)
        else:
            streak_stats['max_lose_streak'] = max(streak_stats['max_lose_streak'], abs(current_streak))
    
    print(f"最长连胜: {streak_stats['max_win_streak']}期 🎉")
    print(f"最长连败: {streak_stats['max_lose_streak']}期 😰")
    
    if streak_stats['win_streaks']:
        avg_win_streak = np.mean(streak_stats['win_streaks'])
        print(f"平均连胜长度: {avg_win_streak:.1f}期")
    
    if streak_stats['lose_streaks']:
        avg_lose_streak = np.mean(streak_stats['lose_streaks'])
        print(f"平均连败长度: {avg_lose_streak:.1f}期")

def generate_prediction_suggestions(consecutive_stats, base_rate):
    """生成预测建议"""
    print(f"\n{'='*60}")
    print("【投注建议生成】")
    print(f"{'='*60}\n")
    
    # 找出命中率明显高于基准的连败次数
    high_confidence_thresholds = []
    for misses in sorted(consecutive_stats.keys()):
        stats = consecutive_stats[misses]
        if stats['total'] >= 10:
            hit_rate = stats['hits'] / stats['total']
            if hit_rate > base_rate * 1.1:  # 高于基准10%
                high_confidence_thresholds.append((misses, hit_rate))
    
    print("【基于连败次数的投注策略】")
    if high_confidence_thresholds:
        for misses, hit_rate in high_confidence_thresholds:
            multiplier = 1.0 + (hit_rate - base_rate) * 2  # 根据超出基准的幅度计算倍数
            print(f"✅ 连续失败{misses}次后:")
            print(f"   - 命中率: {hit_rate:.1%} (基准: {base_rate:.1%})")
            print(f"   - 建议倍数: {multiplier:.1f}x")
            print(f"   - 信心等级: {'⭐⭐⭐' if hit_rate > base_rate * 1.2 else '⭐⭐' if hit_rate > base_rate * 1.15 else '⭐'}")
            print()
    else:
        print("⚠️ 未发现明显的连败后命中率提升规律")
        print("建议：使用固定倍投或基于概率预测的动态策略")
    
    print("\n【通用建议】")
    print("1. 赌徒谬误：独立事件的概率不受历史影响")
    print("2. 但历史数据可以帮助识别：")
    print("   - 模型的预测能力变化（环境变化）")
    print("   - 热度周期（某些生肖/号码的周期性）")
    print("   - 预测模型的置信度校准")
    print("3. 建议策略：")
    print("   - 结合概率预测（已实现）")
    print("   - 动态调整倍投（基于近期表现）")
    print("   - 设置止损点（避免过度追损）")

def main():
    """主函数"""
    print("="*60)
    print(" " * 15 + "命中率规律分析系统")
    print("="*60)
    
    # 加载数据
    print("\n正在加载历史数据...")
    df = load_data()
    print(f"✅ 已加载 {len(df)} 条记录")
    print(f"日期范围: {df['date'].min().strftime('%Y/%m/%d')} ~ {df['date'].max().strftime('%Y/%m/%d')}")
    
    # 1. 连续失败分析（生肖）
    zodiac_stats, zodiac_base = analyze_consecutive_misses_pattern(df, 'zodiac')
    
    # 2. 连续失败分析（号码）
    number_stats, number_base = analyze_consecutive_misses_pattern(df, 'number')
    
    # 3. 时间周期分析
    analyze_time_patterns(df)
    
    # 4. 热度分析
    analyze_hot_cold_patterns(df)
    
    # 5. 连胜连败分析
    analyze_streak_patterns(df)
    
    # 6. 生成建议
    print("\n" + "="*60)
    print("【生肖TOP5预测建议】")
    print("="*60)
    generate_prediction_suggestions(zodiac_stats, zodiac_base)
    
    print("\n" + "="*60)
    print("【号码TOP15预测建议】")
    print("="*60)
    generate_prediction_suggestions(number_stats, number_base)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)

if __name__ == '__main__':
    main()
