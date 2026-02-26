"""
分析最后20期（369-388）为何两个模型都失效
"""

import pandas as pd
from collections import Counter


def analyze_last20_failure():
    """分析最后20期的特殊性"""
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    print("="*80)
    print("最后20期（369-388期）失效原因分析")
    print("="*80 + "\n")
    
    # 获取最后20期
    last20 = df.tail(20)
    animals_last20 = last20['animal'].values
    
    # 获取前面20期（349-368）作为对比
    prev20 = df.iloc[-40:-20]
    animals_prev20 = prev20['animal'].values
    
    # 获取更早的参考期（289-308）
    early20 = df.iloc[-100:-80]
    animals_early20 = early20['animal'].values
    
    print("【生肖分布对比】\n")
    
    freq_last20 = Counter(animals_last20)
    freq_prev20 = Counter(animals_prev20)
    freq_early20 = Counter(animals_early20)
    
    print(f"{'生肖':>4} {'早期20期':>12} {'前20期':>10} {'最后20期':>10} {'趋势'}")
    print("-"*55)
    
    zodiacs = ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪']
    
    for zodiac in zodiacs:
        early = freq_early20.get(zodiac, 0)
        prev = freq_prev20.get(zodiac, 0)
        last = freq_last20.get(zodiac, 0)
        
        # 分析趋势
        if last == 0:
            trend = "❌消失"
        elif last >= 4:
            trend = "🔥爆发"
        elif last > prev:
            trend = "↑上升"
        elif last < prev:
            trend = "↓下降"
        else:
            trend = "→持平"
        
        print(f"{zodiac:>4} {early:>5}次 ({early*5}%) {prev:>4}次 ({prev*5}%) {last:>5}次 ({last*5}%) {trend}")
    
    print("\n【关键发现】\n")
    
    # 找出消失的生肖
    disappeared = [z for z in zodiacs if freq_last20.get(z, 0) == 0]
    print(f"1. 消失生肖: {', '.join(disappeared) if disappeared else '无'}")
    
    # 找出爆发的生肖
    hot_zodiac = [(z, c) for z, c in freq_last20.items() if c >= 4]
    if hot_zodiac:
        print(f"2. 爆发生肖: {', '.join([f'{z}({c}次)' for z, c in hot_zodiac])}")
    
    # 多样性变化
    diversity_early = len(freq_early20) / 12 * 100
    diversity_prev = len(freq_prev20) / 12 * 100
    diversity_last = len(freq_last20) / 12 * 100
    
    print(f"\n3. 多样性变化:")
    print(f"   早期20期: {diversity_early:.1f}% ({len(freq_early20)}/12生肖出现)")
    print(f"   前20期: {diversity_prev:.1f}% ({len(freq_prev20)}/12生肖出现)")
    print(f"   最后20期: {diversity_last:.1f}% ({len(freq_last20)}/12生肖出现)")
    
    # 分析模型预测偏差
    print("\n【模型预测偏差分析】\n")
    
    # 读取验证结果
    validation_df = pd.read_csv('retrained_model_validation_100periods.csv', encoding='utf-8-sig')
    last20_predictions = validation_df.tail(20)
    
    # 统计预测的生肖
    predicted_zodiacs = []
    for _, row in last20_predictions.iterrows():
        preds = row['predicted_top4'].split(', ')
        predicted_zodiacs.extend(preds)
    
    pred_freq = Counter(predicted_zodiacs)
    
    print("模型预测频次 vs 实际出现:")
    print(f"{'生肖':>4} {'预测次数':>10} {'实际出现':>10} {'偏差'}")
    print("-"*40)
    
    for zodiac in zodiacs:
        pred_count = pred_freq.get(zodiac, 0)
        actual_count = freq_last20.get(zodiac, 0)
        bias = pred_count - actual_count
        
        if abs(bias) >= 5:
            marker = "⚠️"
        elif bias > 0:
            marker = "↑"
        elif bias < 0:
            marker = "↓"
        else:
            marker = "="
        
        print(f"{zodiac:>4} {pred_count:>8}次 {actual_count:>10}次 {bias:>+4} {marker}")
    
    print("\n【结论】\n")
    
    # 计算最大偏差
    max_bias = max([abs(pred_freq.get(z, 0) - freq_last20.get(z, 0)) for z in zodiacs])
    
    if max_bias >= 10:
        print("❌ 严重偏差：模型预测与实际相差超过10次")
    elif max_bias >= 5:
        print("⚠️  中度偏差：模型预测存在明显偏差")
    else:
        print("✅ 轻度偏差：模型预测基本准确")
    
    # 给出建议
    print("\n【改进建议】\n")
    
    if len(disappeared) >= 2:
        print(f"1. 模型未能预测到{len(disappeared)}个生肖会消失（{', '.join(disappeared)}）")
        print("   → 建议：增加'生肖缺失预警'功能")
    
    if hot_zodiac:
        hot_names = [z for z, _ in hot_zodiac]
        print(f"2. 模型未充分预测爆发生肖（{', '.join(hot_names)}）")
        print("   → 建议：加强'热门惯性'权重")
    
    if diversity_last < 80:
        print(f"3. 最后20期多样性降低到{diversity_last:.1f}%")
        print("   → 建议：使用更短的时间窗口（10-15期）训练")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    analyze_last20_failure()
