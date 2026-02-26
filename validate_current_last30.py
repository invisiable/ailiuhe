"""
验证当前数据最近30期（第359-388期）的生肖TOP4预测命中率
使用与GUI相同的EnsembleZodiacPredictor模型
"""

import pandas as pd
from ensemble_zodiac_predictor import EnsembleZodiacPredictor

def validate_current_last30():
    print("="*90)
    print("当前数据最近30期（第359-388期）生肖TOP4预测验证")
    print("="*90)
    
    # 读取当前数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    animals = [str(a).strip() for a in df['animal'].values]
    numbers = df['number'].values
    dates = df['date'].values
    
    total_periods = len(df)
    print(f"\n数据文件总期数: {total_periods}")
    print(f"验证期数: 第{total_periods-29}-{total_periods}期")
    print(f"日期范围: {dates[-30]} 至 {dates[-1]}")
    
    # 生肖到号码的映射
    zodiac_numbers = {
        '鼠': [4, 16, 28, 40],
        '牛': [5, 17, 29, 41],
        '虎': [6, 18, 30, 42],
        '兔': [3, 7, 15, 19, 27, 31, 39, 43],
        '龙': [8, 20, 26, 32, 44],
        '蛇': [1, 9, 13, 21, 25, 33, 37, 45],
        '马': [10, 22, 34, 46],
        '羊': [11, 23, 35, 47],
        '猴': [12, 24, 36, 48],
        '鸡': [1, 13, 25, 37, 49],
        '狗': [2, 14, 26, 38],
        '猪': [3, 15, 27, 39]
    }
    
    # 使用EnsembleZodiacPredictor进行预测
    predictor = EnsembleZodiacPredictor()
    
    results = []
    hits = 0
    
    print(f"\n{'期数':<6} {'日期':<12} {'实际号':<6} {'实际生肖':<6} {'预测TOP4':<30} 结果")
    print("-"*90)
    
    start_idx = total_periods - 30
    
    for i in range(start_idx, total_periods):
        period = i + 1
        date = dates[i]
        actual_number = int(numbers[i])
        actual_animal = animals[i]
        
        # 使用之前所有数据进行预测
        history = animals[:i]
        prediction = predictor.predict_from_history(history, top_n=5, debug=False)
        top4 = prediction['top4']
        
        # 判断是否命中
        is_hit = actual_animal in top4
        if is_hit:
            hits += 1
            hit_symbol = '✅'
        else:
            hit_symbol = '❌'
        
        top4_str = ', '.join(top4)
        print(f"{period:<6} {date:<12} {actual_number:<6} {actual_animal:<6} {top4_str:<30} {hit_symbol}")
        
        results.append({
            'period': period,
            'date': date,
            'actual_number': actual_number,
            'actual_animal': actual_animal,
            'top4': top4_str,
            'is_hit': is_hit
        })
    
    hit_rate = hits / 30 * 100
    
    print("\n" + "="*90)
    print("【验证结果】")
    print("="*90)
    print(f"命中次数: {hits}/30")
    print(f"命中率: {hit_rate:.1f}%")
    print(f"理论命中率: 33.3% (4/12生肖)")
    print(f"提升幅度: {hit_rate - 33.3:+.1f}%")
    
    # 计算盈亏（固定1倍投注）
    fixed_profit = hits * 31 - 30 * 16  # 每次中奖+31元，不中-16元
    fixed_roi = fixed_profit / (30 * 16) * 100
    
    print(f"\n【固定1倍投注效果】")
    print(f"投注期数: 30期")
    print(f"单期投注: 16元（4个生肖×4元）")
    print(f"总投注: 480元")
    print(f"总盈利: {fixed_profit:+.0f}元")
    print(f"投资回报率: {fixed_roi:+.1f}%")
    
    # 分析各生肖命中情况
    print(f"\n【各生肖命中统计】")
    zodiac_hit = {}
    zodiac_total = {}
    
    for r in results:
        zodiac = r['actual_animal']
        zodiac_total[zodiac] = zodiac_total.get(zodiac, 0) + 1
        if r['is_hit']:
            zodiac_hit[zodiac] = zodiac_hit.get(zodiac, 0) + 1
    
    for zodiac in sorted(zodiac_total.keys()):
        hit_count = zodiac_hit.get(zodiac, 0)
        total_count = zodiac_total[zodiac]
        rate = hit_count / total_count * 100 if total_count > 0 else 0
        print(f"  {zodiac}: {hit_count}/{total_count} ({rate:.0f}%)")
    
    # 保存结果
    df_results = pd.DataFrame(results)
    df_results.to_csv('zodiac_top4_current_last30_validation.csv', index=False, encoding='utf-8-sig')
    print(f"\n📁 详细结果已保存: zodiac_top4_current_last30_validation.csv")
    
    return hit_rate, fixed_profit

if __name__ == '__main__':
    validate_current_last30()
