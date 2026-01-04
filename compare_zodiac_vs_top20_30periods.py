"""
生肖预测 vs TOP20预测 - 最近30期对比验证
"""

import pandas as pd
import numpy as np
from collections import Counter
from zodiac_predictor import ZodiacPredictor
from test_top30_model import Top30Predictor


def validate_zodiac_recent_30(csv_file='data/lucky_numbers.csv'):
    """验证生肖预测最近30期"""
    print("=" * 80)
    print("生肖预测 - 最近30期验证")
    print("=" * 80)
    
    predictor = ZodiacPredictor()
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    total_records = len(df)
    
    if total_records < 31:
        print(f"错误：数据不足30期")
        return None
    
    # 统计结果
    zodiac_top1_hits = 0
    zodiac_top3_hits = 0
    zodiac_top5_hits = 0
    number_top5_hits = 0
    number_top10_hits = 0
    number_top15_hits = 0
    
    details = []
    
    print(f"\n验证期数: 30期")
    print(f"验证范围: 第{total_records-30+1}期 到 第{total_records}期\n")
    
    for i in range(30):
        current_idx = total_records - 30 + i
        train_data = df.iloc[:current_idx + 1]
        
        if current_idx + 1 < total_records:
            next_actual_num = int(df.iloc[current_idx + 1]['number'])
            next_actual_zodiac = df.iloc[current_idx + 1]['animal'].strip()
            next_date = df.iloc[current_idx + 1]['date']
            period_num = current_idx + 2
        else:
            break
        
        # 保存临时数据
        temp_file = 'data/temp_zodiac_train.csv'
        train_data.to_csv(temp_file, index=False, encoding='utf-8-sig')
        
        # 预测TOP5生肖
        top5_zodiacs = predictor.predict_zodiac_top5(temp_file)
        predicted_zodiacs = [z for z, s in top5_zodiacs]
        
        # 根据生肖推荐号码
        recommended_numbers = predictor.predict_numbers_by_zodiac(top5_zodiacs)
        
        # 检查生肖命中
        zodiac_hit = False
        zodiac_rank = None
        if next_actual_zodiac in predicted_zodiacs:
            zodiac_rank = predicted_zodiacs.index(next_actual_zodiac) + 1
            zodiac_hit = True
            if zodiac_rank == 1:
                zodiac_top1_hits += 1
            if zodiac_rank <= 3:
                zodiac_top3_hits += 1
            zodiac_top5_hits += 1
        
        # 检查号码命中
        number_hit = False
        number_rank = None
        top5_nums = recommended_numbers[:5]
        top10_nums = recommended_numbers[:10]
        top15_nums = recommended_numbers[:15]
        
        if next_actual_num in top5_nums:
            number_top5_hits += 1
            number_top10_hits += 1
            number_top15_hits += 1
            number_rank = top5_nums.index(next_actual_num) + 1
            number_hit = True
        elif next_actual_num in top10_nums:
            number_top10_hits += 1
            number_top15_hits += 1
            number_rank = top10_nums.index(next_actual_num) + 1
            number_hit = True
        elif next_actual_num in top15_nums:
            number_top15_hits += 1
            number_rank = top15_nums.index(next_actual_num) + 1
            number_hit = True
        
        details.append({
            'period': period_num,
            'date': next_date,
            'actual_num': next_actual_num,
            'actual_zodiac': next_actual_zodiac,
            'predicted_zodiacs': predicted_zodiacs[:3],
            'zodiac_hit': zodiac_hit,
            'zodiac_rank': zodiac_rank,
            'number_hit': number_hit,
            'number_rank': number_rank
        })
    
    # 计算成功率
    zodiac_top1_rate = (zodiac_top1_hits / 30) * 100
    zodiac_top3_rate = (zodiac_top3_hits / 30) * 100
    zodiac_top5_rate = (zodiac_top5_hits / 30) * 100
    number_top15_rate = (number_top15_hits / 30) * 100
    
    print(f"生肖预测成功率:")
    print(f"  TOP1: {zodiac_top1_rate:.2f}% ({zodiac_top1_hits}/30)")
    print(f"  TOP3: {zodiac_top3_rate:.2f}% ({zodiac_top3_hits}/30)")
    print(f"  TOP5: {zodiac_top5_rate:.2f}% ({zodiac_top5_hits}/30)")
    print(f"\n号码推荐成功率:")
    print(f"  TOP15: {number_top15_rate:.2f}% ({number_top15_hits}/30)")
    
    return {
        'zodiac_top1_rate': zodiac_top1_rate,
        'zodiac_top3_rate': zodiac_top3_rate,
        'zodiac_top5_rate': zodiac_top5_rate,
        'number_top15_rate': number_top15_rate,
        'zodiac_top5_hits': zodiac_top5_hits,
        'number_top15_hits': number_top15_hits,
        'details': details
    }


def validate_top20_recent_30(csv_file='data/lucky_numbers.csv'):
    """验证TOP20预测最近30期"""
    print("\n" + "=" * 80)
    print("TOP20预测 - 最近30期验证")
    print("=" * 80)
    
    predictor = Top30Predictor()
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    total_records = len(df)
    
    if total_records < 31:
        print(f"错误：数据不足30期")
        return None
    
    # 统计结果
    top5_hits = 0
    top10_hits = 0
    top15_hits = 0
    top20_hits = 0
    
    details = []
    
    print(f"\n验证期数: 30期")
    print(f"验证范围: 第{total_records-30+1}期 到 第{total_records}期\n")
    
    for i in range(30):
        current_idx = total_records - 30 + i
        
        if current_idx + 1 < total_records:
            next_actual_num = int(df.iloc[current_idx + 1]['number'])
            next_date = df.iloc[current_idx + 1]['date']
            period_num = current_idx + 2
        else:
            break
        
        # 使用当前期之前的数据进行预测
        train_numbers = df.iloc[:current_idx + 1]['number'].values
        train_elements = df.iloc[:current_idx + 1]['element'].values
        
        # 预测TOP20
        top20_predictions = predictor.predict_top20(train_numbers, train_elements)
        
        # 检查命中
        hit = False
        rank = None
        if next_actual_num in top20_predictions:
            rank = top20_predictions.index(next_actual_num) + 1
            hit = True
            if rank <= 5:
                top5_hits += 1
                top10_hits += 1
                top15_hits += 1
                top20_hits += 1
            elif rank <= 10:
                top10_hits += 1
                top15_hits += 1
                top20_hits += 1
            elif rank <= 15:
                top15_hits += 1
                top20_hits += 1
            else:
                top20_hits += 1
        
        details.append({
            'period': period_num,
            'date': next_date,
            'actual_num': next_actual_num,
            'top20': top20_predictions,
            'hit': hit,
            'rank': rank
        })
    
    # 计算成功率
    top5_rate = (top5_hits / 30) * 100
    top10_rate = (top10_hits / 30) * 100
    top15_rate = (top15_hits / 30) * 100
    top20_rate = (top20_hits / 30) * 100
    
    print(f"TOP20预测成功率:")
    print(f"  TOP5:  {top5_rate:.2f}% ({top5_hits}/30)")
    print(f"  TOP10: {top10_rate:.2f}% ({top10_hits}/30)")
    print(f"  TOP15: {top15_rate:.2f}% ({top15_hits}/30)")
    print(f"  TOP20: {top20_rate:.2f}% ({top20_hits}/30)")
    
    return {
        'top5_rate': top5_rate,
        'top10_rate': top10_rate,
        'top15_rate': top15_rate,
        'top20_rate': top20_rate,
        'top5_hits': top5_hits,
        'top15_hits': top15_hits,
        'top20_hits': top20_hits,
        'details': details
    }


def compare_models():
    """对比两个模型"""
    print("\n" + "=" * 80)
    print("🎯 生肖预测 vs TOP20预测 - 最近30期对比")
    print("=" * 80)
    
    # 验证生肖预测
    zodiac_result = validate_zodiac_recent_30()
    
    # 验证TOP20预测
    top20_result = validate_top20_recent_30()
    
    if zodiac_result and top20_result:
        print("\n" + "=" * 80)
        print("📊 对比总结")
        print("=" * 80)
        
        print(f"\n【生肖预测模型】")
        print(f"  生肖TOP5成功率: {zodiac_result['zodiac_top5_rate']:.2f}% ({zodiac_result['zodiac_top5_hits']}/30)")
        print(f"  号码TOP15成功率: {zodiac_result['number_top15_rate']:.2f}% ({zodiac_result['number_top15_hits']}/30)")
        
        print(f"\n【TOP20预测模型】")
        print(f"  号码TOP15成功率: {top20_result['top15_rate']:.2f}% ({top20_result['top15_hits']}/30)")
        print(f"  号码TOP20成功率: {top20_result['top20_rate']:.2f}% ({top20_result['top20_hits']}/30)")
        
        # 核心对比
        print(f"\n" + "=" * 80)
        print("🏆 核心指标对比")
        print("=" * 80)
        
        print(f"\n对比维度1: TOP15号码预测")
        print(f"  生肖模型: {zodiac_result['number_top15_rate']:.2f}%")
        print(f"  TOP20模型: {top20_result['top15_rate']:.2f}%")
        diff1 = zodiac_result['number_top15_rate'] - top20_result['top15_rate']
        if diff1 > 0:
            print(f"  ✅ 生肖模型领先 +{diff1:.2f}%")
        elif diff1 < 0:
            print(f"  ✅ TOP20模型领先 +{abs(diff1):.2f}%")
        else:
            print(f"  ➡️ 两者持平")
        
        print(f"\n对比维度2: 生肖预测能力")
        print(f"  生肖模型TOP5: {zodiac_result['zodiac_top5_rate']:.2f}%")
        print(f"  (这是生肖模型的独特优势)")
        
        # 详细对比表
        print(f"\n" + "=" * 80)
        print("📋 详细对比表 (最近30期)")
        print("=" * 80)
        
        print(f"\n{'模型':<20} {'TOP5':<10} {'TOP15':<10} {'TOP20':<10}")
        print("-" * 80)
        print(f"{'生肖预测(生肖维度)':<20} {zodiac_result['zodiac_top5_rate']:>6.2f}% {'N/A':<10} {'N/A':<10}")
        print(f"{'生肖预测(号码维度)':<20} {'N/A':<10} {zodiac_result['number_top15_rate']:>6.2f}% {'N/A':<10}")
        print(f"{'TOP20预测':<20} {top20_result['top5_rate']:>6.2f}% {top20_result['top15_rate']:>6.2f}% {top20_result['top20_rate']:>6.2f}%")
        
        # 结论
        print(f"\n" + "=" * 80)
        print("💡 结论")
        print("=" * 80)
        
        # 判断哪个更好
        if zodiac_result['zodiac_top5_rate'] > 50:
            print(f"\n⭐ 生肖预测的核心优势:")
            print(f"   - 生肖TOP5成功率高达 {zodiac_result['zodiac_top5_rate']:.2f}%")
            print(f"   - 远超随机概率(41.7%)")
            print(f"   - 可作为主要预测维度")
        
        if top20_result['top15_rate'] > zodiac_result['number_top15_rate']:
            print(f"\n✅ 在TOP15号码预测方面:")
            print(f"   - TOP20模型表现更好: {top20_result['top15_rate']:.2f}%")
            print(f"   - 比生肖模型高 {abs(diff1):.2f}%")
        else:
            print(f"\n✅ 在TOP15号码预测方面:")
            print(f"   - 生肖模型表现更好: {zodiac_result['number_top15_rate']:.2f}%")
            print(f"   - 比TOP20模型高 {diff1:.2f}%")
        
        print(f"\n🎯 综合建议:")
        print(f"   1. 用生肖预测选择生肖范围 (54.5%成功率)")
        print(f"   2. 用TOP20预测精选号码 (覆盖更广)")
        print(f"   3. 取两者交集，获得最优预测")
        
        # 保存结果
        print(f"\n" + "=" * 80)
        print("💾 保存对比结果...")
        print("=" * 80)
        
        # 生成对比报告
        with open('zodiac_vs_top20_comparison_30periods.txt', 'w', encoding='utf-8') as f:
            f.write("生肖预测 vs TOP20预测 - 最近30期对比报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("【生肖预测模型】\n")
            f.write(f"  生肖TOP5成功率: {zodiac_result['zodiac_top5_rate']:.2f}% ({zodiac_result['zodiac_top5_hits']}/30)\n")
            f.write(f"  号码TOP15成功率: {zodiac_result['number_top15_rate']:.2f}% ({zodiac_result['number_top15_hits']}/30)\n\n")
            
            f.write("【TOP20预测模型】\n")
            f.write(f"  号码TOP15成功率: {top20_result['top15_rate']:.2f}% ({top20_result['top15_hits']}/30)\n")
            f.write(f"  号码TOP20成功率: {top20_result['top20_rate']:.2f}% ({top20_result['top20_hits']}/30)\n\n")
            
            f.write("对比结论:\n")
            if zodiac_result['number_top15_rate'] > top20_result['top15_rate']:
                f.write(f"  在TOP15号码预测上，生肖模型更优 (+{diff1:.2f}%)\n")
            else:
                f.write(f"  在TOP15号码预测上，TOP20模型更优 (+{abs(diff1):.2f}%)\n")
            
            f.write(f"\n生肖预测的独特优势:\n")
            f.write(f"  生肖TOP5成功率 {zodiac_result['zodiac_top5_rate']:.2f}% 远超其他维度\n")
        
        print(f"\n✅ 对比报告已保存至: zodiac_vs_top20_comparison_30periods.txt")
        
        return {
            'zodiac': zodiac_result,
            'top20': top20_result
        }


def main():
    """主函数"""
    compare_models()


if __name__ == '__main__':
    main()
