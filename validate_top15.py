"""
测试 Top 15 预测成功率
使用综合预测模型V2
"""
import sys
sys.path.insert(0, 'd:\\AIagent')

import pandas as pd
from enhanced_predictor_v2 import EnhancedPredictor
from lucky_number_predictor import LuckyNumberPredictor
import os

def validate_top15(test_periods=10):
    """验证Top 15的预测成功率"""
    print("=" * 80)
    print("综合预测模型 - Top 15 成功率验证")
    print("=" * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_records = len(df)
    
    print(f"\n数据集: {total_records}期")
    print(f"验证: 最近{test_periods}期")
    print(f"预测范围: Top 15\n")
    
    top5_hits = 0
    top10_hits = 0
    top15_hits = 0
    hit_details = []
    
    for i in range(test_periods):
        test_index = total_records - test_periods + i
        period_num = test_index + 1
        
        train_df = df.iloc[:test_index]
        actual = df.iloc[test_index]['number']
        actual_date = df.iloc[test_index]['date']
        
        print(f"{'='*80}")
        print(f"测试第{period_num}期 ({actual_date}), 实际: {actual}")
        
        # 保存临时文件
        temp_file = f'data/temp_top15_{i}.csv'
        train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        
        try:
            # 训练3个模型
            print(f"使用前{test_index}期数据训练...")
            predictors = []
            for model_type in ['gradient_boosting', 'lightgbm', 'xgboost']:
                pred = LuckyNumberPredictor()
                pred.load_data(temp_file, 'number', 'date', 'animal', 'element')
                pred.train_model(model_type, test_size=0.2)
                predictors.append(pred)
            
            # 创建增强预测器
            enhanced = EnhancedPredictor(predictors)
            predictions = enhanced.comprehensive_predict_v2(top_k=15)
            
            # 提取预测结果
            top15_numbers = [pred['number'] for pred in predictions]
            top10_numbers = top15_numbers[:10]
            top5_numbers = top15_numbers[:5]
            
            print(f"\nTop 5:  {top5_numbers}")
            print(f"Top 10: {top10_numbers}")
            print(f"Top 15: {top15_numbers}")
            
            # 检查命中情况
            if actual in top5_numbers:
                rank = top5_numbers.index(actual) + 1
                status = f"✅ Top 5 命中! (第{rank}名)"
                top5_hits += 1
                top10_hits += 1
                top15_hits += 1
                hit_level = "Top 5"
            elif actual in top10_numbers:
                rank = top10_numbers.index(actual) + 1
                status = f"✓ Top 10 命中 (第{rank}名)"
                top10_hits += 1
                top15_hits += 1
                hit_level = "Top 10"
            elif actual in top15_numbers:
                rank = top15_numbers.index(actual) + 1
                status = f"○ Top 15 命中 (第{rank}名)"
                top15_hits += 1
                hit_level = "Top 15"
            else:
                status = "❌ 未命中"
                rank = None
                hit_level = None
            
            print(f"结果: {status}\n")
            
            hit_details.append({
                'period': period_num,
                'date': actual_date,
                'actual': actual,
                'rank': rank,
                'level': hit_level,
                'status': status
            })
            
            # 清理
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            print(f"❌ 错误: {e}\n")
            import traceback
            traceback.print_exc()
    
    # 统计结果
    print("=" * 80)
    print("📊 详细结果")
    print("=" * 80)
    
    print(f"\n{'期数':<8} {'日期':<12} {'实际':<6} {'命中情况':<35} {'排名'}")
    print("-" * 80)
    
    for detail in hit_details:
        rank_str = f"第{detail['rank']}名" if detail['rank'] else "-"
        print(f"{detail['period']:<8} {detail['date']:<12} {detail['actual']:<6} {detail['status']:<35} {rank_str}")
    
    # 统计摘要
    print("\n" + "=" * 80)
    print("📈 成功率统计")
    print("=" * 80)
    
    total = test_periods
    top5_rate = (top5_hits / total * 100)
    top10_rate = (top10_hits / total * 100)
    top15_rate = (top15_hits / total * 100)
    
    print(f"\n总测试期数: {total}")
    print(f"\n{'预测范围':<12} {'命中次数':<12} {'成功率':<12} {'vs 随机':<15} {'提升'}")
    print("-" * 80)
    
    random_top5 = 5 / 49 * 100
    random_top10 = 10 / 49 * 100
    random_top15 = 15 / 49 * 100
    
    print(f"{'Top 5':<12} {f'{top5_hits}/{total}':<12} {top5_rate:>6.1f}%     {random_top5:>6.1f}%         {top5_rate/random_top5:>5.1f}x")
    print(f"{'Top 10':<12} {f'{top10_hits}/{total}':<12} {top10_rate:>6.1f}%     {random_top10:>6.1f}%         {top10_rate/random_top10:>5.1f}x")
    print(f"{'Top 15':<12} {f'{top15_hits}/{total}':<12} {top15_rate:>6.1f}%     {random_top15:>6.1f}%         {top15_rate/random_top15:>5.1f}x")
    
    # 命中详情
    if top15_hits > 0:
        print("\n" + "=" * 80)
        print("✅ 命中详情")
        print("=" * 80)
        
        top5_list = [d for d in hit_details if d['level'] == 'Top 5']
        top10_list = [d for d in hit_details if d['level'] == 'Top 10']
        top15_list = [d for d in hit_details if d['level'] == 'Top 15']
        
        if top5_list:
            print(f"\nTop 5 命中 ({len(top5_list)}次):")
            for d in top5_list:
                print(f"  第{d['period']}期 ({d['date']}): 数字 {d['actual']} (排名第{d['rank']})")
        
        if top10_list:
            print(f"\nTop 10 命中 (第6-10名, {len(top10_list)}次):")
            for d in top10_list:
                print(f"  第{d['period']}期 ({d['date']}): 数字 {d['actual']} (排名第{d['rank']})")
        
        if top15_list:
            print(f"\nTop 15 命中 (第11-15名, {len(top15_list)}次):")
            for d in top15_list:
                print(f"  第{d['period']}期 ({d['date']}): 数字 {d['actual']} (排名第{d['rank']})")
    
    # 性能评估
    print("\n" + "=" * 80)
    print("🎯 性能评估")
    print("=" * 80)
    
    print(f"\nTop 5 成功率:  {top5_rate:>5.1f}%  ", end="")
    if top5_rate >= 20:
        print("✅ 达标 (目标20%)")
    elif top5_rate >= 15:
        print("🟡 接近 (目标20%)")
    else:
        print("🔴 待提升 (目标20%)")
    
    print(f"Top 10 成功率: {top10_rate:>5.1f}%  ", end="")
    if top10_rate >= 30:
        print("✅ 达标 (目标30%)")
    elif top10_rate >= 25:
        print("🟡 接近 (目标30%)")
    else:
        print("🔴 待提升 (目标30%)")
    
    print(f"Top 15 成功率: {top15_rate:>5.1f}%  ", end="")
    if top15_rate >= 40:
        print("✅ 达标 (目标40%)")
    elif top15_rate >= 35:
        print("🟡 接近 (目标40%)")
    elif top15_rate >= 30:
        print("🟢 良好 (目标40%)")
    else:
        print("🔴 待提升 (目标40%)")
    
    print("\n" + "=" * 80)
    
    return {
        'total': total,
        'top5_hits': top5_hits,
        'top10_hits': top10_hits,
        'top15_hits': top15_hits,
        'top5_rate': top5_rate,
        'top10_rate': top10_rate,
        'top15_rate': top15_rate,
        'details': hit_details
    }


if __name__ == "__main__":
    try:
        print("开始验证 Top 15 预测成功率...\n")
        results = validate_top15(test_periods=10)
        
        print("\n" + "=" * 80)
        print("✅ 验证完成!")
        print("=" * 80)
        print(f"\n最终结果:")
        print(f"  Top 5:  {results['top5_rate']:.1f}%")
        print(f"  Top 10: {results['top10_rate']:.1f}%")
        print(f"  Top 15: {results['top15_rate']:.1f}% ⭐")
        print("\n建议: 使用 Top 15 可获得更好的覆盖率!")
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
