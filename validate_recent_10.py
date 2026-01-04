"""
综合预测模型 - 最近10期的预测成功率验证
滚动预测：每次用之前的所有数据预测下一期
"""
import sys
sys.path.insert(0, 'd:\\AIagent')

import pandas as pd
from enhanced_predictor_v2 import EnhancedPredictor
from lucky_number_predictor import LuckyNumberPredictor
import os

def validate_recent_10_periods():
    """验证最近10期的预测成功率"""
    print("=" * 80)
    print("综合预测模型 - 最近10期滚动预测验证")
    print("=" * 80)
    
    # 读取完整数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_records = len(df)
    
    print(f"\n数据集信息:")
    print(f"  总记录数: {total_records}")
    print(f"  最早日期: {df['date'].iloc[0]}")
    print(f"  最新日期: {df['date'].iloc[-1]}")
    print(f"  验证期数: 最近10期 (第{total_records-9}期 至 第{total_records}期)")
    
    print("\n" + "=" * 80)
    print("开始滚动预测...")
    print("=" * 80)
    
    results = []
    top5_hits = 0
    top10_hits = 0
    
    # 对最近10期进行滚动预测
    for i in range(10):
        test_index = total_records - 10 + i
        train_size = test_index
        
        period_num = test_index + 1
        actual_row = df.iloc[test_index]
        actual_number = actual_row['number']
        actual_date = actual_row['date']
        
        print(f"\n{'='*80}")
        print(f"预测第 {period_num} 期 ({actual_date})")
        print(f"{'='*80}")
        print(f"使用前 {train_size} 期数据训练...")
        
        # 准备训练数据
        train_df = df.iloc[:train_size].copy()
        temp_file = f'data/temp_train_{i}.csv'
        train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        
        try:
            # 训练3个模型
            predictors = []
            for model_type in ['gradient_boosting', 'lightgbm', 'xgboost']:
                pred = LuckyNumberPredictor()
                pred.load_data(temp_file, 'number', 'date', 'animal', 'element')
                pred.train_model(model_type, test_size=0.2)
                predictors.append(pred)
            
            # 创建增强预测器
            enhanced = EnhancedPredictor(predictors)
            predictions = enhanced.comprehensive_predict_v2(top_k=10)
            
            # 提取预测结果
            top10_numbers = [pred['number'] for pred in predictions[:10]]
            top5_numbers = top10_numbers[:5]
            
            print(f"\n预测 Top 10: {top10_numbers}")
            print(f"预测 Top 5:  {top5_numbers}")
            print(f"实际数字:    {actual_number}")
            
            # 检查命中情况
            if actual_number in top5_numbers:
                rank = top5_numbers.index(actual_number) + 1
                status = f"✅ Top 5 命中! (排名第{rank})"
                top5_hits += 1
                top10_hits += 1
                hit_top5 = True
                hit_top10 = True
            elif actual_number in top10_numbers:
                rank = top10_numbers.index(actual_number) + 1
                status = f"✓ Top 10 命中 (排名第{rank})"
                top10_hits += 1
                hit_top5 = False
                hit_top10 = True
            else:
                status = "❌ 未命中"
                rank = None
                hit_top5 = False
                hit_top10 = False
            
            print(f"结果: {status}")
            
            results.append({
                'period': period_num,
                'date': actual_date,
                'actual': actual_number,
                'top5': top5_numbers,
                'top10': top10_numbers,
                'hit_top5': hit_top5,
                'hit_top10': hit_top10,
                'rank': rank,
                'status': status
            })
            
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            results.append({
                'period': period_num,
                'date': actual_date,
                'actual': actual_number,
                'top5': [],
                'top10': [],
                'hit_top5': False,
                'hit_top10': False,
                'rank': None,
                'status': f"错误: {e}"
            })
    
    # 打印汇总结果
    print("\n" + "=" * 80)
    print("📊 验证结果汇总")
    print("=" * 80)
    
    print(f"\n{'期数':<8} {'日期':<12} {'实际':<6} {'命中情况':<30} {'排名'}")
    print("-" * 80)
    
    for r in results:
        rank_str = f"第{r['rank']}名" if r['rank'] else "-"
        print(f"{r['period']:<8} {r['date']:<12} {r['actual']:<6} {r['status']:<30} {rank_str}")
    
    # 统计成功率
    print("\n" + "=" * 80)
    print("📈 成功率统计")
    print("=" * 80)
    
    total_tests = len(results)
    top5_rate = (top5_hits / total_tests * 100) if total_tests > 0 else 0
    top10_rate = (top10_hits / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n总测试期数: {total_tests}")
    print(f"\nTop 5 命中:")
    print(f"  命中次数: {top5_hits}")
    print(f"  成功率: {top5_rate:.1f}%")
    
    print(f"\nTop 10 命中:")
    print(f"  命中次数: {top10_hits}")
    print(f"  成功率: {top10_rate:.1f}%")
    
    # 详细命中信息
    if top5_hits > 0:
        print(f"\n✅ Top 5 命中详情:")
        for r in results:
            if r['hit_top5']:
                print(f"   第{r['period']}期 ({r['date']}): 数字 {r['actual']} (排名第{r['rank']})")
    
    if top10_hits > top5_hits:
        print(f"\n✓ Top 10 命中详情 (仅Top 6-10):")
        for r in results:
            if r['hit_top10'] and not r['hit_top5']:
                print(f"   第{r['period']}期 ({r['date']}): 数字 {r['actual']} (排名第{r['rank']})")
    
    # 性能评估
    print("\n" + "=" * 80)
    print("🎯 性能评估")
    print("=" * 80)
    
    print(f"\n目标对比:")
    print(f"  Top 5 目标: 20%  |  实际: {top5_rate:.1f}%  |  ", end="")
    if top5_rate >= 20:
        print("✅ 达标")
    elif top5_rate >= 15:
        print("🟡 接近")
    else:
        print("🔴 待提升")
    
    print(f"  Top 10 目标: 30% |  实际: {top10_rate:.1f}%  |  ", end="")
    if top10_rate >= 30:
        print("✅ 达标")
    elif top10_rate >= 25:
        print("🟡 接近")
    else:
        print("🔴 待提升")
    
    # 随机基准对比
    random_top5 = 5 / 49 * 100
    random_top10 = 10 / 49 * 100
    
    print(f"\n随机基准对比:")
    print(f"  Top 5:  随机 {random_top5:.1f}% vs 模型 {top5_rate:.1f}%  →  提升 {top5_rate/random_top5:.1f}x")
    print(f"  Top 10: 随机 {random_top10:.1f}% vs 模型 {top10_rate:.1f}%  →  提升 {top10_rate/random_top10:.1f}x")
    
    print("\n" + "=" * 80)
    
    return {
        'results': results,
        'top5_hits': top5_hits,
        'top10_hits': top10_hits,
        'total_tests': total_tests,
        'top5_rate': top5_rate,
        'top10_rate': top10_rate
    }

if __name__ == "__main__":
    try:
        data = validate_recent_10_periods()
        print("\n✅ 验证完成!")
        print(f"\n最终结果: Top 5 成功率 {data['top5_rate']:.1f}%, Top 10 成功率 {data['top10_rate']:.1f}%")
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
