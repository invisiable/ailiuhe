"""
验证综合预测结果 - 检查Top 10在哪期命中
"""
import sys
sys.path.insert(0, 'd:\\AIagent')

import pandas as pd
from enhanced_predictor_v2 import EnhancedPredictor
from lucky_number_predictor import LuckyNumberPredictor

def validate_current_predictions():
    """验证当前的综合预测结果"""
    print("=" * 80)
    print("验证综合预测结果 - Top 10 命中分析")
    print("=" * 80)
    
    # 读取完整数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_records = len(df)
    
    print(f"\n数据集信息:")
    print(f"  总记录数: {total_records}")
    print(f"  日期范围: {df['date'].iloc[0]} 至 {df['date'].iloc[-1]}")
    
    # 使用前面的数据训练，预测后面的
    train_size = total_records - 20  # 保留最后20期用于验证
    
    print(f"\n训练配置:")
    print(f"  训练集: 前 {train_size} 期")
    print(f"  验证集: 后 {total_records - train_size} 期")
    
    # 准备训练数据
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    temp_file = 'data/temp_train_current.csv'
    train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
    
    print(f"\n正在训练3个模型...")
    predictors = []
    for model_type in ['gradient_boosting', 'lightgbm', 'xgboost']:
        print(f"  - 训练 {model_type}...")
        pred = LuckyNumberPredictor()
        pred.load_data(temp_file, 'number', 'date', 'animal', 'element')
        pred.train_model(model_type, test_size=0.2)
        predictors.append(pred)
    
    print(f"\n创建增强预测器并执行预测...")
    enhanced = EnhancedPredictor(predictors)
    predictions = enhanced.comprehensive_predict_v2(top_k=10)
    
    # 提取Top 10数字
    top10_numbers = [pred['number'] for pred in predictions[:10]]
    top5_numbers = top10_numbers[:5]
    
    print("\n" + "=" * 80)
    print("📊 综合预测 Top 10 结果")
    print("=" * 80)
    print(f"{'排名':<6} {'数字':<6} {'综合概率':<12}")
    print("-" * 80)
    for i, pred in enumerate(predictions[:10], 1):
        marker = "⭐" if i <= 5 else "  "
        print(f"{marker} {i:>2}.    {pred['number']:>2}     {pred['probability']:>6.4f}")
    
    print("\n" + "=" * 80)
    print("🔍 验证结果 - 检查Top 10在后续哪期命中")
    print("=" * 80)
    
    # 检查每一期
    hits_top10 = []
    hits_top5 = []
    
    print(f"\n{'期数':<8} {'日期':<12} {'实际数字':<10} {'命中情况':<20} {'排名'}")
    print("-" * 80)
    
    for idx, row in test_df.iterrows():
        period = idx + 1
        date = row['date']
        actual = row['number']
        
        if actual in top10_numbers:
            rank = top10_numbers.index(actual) + 1
            if rank <= 5:
                status = "✅ Top 5 命中!"
                hits_top5.append((period, date, actual, rank))
            else:
                status = "✓ Top 10 命中"
            hits_top10.append((period, date, actual, rank))
            print(f"{period:<8} {date:<12} {actual:<10} {status:<20} 第{rank}名")
        else:
            print(f"{period:<8} {date:<12} {actual:<10} ❌ 未命中")
    
    # 统计结果
    print("\n" + "=" * 80)
    print("📈 统计摘要")
    print("=" * 80)
    
    total_test = len(test_df)
    top5_hits = len(hits_top5)
    top10_hits = len(hits_top10)
    
    print(f"\n测试期数: {total_test}")
    print(f"\nTop 5 预测: {top5_numbers}")
    print(f"Top 5 命中: {top5_hits} 次")
    print(f"Top 5 命中率: {top5_hits/total_test*100:.1f}%")
    
    print(f"\nTop 10 预测: {top10_numbers}")
    print(f"Top 10 命中: {top10_hits} 次")
    print(f"Top 10 命中率: {top10_hits/total_test*100:.1f}%")
    
    if hits_top5:
        print(f"\n✅ Top 5 命中详情:")
        for period, date, actual, rank in hits_top5:
            print(f"   第{period}期 ({date}): 数字 {actual} (排名第{rank})")
    else:
        print(f"\n❌ Top 5 未命中")
    
    if hits_top10:
        print(f"\n✓ Top 10 命中详情:")
        for period, date, actual, rank in hits_top10:
            print(f"   第{period}期 ({date}): 数字 {actual} (排名第{rank})")
    else:
        print(f"\n❌ Top 10 未命中")
    
    # 性能评估
    print("\n" + "=" * 80)
    print("🎯 性能评估")
    print("=" * 80)
    
    if top5_hits / total_test >= 0.20:
        print(f"Top 5 命中率: {top5_hits/total_test*100:.1f}% - ✅ 达标 (目标20%)")
    elif top5_hits / total_test >= 0.15:
        print(f"Top 5 命中率: {top5_hits/total_test*100:.1f}% - 🟡 接近目标 (目标20%)")
    else:
        print(f"Top 5 命中率: {top5_hits/total_test*100:.1f}% - 🔴 需要改进 (目标20%)")
    
    if top10_hits / total_test >= 0.30:
        print(f"Top 10 命中率: {top10_hits/total_test*100:.1f}% - ✅ 达标 (目标30%)")
    elif top10_hits / total_test >= 0.25:
        print(f"Top 10 命中率: {top10_hits/total_test*100:.1f}% - 🟡 接近目标 (目标30%)")
    else:
        print(f"Top 10 命中率: {top10_hits/total_test*100:.1f}% - 🔴 需要改进 (目标30%)")
    
    print("\n" + "=" * 80)
    
    # 清理临时文件
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return {
        'top5_numbers': top5_numbers,
        'top10_numbers': top10_numbers,
        'top5_hits': top5_hits,
        'top10_hits': top10_hits,
        'total_test': total_test,
        'hits_top5': hits_top5,
        'hits_top10': hits_top10
    }

if __name__ == "__main__":
    try:
        results = validate_current_predictions()
        print("\n✅ 验证完成")
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
