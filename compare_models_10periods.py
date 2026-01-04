"""
回测所有预测模型在最近10期的Top 15成功率
"""

import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from simple_predictor import SimplePredictor
from hybrid_predictor import HybridPredictor
from lucky_number_predictor import LuckyNumberPredictor


def test_simple_predictor(numbers, test_range):
    """测试简化预测器"""
    predictor = SimplePredictor()
    results = []
    
    for i in test_range:
        actual = numbers[i]
        history = numbers[:i]
        predictions = predictor.predict(history, top_k=15)
        hit = actual in predictions
        results.append(hit)
    
    return results


def test_hybrid_predictor(numbers, test_range):
    """测试混合预测器"""
    predictor = HybridPredictor()
    results = []
    
    for i in test_range:
        actual = numbers[i]
        history = numbers[:i]
        try:
            predictions = predictor.predict_hybrid(history, top_k=15, use_ml=False)
            hit = actual in predictions
        except:
            hit = False
        results.append(hit)
    
    return results


def test_ml_predictor(numbers, test_range, model_type='lightgbm'):
    """测试ML预测器"""
    results = []
    
    for i in test_range:
        actual = numbers[i]
        history = numbers[:i]
        
        try:
            predictor = LuckyNumberPredictor()
            # 临时创建数据文件
            temp_df = pd.DataFrame({
                'date': range(len(history)),
                'number': history,
                'animal': ['鼠'] * len(history),
                'element': ['金'] * len(history)
            })
            temp_df.to_csv('temp_test.csv', index=False, encoding='utf-8-sig')
            
            predictor.load_data('temp_test.csv')
            predictor.train_model(model_type, test_size=0.15)
            predictions = predictor.predict_top_probabilities(top_k=15)
            pred_numbers = [p['number'] for p in predictions]
            hit = actual in pred_numbers
        except:
            hit = False
        
        results.append(hit)
    
    # 清理临时文件
    import os
    if os.path.exists('temp_test.csv'):
        os.remove('temp_test.csv')
    
    return results


def main():
    print("=" * 80)
    print("回测所有预测模型 - 最近10期Top 15成功率对比")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    total_periods = len(numbers)
    test_size = 10
    test_range = range(total_periods - test_size, total_periods)
    
    print(f"\n数据集: {total_periods}期")
    print(f"测试范围: 第{total_periods - test_size + 1}期 - 第{total_periods}期")
    print(f"实际数字: {numbers[test_range[0]:].tolist()}")
    
    print("\n" + "=" * 80)
    print("开始回测...")
    print("=" * 80)
    
    # 测试1: 简化预测器
    print("\n【模型1】简化统计预测器 (Simple Predictor)")
    print("-" * 80)
    simple_results = test_simple_predictor(numbers, test_range)
    simple_rate = sum(simple_results) / len(simple_results) * 100
    print(f"命中情况: {['✅' if r else '❌' for r in simple_results]}")
    print(f"命中次数: {sum(simple_results)}/{len(simple_results)}")
    print(f"成功率: {simple_rate:.1f}%")
    
    # 测试2: 混合预测器（不含ML）
    print("\n【模型2】混合预测器 - 纯统计版 (Hybrid Without ML)")
    print("-" * 80)
    hybrid_results = test_hybrid_predictor(numbers, test_range)
    hybrid_rate = sum(hybrid_results) / len(hybrid_results) * 100
    print(f"命中情况: {['✅' if r else '❌' for r in hybrid_results]}")
    print(f"命中次数: {sum(hybrid_results)}/{len(hybrid_results)}")
    print(f"成功率: {hybrid_rate:.1f}%")
    
    # 测试3: LightGBM预测器
    print("\n【模型3】LightGBM机器学习预测器")
    print("-" * 80)
    print("注意: ML模型需要逐期训练，耗时较长...")
    ml_results = test_ml_predictor(numbers, test_range, 'lightgbm')
    ml_rate = sum(ml_results) / len(ml_results) * 100
    print(f"命中情况: {['✅' if r else '❌' for r in ml_results]}")
    print(f"命中次数: {sum(ml_results)}/{len(ml_results)}")
    print(f"成功率: {ml_rate:.1f}%")
    
    # 汇总对比
    print("\n" + "=" * 80)
    print("综合对比 - Top 15 成功率排名")
    print("=" * 80)
    
    models = [
        ("简化统计预测器", simple_rate, simple_results),
        ("混合预测器(纯统计)", hybrid_rate, hybrid_results),
        ("LightGBM机器学习", ml_rate, ml_results)
    ]
    
    # 排序
    models.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'排名':<6}{'模型':<20}{'成功率':<12}{'命中次数':<12}{'评级':<10}")
    print("-" * 80)
    
    for rank, (name, rate, results) in enumerate(models, 1):
        if rate >= 60:
            grade = "🏆 优秀"
        elif rate >= 50:
            grade = "✅ 良好"
        elif rate >= 40:
            grade = "🟢 合格"
        else:
            grade = "⚠️  一般"
        
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        hits = sum(results)
        total = len(results)
        
        print(f"{medal} {rank:<3}{name:<20}{rate:>5.1f}%{' '*6}{hits}/{total}{' '*6}{grade}")
    
    # 随机概率对比
    random_rate = 15 / 49 * 100
    print(f"\n随机概率基准: {random_rate:.1f}% (15/49)")
    
    print("\n提升倍数:")
    for name, rate, _ in models:
        improvement = rate / random_rate
        print(f"  {name}: {improvement:.2f}x")
    
    # 详细结果
    print("\n" + "=" * 80)
    print("逐期详细对比")
    print("=" * 80)
    
    print(f"\n{'期数':<8}{'实际':<6}{'简化':<6}{'混合':<6}{'ML':<6}")
    print("-" * 80)
    
    for idx, i in enumerate(test_range):
        period = i + 1
        actual = numbers[i]
        s = "✅" if simple_results[idx] else "❌"
        h = "✅" if hybrid_results[idx] else "❌"
        m = "✅" if ml_results[idx] else "❌"
        print(f"{period:<8}{actual:<6}{s:<6}{h:<6}{m:<6}")
    
    # 最佳模型推荐
    print("\n" + "=" * 80)
    print("推荐结论")
    print("=" * 80)
    
    best_model, best_rate, _ = models[0]
    
    print(f"\n🏆 最佳模型: {best_model}")
    print(f"   Top 15成功率: {best_rate:.1f}%")
    
    if best_rate >= 60:
        print(f"   状态: ✅ 已达到60%目标!")
    elif best_rate >= 50:
        print(f"   状态: 接近目标，建议继续优化")
    else:
        print(f"   状态: 距离60%目标还有{60-best_rate:.1f}%差距")
    
    print(f"\n💡 使用建议:")
    if best_rate >= 50:
        print(f"   推荐使用 {best_model}")
        print(f"   预期命中率: {best_rate:.1f}%")
    else:
        print(f"   建议组合使用多个模型")
        print(f"   取交集或按权重融合结果")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
