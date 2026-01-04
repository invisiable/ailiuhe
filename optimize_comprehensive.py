"""
优化综合预测系统
调整权重和策略，目标：Top 5 命中率50%+
"""
import numpy as np
import pandas as pd
from collections import Counter
from lucky_number_predictor import LuckyNumberPredictor
import warnings
warnings.filterwarnings('ignore')


def optimize_weights(train_size=80, valid_size=20):
    """
    优化权重配置
    在验证集上找到最佳权重组合
    """
    print("="*80)
    print("优化权重配置")
    print("="*80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    # 尝试不同的权重组合
    weight_configs = [
        # [模型, 生肖, 五行, 频率, 趋势, 间隔]
        ([0.40, 0.15, 0.20, 0.10, 0.10, 0.05], "默认配置"),
        ([0.30, 0.20, 0.25, 0.10, 0.10, 0.05], "增强五行和生肖"),
        ([0.50, 0.10, 0.15, 0.10, 0.10, 0.05], "模型主导"),
        ([0.25, 0.20, 0.25, 0.15, 0.10, 0.05], "均衡配置"),
        ([0.20, 0.15, 0.30, 0.15, 0.15, 0.05], "五行主导"),
        ([0.30, 0.15, 0.20, 0.20, 0.10, 0.05], "频率增强"),
        ([0.35, 0.15, 0.20, 0.10, 0.15, 0.05], "趋势增强")
    ]
    
    print(f"\n在 {valid_size} 个样本上测试不同权重配置...\n")
    
    best_config = None
    best_rate = 0
    
    for weights, name in weight_configs:
        hits = 0
        total = 0
        
        for i in range(valid_size):
            test_index = train_size + i
            if test_index >= len(df):
                break
            
            train_df = df.iloc[:test_index].copy()
            actual_row = df.iloc[test_index]
            
            temp_file = 'data/temp_train.csv'
            train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
            
            try:
                from comprehensive_predictor import ComprehensivePredictor
                
                predictor = LuckyNumberPredictor()
                predictor.load_data(temp_file, number_column='number', date_column='date',
                                   animal_column='animal', element_column='element')
                predictor.train_model('gradient_boosting', test_size=0.2)
                
                comp_predictor = ComprehensivePredictor(predictor)
                predictions = comp_predictor.comprehensive_predict(top_k=5, weights=weights)
                
                actual_number = actual_row['number']
                predicted_numbers = [p['number'] for p in predictions]
                
                if actual_number in predicted_numbers:
                    hits += 1
                total += 1
                
            except:
                continue
        
        hit_rate = (hits / total * 100) if total > 0 else 0
        print(f"{name:<25} 命中率: {hit_rate:>5.1f}% ({hits}/{total})")
        
        if hit_rate > best_rate:
            best_rate = hit_rate
            best_config = (weights, name)
    
    print(f"\n最佳配置: {best_config[1]}")
    print(f"权重: {best_config[0]}")
    print(f"命中率: {best_rate:.1f}%")
    
    return best_config[0]


def test_with_different_topk(weights=None, train_size=100, test_samples=20):
    """测试不同Top K的命中率"""
    print("\n" + "="*80)
    print("测试不同Top K的命中率")
    print("="*80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    topk_configs = [3, 5, 8, 10]
    
    print(f"\n测试 {test_samples} 个样本\n")
    
    for top_k in topk_configs:
        hits = 0
        total = 0
        
        for i in range(test_samples):
            test_index = train_size + i
            if test_index >= len(df):
                break
            
            train_df = df.iloc[:test_index].copy()
            actual_row = df.iloc[test_index]
            
            temp_file = 'data/temp_train.csv'
            train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
            
            try:
                from comprehensive_predictor import ComprehensivePredictor
                
                predictor = LuckyNumberPredictor()
                predictor.load_data(temp_file, number_column='number', date_column='date',
                                   animal_column='animal', element_column='element')
                predictor.train_model('gradient_boosting', test_size=0.2)
                
                comp_predictor = ComprehensivePredictor(predictor)
                predictions = comp_predictor.comprehensive_predict(top_k=top_k, weights=weights)
                
                actual_number = actual_row['number']
                predicted_numbers = [p['number'] for p in predictions]
                
                if actual_number in predicted_numbers:
                    hits += 1
                total += 1
                
            except Exception as e:
                continue
        
        hit_rate = (hits / total * 100) if total > 0 else 0
        target_status = "[达标]" if hit_rate >= 50 else "[未达标]"
        print(f"Top {top_k:<3} 命中率: {hit_rate:>5.1f}% ({hits:>2}/{total}) {target_status}")


if __name__ == "__main__":
    # 1. 优化权重
    print("\n步骤1: 优化权重配置\n")
    best_weights = optimize_weights(train_size=80, valid_size=20)
    
    # 2. 测试不同Top K
    print("\n\n步骤2: 测试不同Top K")
    test_with_different_topk(weights=best_weights, train_size=100, test_samples=20)
    
    print("\n" + "="*80)
    print("优化完成")
    print("="*80)
    print(f"\n推荐配置:")
    print(f"  最佳权重: {best_weights}")
    print(f"  使用 Top 8-10 可达到更高命中率")
    print("\n" + "="*80)
