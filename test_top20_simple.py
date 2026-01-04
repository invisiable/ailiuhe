"""
简化的Top 20测试 - 基于现有训练好的模型
"""

import pandas as pd
import numpy as np
from lucky_number_predictor import LuckyNumberPredictor
from enhanced_predictor_v2 import EnhancedPredictor


def analyze_extreme_trend(numbers, window=10):
    """分析极端值趋势"""
    recent = numbers[-window:]
    extreme_small = [n for n in recent if n <= 10]
    extreme_large = [n for n in recent if n >= 40]
    
    return {
        'small_count': len(extreme_small),
        'large_count': len(extreme_large),
        'small_ratio': len(extreme_small) / len(recent),
        'large_ratio': len(extreme_large) / len(recent)
    }


def get_extreme_candidates(numbers, existing, k=5):
    """获取极端值候选"""
    recent_5 = set(numbers[-5:])
    recent_30 = numbers[-30:]
    
    # 极小值候选 (1-10)，排除最近5期和已预测的
    small_candidates = [n for n in range(1, 11) 
                       if n not in recent_5 and n not in existing]
    
    # 极大值候选 (40-49)
    large_candidates = [n for n in range(40, 50) 
                       if n not in recent_5 and n not in existing]
    
    # 基于最近30期频率排序
    small_freq = {n: recent_30.count(n) for n in small_candidates}
    large_freq = {n: recent_30.count(n) for n in large_candidates}
    
    # 选择频率高的
    selected_small = sorted(small_freq.keys(), 
                           key=lambda x: small_freq[x], 
                           reverse=True)[:k//2 + k%2]
    selected_large = sorted(large_freq.keys(), 
                           key=lambda x: large_freq[x], 
                           reverse=True)[:k//2]
    
    return list(selected_small) + list(selected_large)


def predict_top20_enhanced():
    """
    Top 20预测 - 增强版
    使用训练好的模型 + 极端值补充
    """
    print("=" * 80)
    print("Top 20 预测策略测试")
    print("=" * 80)
    
    # 1. 训练模型
    print("\n📦 训练模型...")
    model_types = ['gradient_boosting', 'lightgbm', 'xgboost']
    predictors = []
    
    for model_type in model_types:
        print(f"   训练 {model_type}...")
        predictor = LuckyNumberPredictor()
        predictor.train(
            file_path='lucky_numbers.csv',
            number_col='number',
            date_col='date',
            animal_col='animal',
            element_col='element',
            model_type=model_type
        )
        predictors.append(predictor)
    
    # 2. 创建增强预测器
    enhanced = EnhancedPredictor(predictors)
    
    # 3. 读取数据
    df = pd.read_csv('lucky_numbers.csv')
    all_numbers = df['number'].tolist()
    
    # 4. 在最近10期上测试
    print("\n" + "=" * 80)
    print("在最近10期上验证")
    print("=" * 80)
    
    total_periods = len(df)
    results = {'top5': 0, 'top10': 0, 'top15': 0, 'top20': 0, 'details': []}
    
    for i in range(total_periods - 10, total_periods):
        # 使用前i期数据训练
        temp_df = df.iloc[:i]
        temp_file = f'temp_test_{i}.csv'
        temp_df.to_csv(temp_file, index=False)
        
        # 获取实际值
        actual = df.iloc[i]['number']
        
        print(f"\n第{i+1}期 (实际: {actual}):")
        
        # 获取Top 15预测
        try:
            top15 = enhanced.comprehensive_predict_v2(
                file_path=temp_file,
                number_col='number',
                date_col='date',
                animal_col='animal',
                element_col='element',
                top_k=15
            )
            predicted_top15 = [r['number'] for r in top15]
            
            # 分析极端值趋势
            hist_numbers = temp_df['number'].tolist()
            trend = analyze_extreme_trend(hist_numbers)
            
            print(f"  Top 15: {predicted_top15}")
            print(f"  极端值趋势: 小({trend['small_count']}) 大({trend['large_count']})")
            
            # 获取额外的极端值候选
            extra_candidates = get_extreme_candidates(hist_numbers, set(predicted_top15), k=5)
            
            # 合并Top 20
            top20 = predicted_top15 + extra_candidates
            print(f"  Top 20: {top20}")
            print(f"  新增极端值: {extra_candidates}")
            
            # 检查命中
            if actual in top20:
                rank = top20.index(actual) + 1
                print(f"  ✅ 命中! 排名: {rank}")
                
                if rank <= 5:
                    results['top5'] += 1
                if rank <= 10:
                    results['top10'] += 1
                if rank <= 15:
                    results['top15'] += 1
                results['top20'] += 1
            else:
                print(f"  ❌ 未命中")
            
            # 保存详情
            results['details'].append({
                'period': i + 1,
                'actual': actual,
                'predicted': top20,
                'hit': actual in top20,
                'rank': top20.index(actual) + 1 if actual in top20 else -1
            })
            
        except Exception as e:
            print(f"  ⚠️ 预测失败: {str(e)}")
        
        # 清理临时文件
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # 5. 统计结果
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    
    total = len(results['details'])
    if total > 0:
        top5_rate = results['top5'] / total * 100
        top10_rate = results['top10'] / total * 100
        top15_rate = results['top15'] / total * 100
        top20_rate = results['top20'] / total * 100
        
        print(f"\nTop 5:  {results['top5']}/{total} = {top5_rate:.1f}%")
        print(f"Top 10: {results['top10']}/{total} = {top10_rate:.1f}%")
        print(f"Top 15: {results['top15']}/{total} = {top15_rate:.1f}%")
        print(f"Top 20: {results['top20']}/{total} = {top20_rate:.1f}% ⭐")
        
        # 对比随机
        random_top15 = 15 / 49 * 100
        random_top20 = 20 / 49 * 100
        
        print(f"\n对比随机:")
        print(f"Top 15: {top15_rate:.1f}% vs 随机{random_top15:.1f}% (提升{top15_rate/random_top15:.2f}x)")
        print(f"Top 20: {top20_rate:.1f}% vs 随机{random_top20:.1f}% (提升{top20_rate/random_top20:.2f}x)")
        
        if top20_rate > top15_rate:
            print(f"\n✅ Top 20相比Top 15提升: +{top20_rate - top15_rate:.1f}%")
        else:
            print(f"\n⚠️ Top 20未能提升")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    predict_top20_enhanced()
