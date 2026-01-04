"""
极端值感知预测模型
策略：专门关注极端值（1-10, 40-49）的预测
"""
import sys
sys.path.insert(0, 'd:\\AIagent')

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class ExtremeValuePredictor:
    """极端值感知预测器"""
    
    def __init__(self):
        self.raw_numbers = []
        
    def load_data(self, file_path):
        """加载数据"""
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        self.raw_numbers = df['number'].values.tolist()
        
    def analyze_patterns(self, recent=30):
        """分析最近的模式"""
        recent_data = self.raw_numbers[-recent:]
        
        # 统计各区间出现频率
        extreme_low = [n for n in recent_data if 1 <= n <= 10]
        low = [n for n in recent_data if 11 <= n <= 20]
        mid = [n for n in recent_data if 21 <= n <= 30]
        high = [n for n in recent_data if 31 <= n <= 40]
        extreme_high = [n for n in recent_data if 41 <= n <= 49]
        
        pattern = {
            'extreme_low': len(extreme_low),
            'low': len(low),
            'mid': len(mid),
            'high': len(high),
            'extreme_high': len(extreme_high),
            'extreme_total': len(extreme_low) + len(extreme_high),
            'extreme_ratio': (len(extreme_low) + len(extreme_high)) / len(recent_data)
        }
        
        # 连续性分析
        last_5 = recent_data[-5:]
        has_extreme = any(n <= 10 or n >= 40 for n in last_5)
        avg_last_5 = np.mean(last_5)
        
        pattern['has_recent_extreme'] = has_extreme
        pattern['avg_last_5'] = avg_last_5
        pattern['last_num'] = recent_data[-1]
        
        return pattern
    
    def predict_extreme_aware(self, top_k=15):
        """极端值感知预测"""
        pattern = self.analyze_patterns()
        
        print(f"\n模式分析:")
        print(f"  极端值比例: {pattern['extreme_ratio']*100:.1f}%")
        print(f"  最近5期平均: {pattern['avg_last_5']:.1f}")
        print(f"  最后数字: {pattern['last_num']}")
        print(f"  最近有极端值: {'是' if pattern['has_recent_extreme'] else '否'}")
        
        candidates = []
        
        # 策略1: 基于极端值比例
        if pattern['extreme_ratio'] < 0.3:  # 极端值较少，可能要出现
            print("\n策略: 极端值较少，增加极端值候选")
            # 极端低值
            candidates.extend(range(1, 11))
            # 极端高值  
            candidates.extend(range(40, 50))
            weight_extreme = 3.0
        else:
            weight_extreme = 1.0
        
        # 策略2: 基于最近平均值
        if pattern['avg_last_5'] < 20:
            print("策略: 最近偏低，可能反弹")
            candidates.extend(range(20, 40))
            weight_mid_high = 2.0
        elif pattern['avg_last_5'] > 30:
            print("策略: 最近偏高，可能回落")
            candidates.extend(range(1, 25))
            weight_mid_low = 2.0
        else:
            candidates.extend(range(15, 35))
            weight_mid_high = weight_mid_low = 1.5
        
        # 策略3: 邻近数字
        last_num = pattern['last_num']
        for offset in range(-15, 16):
            num = last_num + offset
            if 1 <= num <= 49:
                candidates.append(num)
        
        # 策略4: 频率反向（选未出现的）
        recent_30 = self.raw_numbers[-30:]
        freq = Counter(recent_30)
        all_nums = set(range(1, 50))
        unused = all_nums - set(recent_30)
        
        if unused:
            print(f"策略: 加入{len(unused)}个最近未出现的数字")
            candidates.extend(unused)
        
        # 统计候选数字的权重
        num_weights = Counter(candidates)
        
        # 应用策略权重
        final_scores = {}
        for num, count in num_weights.items():
            score = count
            # 极端值加权
            if num <= 10 or num >= 40:
                score *= weight_extreme
            # 在未出现集合中加权
            if num in unused:
                score *= 2.0
            final_scores[num] = score
        
        # 排序
        sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for num, score in sorted_nums[:top_k]:
            results.append({
                'number': num,
                'score': score,
                'probability': score / sum(s for _, s in sorted_nums[:top_k])
            })
        
        return results


def validate_extreme_model(test_periods=10):
    """验证极端值模型"""
    print("=" * 80)
    print("极端值感知预测模型验证")
    print("=" * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_records = len(df)
    
    print(f"\n数据集: {total_records}期")
    print(f"验证: 最近{test_periods}期\n")
    
    top5_hits = 0
    top10_hits = 0
    top15_hits = 0
    
    for i in range(test_periods):
        test_index = total_records - test_periods + i
        period_num = test_index + 1
        
        train_df = df.iloc[:test_index]
        actual = df.iloc[test_index]['number']
        actual_date = df.iloc[test_index]['date']
        
        print(f"{'='*80}")
        print(f"测试第{period_num}期 ({actual_date}), 实际: {actual}")
        
        # 保存临时文件
        temp_file = f'data/temp_extreme_{i}.csv'
        train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        
        try:
            predictor = ExtremeValuePredictor()
            predictor.load_data(temp_file)
            
            predictions = predictor.predict_extreme_aware(top_k=15)
            top15 = [p['number'] for p in predictions]
            top10 = top15[:10]
            top5 = top15[:5]
            
            print(f"\nTop 5: {top5}")
            print(f"Top 10: {top10}")
            print(f"Top 15: {top15}")
            
            if actual in top5:
                rank = top5.index(actual) + 1
                print(f"✅ Top 5命中! (第{rank}名)")
                top5_hits += 1
                top10_hits += 1
                top15_hits += 1
            elif actual in top10:
                rank = top10.index(actual) + 1
                print(f"✓ Top 10命中 (第{rank}名)")
                top10_hits += 1
                top15_hits += 1
            elif actual in top15:
                rank = top15.index(actual) + 1
                print(f"○ Top 15命中 (第{rank}名)")
                top15_hits += 1
            else:
                print(f"❌ 未命中")
            
            # 清理
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    print(f"\nTop 5 命中: {top5_hits}/{test_periods} = {top5_hits/test_periods*100:.1f}%")
    print(f"Top 10 命中: {top10_hits}/{test_periods} = {top10_hits/test_periods*100:.1f}%")
    print(f"Top 15 命中: {top15_hits}/{test_periods} = {top15_hits/test_periods*100:.1f}%")
    
    return {
        'top5': top5_hits/test_periods*100,
        'top10': top10_hits/test_periods*100,
        'top15': top15_hits/test_periods*100
    }


if __name__ == "__main__":
    try:
        results = validate_extreme_model(10)
        print(f"\n✅ 完成！")
        print(f"Top 5: {results['top5']:.1f}%")
        print(f"Top 10: {results['top10']:.1f}%") 
        print(f"Top 15: {results['top15']:.1f}%")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
