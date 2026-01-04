"""
优化预测器 - 目标：Top 15 达到 60% 成功率

优化策略：
1. 极端值感知：检测并加强极端值候选
2. 动态权重：根据最近趋势调整各方法权重
3. 区域平衡：确保1-10, 11-20, 21-30, 31-40, 41-49都有覆盖
4. 历史模式：分析最近30期的数字分布模式
"""

import numpy as np
import pandas as pd
from collections import Counter
from lucky_number_predictor import LuckyNumberPredictor


class OptimizedPredictor:
    """优化的预测器 - 针对极端值和区域平衡"""
    
    def __init__(self, predictors):
        """
        Args:
            predictors: 已训练的预测器列表
        """
        self.predictors = predictors
        self.element_numbers = {
            '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
            '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
            '水': [13, 14, 21, 22, 29, 30, 43, 44],
            '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
            '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
        }
    
    def analyze_recent_pattern(self, numbers, window=30):
        """
        分析最近的数字分布模式
        
        Returns:
            dict: 包含各种统计信息
        """
        recent = numbers[-window:]
        
        # 区间分布
        zones = {
            'extreme_small': [n for n in recent if 1 <= n <= 10],
            'small': [n for n in recent if 11 <= n <= 20],
            'mid': [n for n in recent if 21 <= n <= 30],
            'large': [n for n in recent if 31 <= n <= 40],
            'extreme_large': [n for n in recent if 41 <= n <= 49]
        }
        
        zone_ratios = {k: len(v)/len(recent) for k, v in zones.items()}
        
        # 检测极端值趋势
        extreme_count = len(zones['extreme_small']) + len(zones['extreme_large'])
        extreme_ratio = extreme_count / len(recent)
        
        # 最近10期的极端值比例
        recent_10 = numbers[-10:]
        extreme_10 = sum(1 for n in recent_10 if n <= 10 or n >= 40)
        extreme_10_ratio = extreme_10 / 10
        
        return {
            'zones': zones,
            'zone_ratios': zone_ratios,
            'extreme_ratio': extreme_ratio,
            'extreme_10_ratio': extreme_10_ratio,
            'has_extreme_trend': extreme_10_ratio > 0.4,
            'recent': recent
        }
    
    def get_model_predictions(self, top_k=30):
        """获取多模型预测结果"""
        all_predictions = {}
        
        for predictor in self.predictors:
            try:
                preds = predictor.predict_top_probabilities(top_k=top_k)
                for pred in preds:
                    num = pred['number']
                    prob = pred['probability']
                    if num not in all_predictions:
                        all_predictions[num] = []
                    all_predictions[num].append(prob)
            except:
                continue
        
        # 计算平均概率
        avg_predictions = {}
        for num, probs in all_predictions.items():
            avg_predictions[num] = np.mean(probs)
        
        return avg_predictions
    
    def get_extreme_candidates(self, pattern, existing_nums, k=6):
        """
        获取极端值候选（重点优化）
        
        策略：
        1. 如果最近10期极端值>40%，增加极端值权重
        2. 选择最近30期频率最高的极端值
        3. 避免最近3期重复
        """
        recent_3 = set(pattern['recent'][-3:])
        recent_30 = pattern['recent']
        
        # 极小值候选
        small_candidates = [n for n in range(1, 11) 
                           if n not in recent_3 and n not in existing_nums]
        # 极大值候选
        large_candidates = [n for n in range(40, 50) 
                           if n not in recent_3 and n not in existing_nums]
        
        # 基于频率排序（转换为list处理）
        recent_30_list = recent_30.tolist() if isinstance(recent_30, np.ndarray) else list(recent_30)
        small_freq = {n: recent_30_list.count(n) for n in small_candidates}
        large_freq = {n: recent_30_list.count(n) for n in large_candidates}
        
        # 根据趋势决定分配
        if pattern['has_extreme_trend']:
            # 极端值趋势明显，增加极端值数量
            small_k = k // 2 + 1
            large_k = k - small_k
        else:
            # 正常分配
            small_k = k // 2
            large_k = k - small_k
        
        # 选择频率高的或者近期未出现的
        selected_small = sorted(small_freq.keys(), 
                               key=lambda x: (small_freq[x], -recent_30_list[::-1].index(x) if x in recent_30_list else 100),
                               reverse=True)[:small_k]
        selected_large = sorted(large_freq.keys(),
                               key=lambda x: (large_freq[x], -recent_30_list[::-1].index(x) if x in recent_30_list else 100),
                               reverse=True)[:large_k]
        
        return list(selected_small) + list(selected_large)
    
    def get_zone_balanced_candidates(self, pattern, existing_nums, total_needed=15):
        """
        获取区域平衡的候选
        
        确保每个区域都有覆盖：
        - 极小(1-10): 至少2个
        - 小(11-20): 至少2个
        - 中(21-30): 至少3个
        - 大(31-40): 至少2个
        - 极大(41-49): 至少1个
        """
        existing = set(existing_nums)
        recent_5 = set(pattern['recent'][-5:].tolist() if isinstance(pattern['recent'], np.ndarray) else pattern['recent'][-5:])
        recent_30 = pattern['recent'].tolist() if isinstance(pattern['recent'], np.ndarray) else list(pattern['recent'])
        
        # 定义每个区域的目标数量
        zone_targets = {
            'extreme_small': (1, 10, 2),
            'small': (11, 20, 2),
            'mid': (21, 30, 3),
            'large': (31, 40, 2),
            'extreme_large': (41, 49, 1)
        }
        
        zone_candidates = {}
        
        for zone_name, (start, end, target) in zone_targets.items():
            # 统计已有的该区域数字
            zone_existing = [n for n in existing if start <= n <= end]
            need = max(0, target - len(zone_existing))
            
            if need > 0:
                # 获取该区域候选
                candidates = [n for n in range(start, end + 1)
                            if n not in existing and n not in recent_5]
                
                # 基于频率排序
                freq = {n: recent_30.count(n) for n in candidates}
                selected = sorted(candidates, 
                                key=lambda x: freq.get(x, 0), 
                                reverse=True)[:need]
                zone_candidates[zone_name] = selected
        
        # 合并所有区域候选
        all_zone_candidates = []
        for candidates in zone_candidates.values():
            all_zone_candidates.extend(candidates)
        
        return all_zone_candidates
    
    def predict_optimized(self, top_k=15):
        """
        优化的预测方法
        
        流程：
        1. 获取基础模型预测（Top 10）
        2. 分析最近数字模式
        3. 根据模式动态调整权重
        4. 添加极端值候选
        5. 添加区域平衡候选
        6. 综合评分排序
        """
        print(f"\n[优化预测系统] - 目标Top {top_k}")
        print("-" * 70)
        
        # 获取历史数据
        if not self.predictors or not hasattr(self.predictors[0], 'raw_numbers'):
            raise Exception("预测器未加载数据")
        
        numbers = self.predictors[0].raw_numbers
        animals = self.predictors[0].raw_animals
        elements = self.predictors[0].raw_elements
        
        # 1. 分析模式
        print("1. 分析最近数字模式...")
        pattern = self.analyze_recent_pattern(numbers)
        print(f"   极端值比例(最近30期): {pattern['extreme_ratio']*100:.1f}%")
        print(f"   极端值比例(最近10期): {pattern['extreme_10_ratio']*100:.1f}%")
        print(f"   极端值趋势: {'[HIGH]' if pattern['has_extreme_trend'] else '[NORMAL]'}")
        
        # 2. 获取基础模型预测
        print("2. 多模型集成预测...")
        model_preds = self.get_model_predictions(top_k=30)
        print(f"   生成 {len(model_preds)} 个候选")
        
        # 3. 动态权重调整
        print("3. 动态权重调整...")
        if pattern['has_extreme_trend']:
            # 极端值趋势，大幅降低模型权重，增加极端值方法权重
            weights = {
                'model': 0.15,      # 大幅降低
                'extreme': 0.40,    # 大幅增加
                'zone': 0.25,       # 增加
                'element': 0.10,
                'frequency': 0.10
            }
            print(f"   检测到极端值趋势，调整权重: model=0.15, extreme=0.40")
        else:
            weights = {
                'model': 0.35,
                'extreme': 0.15,
                'zone': 0.15,
                'element': 0.20,
                'frequency': 0.15
            }
            print(f"   正常模式，标准权重: model=0.35")
        
        # 4. 综合评分
        print("4. 综合评分...")
        candidates_scores = {}
        
        # 4.1 模型预测分数
        max_model_score = max(model_preds.values()) if model_preds else 1
        for num, score in model_preds.items():
            candidates_scores[num] = candidates_scores.get(num, 0) + \
                                   (score / max_model_score) * weights['model']
        
        # 4.2 极端值候选（增加数量）
        base_nums = list(model_preds.keys())[:8]  # 减少基础模型数量
        extreme_candidates = self.get_extreme_candidates(pattern, base_nums, k=10)  # 增加极端值候选
        print(f"   极端值候选: {extreme_candidates}")
        for num in extreme_candidates:
            candidates_scores[num] = candidates_scores.get(num, 0) + weights['extreme']
        
        # 4.3 区域平衡
        zone_candidates = self.get_zone_balanced_candidates(pattern, base_nums, total_needed=15)
        print(f"   区域平衡候选: {zone_candidates}")
        for num in zone_candidates:
            candidates_scores[num] = candidates_scores.get(num, 0) + weights['zone']
        
        # 4.4 五行分析
        element_counter = Counter(elements[-20:]) if elements is not None else {}
        if element_counter:
            top_element = element_counter.most_common(1)[0][0]
            reverse_element = {v: k for k, v in self.predictors[0].element_mapping.items()}
            top_element_name = reverse_element.get(top_element, '')
            if top_element_name in self.element_numbers:
                for num in self.element_numbers[top_element_name]:
                    if 1 <= num <= 49:
                        candidates_scores[num] = candidates_scores.get(num, 0) + weights['element'] * 0.5
        
        # 4.5 频率分析
        recent_30 = numbers[-30:].tolist() if isinstance(numbers, np.ndarray) else list(numbers[-30:])
        freq_counter = Counter(recent_30)
        max_freq = max(freq_counter.values()) if freq_counter else 1
        for num in range(1, 50):
            freq = freq_counter.get(num, 0)
            candidates_scores[num] = candidates_scores.get(num, 0) + \
                                   (freq / max_freq) * weights['frequency']
        
        # 5. 排序并返回Top K
        print("5. 综合排序...")
        sorted_candidates = sorted(candidates_scores.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)
        
        # 构造返回结果
        results = []
        for num, score in sorted_candidates[:top_k]:
            # 获取生肖和五行
            animal_name = '未知'
            element_name = '未知'
            
            if animals is not None:
                animal_code = (num - 1) % 12
                reverse_animal = {v: k for k, v in self.predictors[0].animal_mapping.items()}
                animal_name = reverse_animal.get(animal_code, '未知')
            
            # 根据五行映射获取
            for elem, nums in self.element_numbers.items():
                if num in nums:
                    element_name = elem
                    break
            
            results.append({
                'number': num,
                'animal': animal_name,
                'element': element_name,
                'probability': score
            })
        
        print(f"   Top {top_k} 预测完成: {[r['number'] for r in results]}")
        
        return results


def test_optimized_predictor():
    """测试优化预测器"""
    print("=" * 80)
    print("优化预测器测试 - 目标 Top 15 达到 60%")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"\n[数据集信息]")
    print(f"   总期数: {len(df)}")
    print(f"   测试范围: 最近10期")
    
    # 训练模型
    print(f"\n[训练模型]")
    model_types = ['gradient_boosting', 'lightgbm', 'xgboost']
    predictors = []
    
    for i, model_type in enumerate(model_types, 1):
        print(f"   [{i}/3] {model_type}...", end='', flush=True)
        predictor = LuckyNumberPredictor()
        
        # 使用前面的数据训练
        train_df = df.iloc[:-10]
        train_file = f'temp_train_{model_type}.csv'
        train_df.to_csv(train_file, index=False)
        
        predictor.load_data(train_file,
                          number_column='number',
                          date_column='date',
                          animal_column='animal',
                          element_column='element')
        
        # 处理NaN
        if np.any(np.isnan(predictor.X)):
            predictor.X = np.nan_to_num(predictor.X, nan=0.0)
        if np.any(np.isnan(predictor.y)):
            predictor.y = np.nan_to_num(predictor.y, nan=0.0)
        
        predictor.train_model(model_type=model_type)
        predictors.append(predictor)
        print(" [OK]")
    
    # 创建优化预测器
    optimized = OptimizedPredictor(predictors)
    
    # 测试最近10期
    print(f"\n" + "=" * 80)
    print("测试最近10期")
    print("=" * 80)
    
    results = {
        'top5': 0, 'top10': 0, 'top15': 0, 'top20': 0,
        'details': []
    }
    
    total_periods = len(df)
    
    for idx in range(total_periods - 10, total_periods):
        period_num = idx + 1
        actual_number = df.iloc[idx]['number']
        actual_date = df.iloc[idx]['date']
        
        # 使用之前的数据预测
        test_df = df.iloc[:idx]
        test_file = f'temp_test_{idx}.csv'
        test_df.to_csv(test_file, index=False)
        
        print(f"\n第{period_num}期 ({actual_date}): 实际 = {actual_number}")
        
        try:
            # 重新加载数据
            for predictor in predictors:
                predictor.load_data(test_file,
                                  number_column='number',
                                  date_column='date',
                                  animal_column='animal',
                                  element_column='element')
                if np.any(np.isnan(predictor.X)):
                    predictor.X = np.nan_to_num(predictor.X, nan=0.0)
                if np.any(np.isnan(predictor.y)):
                    predictor.y = np.nan_to_num(predictor.y, nan=0.0)
            
            # 预测
            predictions = optimized.predict_optimized(top_k=20)
            predicted_numbers = [p['number'] for p in predictions]
            
            # 检查命中
            if actual_number in predicted_numbers:
                rank = predicted_numbers.index(actual_number) + 1
                
                if rank <= 5:
                    level = "[*] Top 5"
                    results['top5'] += 1
                    results['top10'] += 1
                    results['top15'] += 1
                    results['top20'] += 1
                elif rank <= 10:
                    level = "[v] Top 10"
                    results['top10'] += 1
                    results['top15'] += 1
                    results['top20'] += 1
                elif rank <= 15:
                    level = "[o] Top 15"
                    results['top15'] += 1
                    results['top20'] += 1
                else:
                    level = "[+] Top 20"
                    results['top20'] += 1
                
                print(f"   [HIT] 命中! 排名: {rank} {level}")
            else:
                print(f"   [MISS] 未命中")
            
            print(f"   预测Top15: {predicted_numbers[:15]}")
            
            results['details'].append({
                'period': period_num,
                'actual': actual_number,
                'hit': actual_number in predicted_numbers[:15],
                'rank': predicted_numbers.index(actual_number) + 1 if actual_number in predicted_numbers else -1
            })
            
        except Exception as e:
            print(f"   [ERROR] 预测失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 统计结果
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    
    total = len(results['details'])
    
    if total > 0:
        print(f"\n命中统计 (最近{total}期):")
        print(f"   Top 5:  {results['top5']}/{total} = {results['top5']/total*100:.1f}%")
        print(f"   Top 10: {results['top10']}/{total} = {results['top10']/total*100:.1f}%")
        print(f"   Top 15: {results['top15']}/{total} = {results['top15']/total*100:.1f}%")
        print(f"   Top 20: {results['top20']}/{total} = {results['top20']/total*100:.1f}%")
        
        # 对比
        top15_rate = results['top15']/total*100
        random_15 = 15/49*100
        improvement = top15_rate / random_15
        
        print(f"\nTop 15 性能:")
        print(f"   实际: {top15_rate:.1f}%")
        print(f"   随机: {random_15:.1f}%")
        print(f"   提升: {improvement:.2f}x")
        
        if top15_rate >= 60:
            print(f"   [SUCCESS] 达到60%目标！")
        elif top15_rate >= 50:
            print(f"   [GOOD] 超过50%，接近目标")
        else:
            print(f"   [WARN] 未达目标，需进一步优化")
    
    # 清理
    print(f"\n[清理临时文件]")
    import os
    for f in os.listdir('.'):
        if f.startswith('temp_'):
            os.remove(f)
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    test_optimized_predictor()
