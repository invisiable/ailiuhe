"""
综合推测系统
结合数字模型预测、生肖周期推演、五行规律推演
目标：Top 5 命中率达到50%
"""
import numpy as np
import pandas as pd
from collections import Counter
from lucky_number_predictor import LuckyNumberPredictor


class ComprehensivePredictor:
    """综合预测器：融合多种预测方法"""
    
    def __init__(self, predictor):
        """
        Args:
            predictor: 已训练的LuckyNumberPredictor实例
        """
        self.predictor = predictor
        
    def predict_by_number_model(self, top_k=30):
        """
        方法1：基于数字模型的预测
        使用机器学习模型直接预测数字
        """
        predictions = self.predictor.predict_top_probabilities(top_k=top_k)
        
        # 提取数字和概率
        number_probs = {}
        for pred in predictions:
            number_probs[pred['number']] = pred['probability']
        
        return number_probs
    
    def predict_by_animal_cycle(self):
        """
        方法2：基于生肖周期推演
        分析历史数据中每个生肖对应的数字分布
        """
        raw_numbers = self.predictor.raw_numbers
        raw_animals = self.predictor.raw_animals
        
        if raw_animals is None:
            return {}
        
        # 预测下一个生肖
        last_animal = raw_animals[-1]
        next_animal = (last_animal + 1) % 12
        next_animal_name = self.predictor.reverse_animal_mapping[next_animal]
        
        # 统计该生肖历史上对应的数字
        animal_number_map = {}
        for num, animal in zip(raw_numbers, raw_animals):
            if animal == next_animal:
                animal_number_map[num] = animal_number_map.get(num, 0) + 1
        
        # 转换为概率
        total = sum(animal_number_map.values())
        if total == 0:
            return {}
        
        animal_probs = {num: count/total for num, count in animal_number_map.items()}
        
        return animal_probs
    
    def predict_by_element_cycle(self):
        """
        方法3：基于五行推演
        分析五行对应的数字规律
        """
        # 根据最近的数字趋势预测五行
        recent_numbers = self.predictor.raw_numbers[-20:]
        
        # 统计最近五行分布
        element_counter = Counter()
        for num in recent_numbers:
            element = self.predictor.number_to_element.get(num)
            if element:
                element_counter[element] += 1
        
        # 预测最可能的3个五行
        top_elements = element_counter.most_common(3)
        
        if not top_elements:
            # 如果没有数据，使用五行循环
            raw_elements = self.predictor.raw_elements
            if raw_elements is not None:
                last_element = raw_elements[-1]
                next_element = (last_element + 1) % 5
                next_element_name = self.predictor.reverse_element_mapping[next_element]
                top_elements = [(next_element_name, 1)]
        
        # 根据五行获取对应的数字
        element_probs = {}
        for element, count in top_elements:
            numbers = self.predictor.element_numbers.get(element, [])
            weight = count / sum(c for _, c in top_elements)
            for num in numbers:
                element_probs[num] = element_probs.get(num, 0) + weight / len(numbers)
        
        return element_probs
    
    def predict_by_frequency(self):
        """
        方法4：基于历史频率
        分析最近出现频率高的数字
        """
        recent_numbers = self.predictor.raw_numbers[-30:]
        number_counter = Counter(recent_numbers)
        
        # 转换为概率
        total = len(recent_numbers)
        freq_probs = {num: count/total for num, count in number_counter.items()}
        
        return freq_probs
    
    def predict_by_trend(self):
        """
        方法5：基于趋势分析
        分析数字变化趋势
        """
        recent_numbers = self.predictor.raw_numbers[-10:]
        
        # 计算平均变化
        changes = np.diff(recent_numbers)
        avg_change = np.mean(changes)
        std_change = np.std(changes)
        
        # 预测可能的范围
        last_number = recent_numbers[-1]
        
        # 生成候选数字（基于趋势）
        trend_probs = {}
        for offset in range(-15, 16):
            predicted = int(last_number + avg_change + offset)
            if 1 <= predicted <= 49:
                # 基于距离的概率（越接近预测值概率越高）
                distance = abs(offset)
                prob = np.exp(-distance / 5)
                trend_probs[predicted] = prob
        
        # 归一化
        total = sum(trend_probs.values())
        if total > 0:
            trend_probs = {k: v/total for k, v in trend_probs.items()}
        
        return trend_probs
    
    def predict_by_gap_analysis(self):
        """
        方法6：基于数字间隔分析
        分析哪些数字很久没出现了
        """
        all_numbers = set(range(1, 50))
        recent_numbers = set(self.predictor.raw_numbers[-20:])
        
        # 找出最近20期未出现的数字
        missing_numbers = all_numbers - recent_numbers
        
        # 统计每个数字距离上次出现的期数
        last_appearance = {}
        for num in all_numbers:
            for i in range(len(self.predictor.raw_numbers)-1, -1, -1):
                if self.predictor.raw_numbers[i] == num:
                    last_appearance[num] = len(self.predictor.raw_numbers) - i
                    break
            if num not in last_appearance:
                last_appearance[num] = len(self.predictor.raw_numbers)
        
        # 间隔越久，概率越高
        gap_probs = {}
        max_gap = max(last_appearance.values())
        for num, gap in last_appearance.items():
            gap_probs[num] = gap / max_gap
        
        # 归一化
        total = sum(gap_probs.values())
        if total > 0:
            gap_probs = {k: v/total for k, v in gap_probs.items()}
        
        return gap_probs
    
    def comprehensive_predict(self, top_k=5, weights=None):
        """
        综合预测：融合所有方法
        
        Args:
            top_k: 返回Top K个预测
            weights: 各方法权重 [模型, 生肖, 五行, 频率, 趋势, 间隔]
        
        Returns:
            Top K 预测结果
        """
        if weights is None:
            # 默认权重：模型40%, 生肖15%, 五行20%, 频率10%, 趋势10%, 间隔5%
            weights = [0.40, 0.15, 0.20, 0.10, 0.10, 0.05]
        
        print("\n" + "="*70)
        print("综合预测系统 - 多方法融合")
        print("="*70)
        
        # 执行各方法预测
        print("\n1. 数字模型预测...")
        model_probs = self.predict_by_number_model(top_k=30)
        print(f"   生成 {len(model_probs)} 个候选数字")
        
        print("\n2. 生肖周期推演...")
        animal_probs = self.predict_by_animal_cycle()
        print(f"   生成 {len(animal_probs)} 个候选数字")
        
        print("\n3. 五行规律推演...")
        element_probs = self.predict_by_element_cycle()
        print(f"   生成 {len(element_probs)} 个候选数字")
        
        print("\n4. 历史频率分析...")
        freq_probs = self.predict_by_frequency()
        print(f"   生成 {len(freq_probs)} 个候选数字")
        
        print("\n5. 趋势分析...")
        trend_probs = self.predict_by_trend()
        print(f"   生成 {len(trend_probs)} 个候选数字")
        
        print("\n6. 间隔分析...")
        gap_probs = self.predict_by_gap_analysis()
        print(f"   生成 {len(gap_probs)} 个候选数字")
        
        # 融合所有概率
        print("\n7. 融合所有方法...")
        all_numbers = set()
        all_numbers.update(model_probs.keys())
        all_numbers.update(animal_probs.keys())
        all_numbers.update(element_probs.keys())
        all_numbers.update(freq_probs.keys())
        all_numbers.update(trend_probs.keys())
        all_numbers.update(gap_probs.keys())
        
        combined_probs = {}
        for num in all_numbers:
            score = 0
            score += model_probs.get(num, 0) * weights[0]
            score += animal_probs.get(num, 0) * weights[1]
            score += element_probs.get(num, 0) * weights[2]
            score += freq_probs.get(num, 0) * weights[3]
            score += trend_probs.get(num, 0) * weights[4]
            score += gap_probs.get(num, 0) * weights[5]
            combined_probs[num] = score
        
        # 排序并返回Top K
        sorted_predictions = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for num, prob in sorted_predictions[:top_k]:
            # 获取对应的生肖和五行
            animal = self.predictor.reverse_animal_mapping[(self.predictor.raw_animals[-1] + 1) % 12] if self.predictor.raw_animals is not None else '未知'
            element = self.predictor.number_to_element.get(num, '未知')
            
            results.append({
                'number': num,
                'animal': animal,
                'element': element,
                'probability': prob,
                'model_score': model_probs.get(num, 0) * weights[0],
                'animal_score': animal_probs.get(num, 0) * weights[1],
                'element_score': element_probs.get(num, 0) * weights[2],
                'freq_score': freq_probs.get(num, 0) * weights[3],
                'trend_score': trend_probs.get(num, 0) * weights[4],
                'gap_score': gap_probs.get(num, 0) * weights[5]
            })
        
        print(f"   最终生成 Top {top_k} 预测\n")
        
        return results


def validate_comprehensive_model(train_size=100, test_samples=20, top_k=5):
    """验证综合模型的准确率"""
    print("="*80)
    print(f"综合模型验证 - Top {top_k} 命中率测试")
    print("="*80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    hits = 0
    total = 0
    
    print(f"\n测试配置:")
    print(f"  训练集大小: {train_size}")
    print(f"  测试样本数: {test_samples}")
    print(f"  预测数量: Top {top_k}")
    
    print(f"\n{'期数':<8} {'实际':<8} {'Top {top_k} 预测':<40} {'命中'}")
    print("-"*80)
    
    for i in range(test_samples):
        test_index = train_size + i
        if test_index >= len(df):
            break
        
        train_df = df.iloc[:test_index].copy()
        actual_row = df.iloc[test_index]
        
        temp_file = 'data/temp_train.csv'
        train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        
        try:
            # 训练模型
            predictor = LuckyNumberPredictor()
            predictor.load_data(temp_file, number_column='number', date_column='date',
                               animal_column='animal', element_column='element')
            predictor.train_model('gradient_boosting', test_size=0.2)
            
            # 综合预测
            comp_predictor = ComprehensivePredictor(predictor)
            predictions = comp_predictor.comprehensive_predict(top_k=top_k)
            
            actual_number = actual_row['number']
            predicted_numbers = [p['number'] for p in predictions]
            
            hit = actual_number in predicted_numbers
            if hit:
                hits += 1
            total += 1
            
            pred_str = ', '.join([str(n) for n in predicted_numbers])
            marker = "YES" if hit else "NO"
            
            print(f"{test_index:<8d} {actual_number:<8d} [{pred_str:<38}] {marker}")
            
        except Exception as e:
            print(f"{test_index:<8d} 预测失败: {e}")
            continue
    
    # 结果统计
    hit_rate = (hits / total * 100) if total > 0 else 0
    
    print("\n" + "="*80)
    print("验证结果")
    print("="*80)
    print(f"\n总测试数: {total}")
    print(f"命中数: {hits}")
    print(f"Top {top_k} 命中率: {hit_rate:.1f}%")
    
    if hit_rate >= 50:
        print(f"\n[SUCCESS] Goal achieved (>=50%)")
    elif hit_rate >= 40:
        print(f"\n[GOOD] Close to goal (40-50%)")
    else:
        print(f"\n[IMPROVE] Need improvement (<40%)")
    
    print("\n" + "="*80)
    
    return hit_rate


if __name__ == "__main__":
    # 先测试单次预测
    print("\n测试综合预测系统...\n")
    
    predictor = LuckyNumberPredictor()
    predictor.load_data('data/lucky_numbers.csv', number_column='number', 
                       date_column='date', animal_column='animal', element_column='element')
    predictor.train_model('gradient_boosting', test_size=0.2)
    
    comp_predictor = ComprehensivePredictor(predictor)
    predictions = comp_predictor.comprehensive_predict(top_k=5)
    
    print("="*70)
    print("综合预测结果 - Top 5")
    print("="*70)
    print(f"\n{'排名':<6} {'数字':<6} {'生肖':<6} {'五行':<6} {'综合得分':<10} {'得分详情'}")
    print("-"*70)
    
    for i, pred in enumerate(predictions, 1):
        print(f"{i:<6} {pred['number']:<6} {pred['animal']:<6} {pred['element']:<6} "
              f"{pred['probability']:<10.4f} "
              f"M:{pred['model_score']:.3f} A:{pred['animal_score']:.3f} "
              f"E:{pred['element_score']:.3f} F:{pred['freq_score']:.3f} "
              f"T:{pred['trend_score']:.3f} G:{pred['gap_score']:.3f}")
    
    # 验证准确率
    print("\n\n开始历史数据验证...\n")
    hit_rate = validate_comprehensive_model(train_size=100, test_samples=20, top_k=5)
