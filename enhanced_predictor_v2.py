"""
高命中率综合预测系统 V2
使用多模型集成 + 扩展候选池
目标：Top 5 命中率50%+
"""
import numpy as np
import pandas as pd
from collections import Counter
from lucky_number_predictor import LuckyNumberPredictor
import warnings
warnings.filterwarnings('ignore')


class EnhancedPredictor:
    """增强预测器 - 多模型集成"""
    
    def __init__(self, predictors):
        """
        Args:
            predictors: 多个已训练的LuckyNumberPredictor实例列表
        """
        self.predictors = predictors
    
    def multi_model_predict(self, top_k=50):
        """多模型集成预测"""
        all_predictions = {}
        
        for predictor in self.predictors:
            try:
                preds = predictor.predict_top_probabilities(top_k=top_k)
                for pred in preds:
                    num = pred['number']
                    prob = pred['probability']
                    all_predictions[num] = all_predictions.get(num, [])
                    all_predictions[num].append(prob)
            except:
                continue
        
        # 计算平均概率
        avg_predictions = {}
        for num, probs in all_predictions.items():
            avg_predictions[num] = np.mean(probs)
        
        return avg_predictions
    
    def comprehensive_predict_v2(self, top_k=5):
        """V2版本：更激进的综合预测"""
        print(f"\n增强预测系统 V2 - 多模型集成")
        print("-"*70)
        
        # 1. 多模型预测
        print("1. 多模型集成预测...")
        model_probs = self.multi_model_predict(top_k=50)
        print(f"   集成了 {len(self.predictors)} 个模型，生成 {len(model_probs)} 个候选")
        
        # 使用第一个predictor的数据进行其他分析
        predictor = self.predictors[0]
        
        # 2. 五行强化
        print("2. 五行分布分析...")
        recent_numbers = predictor.raw_numbers[-30:]
        element_distribution = Counter()
        for num in recent_numbers:
            element = predictor.number_to_element.get(num)
            if element:
                element_distribution[element] += 1
        
        # 找出出现最少的五行（可能是下一个）
        all_elements = ['金', '木', '水', '火', '土']
        element_scores = {}
        for element in all_elements:
            count = element_distribution.get(element, 0)
            # 反向：出现少的得分高
            element_scores[element] = 1.0 / (count + 1)
        
        # 归一化
        total = sum(element_scores.values())
        element_scores = {k: v/total for k, v in element_scores.items()}
        
        element_probs = {}
        for element, score in element_scores.items():
            numbers = predictor.element_numbers.get(element, [])
            for num in numbers:
                element_probs[num] = element_probs.get(num, 0) + score / len(numbers)
        
        print(f"   五行得分: {element_scores}")
        print(f"   生成 {len(element_probs)} 个候选")
        
        # 3. 生肖关联
        print("3. 生肖历史关联...")
        animal_probs = {}
        if predictor.raw_animals is not None:
            last_animal = predictor.raw_animals[-1]
            # 预测下一个生肖
            next_animal = (last_animal + 1) % 12
            
            # 统计该生肖历史上出现的数字
            for i, (num, animal) in enumerate(zip(predictor.raw_numbers, predictor.raw_animals)):
                if animal == next_animal:
                    animal_probs[num] = animal_probs.get(num, 0) + 1
            
            total = sum(animal_probs.values())
            if total > 0:
                animal_probs = {k: v/total for k, v in animal_probs.items()}
        
        print(f"   生成 {len(animal_probs)} 个候选")
        
        # 4. 区间分析
        print("4. 数字区间分析...")
        recent_20 = predictor.raw_numbers[-20:]
        bins = [(1,10), (11,20), (21,30), (31,40), (41,49)]
        bin_counts = {b: 0 for b in bins}
        
        for num in recent_20:
            for bin_range in bins:
                if bin_range[0] <= num <= bin_range[1]:
                    bin_counts[bin_range] += 1
                    break
        
        # 找出出现少的区间
        min_count = min(bin_counts.values())
        under_represented_bins = [b for b, c in bin_counts.items() if c <= min_count + 2]
        
        bin_probs = {}
        for bin_range in under_represented_bins:
            for num in range(bin_range[0], bin_range[1] + 1):
                bin_probs[num] = 1.0 / len(under_represented_bins) / (bin_range[1] - bin_range[0] + 1)
        
        print(f"   关注区间: {under_represented_bins}")
        print(f"   生成 {len(bin_probs)} 个候选")
        
        # 5. 相邻数字
        print("5. 相邻数字规律...")
        last_num = predictor.raw_numbers[-1]
        neighbor_probs = {}
        for offset in range(-10, 11):
            candidate = last_num + offset
            if 1 <= candidate <= 49:
                # 距离越近概率越高
                prob = 1.0 / (abs(offset) + 1)
                neighbor_probs[candidate] = prob
        
        # 归一化
        total = sum(neighbor_probs.values())
        neighbor_probs = {k: v/total for k, v in neighbor_probs.items()}
        print(f"   基于最后数字 {last_num}，生成 {len(neighbor_probs)} 个候选")
        
        # 6. 冷热号分析
        print("6. 冷热号分析...")
        all_numbers = list(range(1, 50))
        recent_50 = set(predictor.raw_numbers[-50:])
        recent_30 = set(predictor.raw_numbers[-30:])
        recent_10 = set(predictor.raw_numbers[-10:])
        
        cold_hot_probs = {}
        for num in all_numbers:
            score = 0
            if num not in recent_10:
                score += 0.5  # 最近10期未出现
            if num not in recent_30:
                score += 0.3  # 最近30期未出现
            if num not in recent_50:
                score += 0.2  # 最近50期未出现
            cold_hot_probs[num] = score
        
        # 归一化
        total = sum(cold_hot_probs.values())
        if total > 0:
            cold_hot_probs = {k: v/total for k, v in cold_hot_probs.items()}
        
        print(f"   生成 {len(cold_hot_probs)} 个候选")
        
        # 综合所有方法
        print("7. 综合所有方法...")
        # 权重配置：模型35%, 五行25%, 生肖15%, 区间10%, 相邻10%, 冷热5%
        weights = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        
        all_numbers_set = set()
        all_numbers_set.update(model_probs.keys())
        all_numbers_set.update(element_probs.keys())
        all_numbers_set.update(animal_probs.keys())
        all_numbers_set.update(bin_probs.keys())
        all_numbers_set.update(neighbor_probs.keys())
        all_numbers_set.update(cold_hot_probs.keys())
        
        combined_probs = {}
        for num in all_numbers_set:
            score = 0
            score += model_probs.get(num, 0) * weights[0]
            score += element_probs.get(num, 0) * weights[1]
            score += animal_probs.get(num, 0) * weights[2]
            score += bin_probs.get(num, 0) * weights[3]
            score += neighbor_probs.get(num, 0) * weights[4]
            score += cold_hot_probs.get(num, 0) * weights[5]
            combined_probs[num] = score
        
        # 排序
        sorted_predictions = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for num, prob in sorted_predictions[:top_k]:
            element = predictor.number_to_element.get(num, '未知')
            animal = predictor.reverse_animal_mapping[(predictor.raw_animals[-1] + 1) % 12] if predictor.raw_animals is not None else '未知'
            
            results.append({
                'number': num,
                'animal': animal,
                'element': element,
                'probability': prob
            })
        
        print(f"   最终Top {top_k}预测完成\n")
        
        return results


def validate_enhanced_model(train_size=100, test_samples=20, top_k=5):
    """验证增强模型"""
    print("="*80)
    print(f"增强模型验证 - Top {top_k} 命中率测试")
    print("="*80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    hits = 0
    total = 0
    hit_details = []
    
    print(f"\n测试配置:")
    print(f"  训练集: {train_size} 期")
    print(f"  测试集: {test_samples} 期")
    print(f"  预测数量: Top {top_k}")
    print(f"  使用模型: 梯度提升 + LightGBM + XGBoost")
    
    print(f"\n{'期数':<8} {'实际':<8} {'Top {top_k} 预测':<50} {'命中'}")
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
            # 训练多个模型
            predictors = []
            for model_type in ['gradient_boosting', 'lightgbm', 'xgboost']:
                predictor = LuckyNumberPredictor()
                predictor.load_data(temp_file, number_column='number', date_column='date',
                                   animal_column='animal', element_column='element')
                predictor.train_model(model_type, test_size=0.2)
                predictors.append(predictor)
            
            # 增强预测
            enhanced = EnhancedPredictor(predictors)
            predictions = enhanced.comprehensive_predict_v2(top_k=top_k)
            
            actual_number = actual_row['number']
            predicted_numbers = [p['number'] for p in predictions]
            
            hit = actual_number in predicted_numbers
            if hit:
                hits += 1
                rank = predicted_numbers.index(actual_number) + 1
                hit_details.append((test_index, actual_number, rank))
            total += 1
            
            pred_str = ', '.join([f"{n}" for n in predicted_numbers])
            marker = "YES" if hit else "NO"
            
            print(f"{test_index:<8d} {actual_number:<8d} [{pred_str:<48}] {marker}")
            
        except Exception as e:
            print(f"{test_index:<8d} Error: {str(e)[:40]}")
            continue
    
    # 结果
    hit_rate = (hits / total * 100) if total > 0 else 0
    
    print("\n" + "="*80)
    print("验证结果")
    print("="*80)
    print(f"\n总测试数: {total}")
    print(f"命中数: {hits}")
    print(f"Top {top_k} 命中率: {hit_rate:.1f}%")
    
    if hit_details:
        print(f"\n命中详情:")
        for test_index, actual, rank in hit_details:
            print(f"  第{test_index}期: 实际{actual}, 排名第{rank}")
    
    if hit_rate >= 50:
        print(f"\n[SUCCESS] 达到目标! (>= 50%)")
    elif hit_rate >= 40:
        print(f"\n[GOOD] 接近目标 (40-50%)")
    elif hit_rate >= 30:
        print(f"\n[FAIR] 表现尚可 (30-40%)")
    else:
        print(f"\n[IMPROVE] 需要改进 (< 30%)")
    
    print("\n" + "="*80)
    
    return hit_rate


if __name__ == "__main__":
    hit_rate = validate_enhanced_model(train_size=100, test_samples=20, top_k=5)
