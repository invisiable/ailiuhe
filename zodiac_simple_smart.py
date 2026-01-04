"""
生肖预测器 v10.0 - 简化智能选择器
核心策略：以v5.0 balanced（52%）为主，仅在明显不同场景时切换
"""

import pandas as pd
import numpy as np
from collections import Counter
from zodiac_super_predictor import ZodiacSuperPredictor

class ZodiacSimpleSmart:
    """简化版智能预测器 - 以v5.0为主，最小化动态切换"""
    
    def __init__(self):
        # 使用v5.0作为基础预测器
        self.base_predictor = ZodiacSuperPredictor()
        self.zodiacs = self.base_predictor.zodiacs
        self.zodiac_numbers = self.base_predictor.zodiac_numbers
    
    def _detect_scenario(self, animals, window=20):
        """
        检测当前场景类型
        返回: 'normal' 或特殊场景类型
        """
        if len(animals) < window:
            recent = animals
        else:
            recent = animals[-window:]
        
        counter = Counter(recent)
        
        # 计算关键指标
        diversity = len(counter) / 12
        hot_zodiacs = sum(1 for count in counter.values() if count >= 3)
        concentration = sum(count for zodiac, count in counter.items() if count >= 3) / len(recent) if counter else 0
        
        # 检测极端热门场景（需要hot_aware模型）
        if concentration > 0.50 and hot_zodiacs >= 4:
            return 'extreme_hot'
        
        # 检测极端冷门/分散场景（需要ultra_cold模型）  
        if diversity > 0.80 and concentration < 0.25:
            return 'extreme_cold'
        
        # 其他情况都用v5.0
        return 'normal'
    
    def predict_from_history(self, animals, top_n=5, debug=False):
        """
        智能预测
        """
        scenario = self._detect_scenario(animals)
        
        if scenario == 'normal':
            # 90%+的情况，使用v5.0权重
            weights = {
                'ultra_cold': 0.35, 'anti_hot': 0.20, 'gap': 0.18,
                'rotation': 0.12, 'absence_penalty': 0.08, 
                'diversity': 0.04, 'similarity': 0.03
            }
            result = self._predict_with_weights(animals, weights, top_n)
            selected_model = 'v5.0平衡模型'
        elif scenario == 'extreme_hot':
            # 极端热门场景，使用热门感知权重
            weights = {
                'ultra_cold': 0.20, 'anti_hot': 0.08, 'gap': 0.22,
                'rotation': 0.15, 'absence_penalty': 0.05, 
                'hot_momentum': 0.22, 'diversity': 0.05, 'similarity': 0.03
            }
            result = self._predict_with_weights(animals, weights, top_n)
            selected_model = '热门感知模型'
        else:  # extreme_cold
            # 极端冷门场景，使用极致冷门权重
            weights = {
                'ultra_cold': 0.45, 'anti_hot': 0.25, 'gap': 0.15,
                'rotation': 0.08, 'absence_penalty': 0.04, 
                'diversity': 0.02, 'similarity': 0.01
            }
            result = self._predict_with_weights(animals, weights, top_n)
            selected_model = '极致冷门模型'
        
        if debug:
            print(f"\n场景检测: {scenario}")
            print(f"选择模型: {selected_model}")
        
        result['selected_model'] = selected_model
        return result
    
    def _hot_momentum_strategy(self, animals):
        """热门惯性策略 - 从zodiac_smart_selector复制"""
        scores = {}
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        
        for zodiac in self.zodiacs:
            score = 0.0
            count_20 = recent_20.count(zodiac)
            if count_20 >= 4:
                score += 40.0
            elif count_20 == 3:
                score += 28.0
            elif count_20 == 2:
                score += 16.0
            elif count_20 == 1:
                score += 6.0
            
            count_30 = recent_30.count(zodiac)
            if count_30 >= 5:
                score += 12.0
            elif count_30 >= 3:
                score += 6.0
            scores[zodiac] = score
        return scores
    
    def _predict_with_weights(self, animals, weights, top_n):
        """使用指定权重进行预测"""
        # 收集各策略评分
        strategies_scores = {
            'ultra_cold': self.base_predictor._ultra_cold_strategy(animals),
            'anti_hot': self.base_predictor._anti_hot_strategy(animals),
            'gap': self.base_predictor._gap_analysis(animals),
            'rotation': self.base_predictor._rotation_advanced(animals),
            'absence_penalty': self.base_predictor._continuous_absence_penalty(animals),
            'diversity': self.base_predictor._diversity_boost(animals),
            'similarity': self.base_predictor._historical_similarity(animals)
        }
        
        if 'hot_momentum' in weights:
            strategies_scores['hot_momentum'] = self._hot_momentum_strategy(animals)
        
        # 加权融合
        final_scores = {}
        for zodiac in self.zodiacs:
            score = 0.0
            for strategy_name, strategy_scores in strategies_scores.items():
                weight = weights.get(strategy_name, 0)
                score += strategy_scores.get(zodiac, 0) * weight
            final_scores[zodiac] = score
        
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_zodiacs = [z for z, s in sorted_zodiacs[:top_n]]
        
        return {
            'top5': top_zodiacs,
            'all_scores': sorted_zodiacs
        }
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_n=5):
        """标准预测接口"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        result = self.predict_from_history(animals, top_n, debug=True)
        
        # 推荐号码
        recommended_numbers = []
        for rank, (zodiac, score) in enumerate(result['all_scores'][:top_n], 1):
            weight = top_n + 1 - rank
            for num in self.zodiac_numbers[zodiac]:
                recommended_numbers.append((num, weight))
        
        num_scores = {}
        for num, w in recommended_numbers:
            num_scores[num] = num_scores.get(num, 0) + w
        
        sorted_nums = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_nums[:15]]
        
        return {
            'model': '简化智能选择器',
            'version': '10.0',
            'selected_model': result['selected_model'],
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_animal': df.iloc[-1]['animal'],
            'top5': result['top5'],
            'top15_numbers': top_numbers
        }
    
    def get_recent_20_validation(self, csv_file='data/lucky_numbers.csv'):
        """获取最近20期验证结果（GUI兼容）"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        total = len(df)
        
        results = []
        for i in range(max(0, total - 20), total):
            animals = [str(a).strip() for a in df['animal'].values[:i]]
            prediction = self.predict_from_history(animals, top_n=5)
            
            actual = str(df['animal'].values[i]).strip()
            date = df.iloc[i]['date']
            
            hit_rank = 0
            if actual in prediction['top5']:
                hit_rank = prediction['top5'].index(actual) + 1
            
            results.append({
                'date': date,
                'actual': actual,
                'predicted_top5': prediction['top5'],
                'hit': hit_rank > 0,
                'rank': hit_rank,
                'model': prediction['selected_model']
            })
        
        return results

if __name__ == '__main__':
    predictor = ZodiacSimpleSmart()
    result = predictor.predict()
    
    print(f"\n模型: {result['model']}")
    print(f"版本: {result['version']}")
    print(f"选择的模型: {result['selected_model']}")
    print(f"总期数: {result['total_periods']}")
    print(f"最新期: {result['last_date']} - {result['last_animal']}")
    print(f"\n预测TOP5: {', '.join(result['top5'])}")
    print(f"推荐号码TOP15: {result['top15_numbers']}")
