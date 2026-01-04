"""
生肖预测器 v8.0 - 多模型集成版
集成冷门策略 + 优化热门惯性（20期判断）
"""

import pandas as pd
import numpy as np
from collections import Counter

class ZodiacEnsemblePredictor:
    """生肖集成预测器 - 多策略融合"""
    
    def __init__(self):
        self.zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        self.zodiac_numbers = {
            '鼠': [1, 13, 25, 37, 49], '牛': [2, 14, 26, 38],
            '虎': [3, 15, 27, 39], '兔': [4, 16, 28, 40],
            '龙': [5, 17, 29, 41], '蛇': [6, 18, 30, 42],
            '马': [7, 19, 31, 43], '羊': [8, 20, 32, 44],
            '猴': [9, 21, 33, 45], '鸡': [10, 22, 34, 46],
            '狗': [11, 23, 35, 47], '猪': [12, 24, 36, 48]
        }
        
        # 模型A: 纯冷门策略（v5.0）
        self.model_a_weights = {
            'ultra_cold': 0.35,
            'anti_hot': 0.20,
            'gap': 0.18,
            'rotation': 0.12,
            'absence_penalty': 0.08,
            'diversity': 0.04,
            'similarity': 0.03
        }
        
        # 模型B: 冷门+惯性平衡（优化版）
        self.model_b_weights = {
            'ultra_cold': 0.25,           # 进一步降低冷门权重
            'anti_hot': 0.10,             # 大幅降低反热门权重
            'gap': 0.20,                  # 提升间隔
            'rotation': 0.12,             # 保持轮转
            'absence_penalty': 0.06,      # 降低惩罚
            'hot_momentum_20': 0.20,      # 大幅增强！20期热门惯性
            'diversity': 0.04,            # 保持多样性
            'similarity': 0.03            # 保持相似性
        }
        
        # 模型C: 间隔主导型
        self.model_c_weights = {
            'ultra_cold': 0.25,
            'anti_hot': 0.15,
            'gap': 0.28,                  # 强化间隔分析
            'rotation': 0.18,             # 强化轮转
            'absence_penalty': 0.06,
            'diversity': 0.05,
            'similarity': 0.03
        }
        
        # 集成权重（动态调整 - 提升惯性模型权重）
        self.ensemble_weights = {
            'model_a': 0.40,  # 降低冷门模型权重
            'model_b': 0.45,  # 大幅提升惯性模型权重
            'model_c': 0.15   # 保持间隔模型权重
        }
    
    def _ultra_cold_strategy(self, animals):
        """极致冷门策略"""
        scores = {}
        counter = Counter(animals[-50:] if len(animals) >= 50 else animals)
        
        min_count = min(counter.values()) if counter else 0
        max_count = max(counter.values()) if counter else 1
        
        for zodiac in self.zodiacs:
            count = counter.get(zodiac, 0)
            if max_count > min_count:
                normalized = (count - min_count) / (max_count - min_count)
                scores[zodiac] = (1 - normalized) * 100
            else:
                scores[zodiac] = 50.0
        
        return scores
    
    def _anti_hot_strategy(self, animals):
        """反热门策略"""
        scores = {}
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        
        for zodiac in self.zodiacs:
            count = recent_10.count(zodiac)
            if count == 0:
                scores[zodiac] = 20.0
            elif count == 1:
                scores[zodiac] = 10.0
            else:
                scores[zodiac] = -5.0 * count
        
        return scores
    
    def _hot_momentum_20_strategy(self, animals):
        """
        热门惯性策略（20期优化版）
        以最近20期为主要判断依据，结合30期趋势
        """
        scores = {}
        
        # 主要窗口：最近20期
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        # 辅助窗口：最近30期
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        
        for zodiac in self.zodiacs:
            score = 0.0
            
            # 最近20期是核心（权重70%）
            count_20 = recent_20.count(zodiac)
            if count_20 >= 4:
                score += 35.0  # 强热门（20期出现4+次）
            elif count_20 == 3:
                score += 25.0  # 中热门（20期出现3次）
            elif count_20 == 2:
                score += 15.0  # 轻热门（20期出现2次）
            elif count_20 == 1:
                score += 5.0   # 出现过
            
            # 最近30期作为趋势补充（权重30%）
            count_30 = recent_30.count(zodiac)
            if count_30 >= 5:
                score += 15.0
            elif count_30 >= 3:
                score += 8.0
            
            scores[zodiac] = score
        
        return scores
    
    def _gap_analysis(self, animals):
        """间隔分析"""
        scores = {}
        
        for zodiac in self.zodiacs:
            try:
                last_idx = len(animals) - 1 - animals[::-1].index(zodiac)
                gap = len(animals) - last_idx - 1
                
                if gap >= 15:
                    scores[zodiac] = 10.0
                elif gap >= 10:
                    scores[zodiac] = 7.0
                elif gap >= 6:
                    scores[zodiac] = 3.0
                elif gap >= 3:
                    scores[zodiac] = 1.0
                else:
                    scores[zodiac] = -3.0
            except ValueError:
                scores[zodiac] = 5.0
        
        return scores
    
    def _rotation_advanced(self, animals):
        """轮转规律"""
        scores = {}
        
        for zodiac in self.zodiacs:
            try:
                indices = [i for i, a in enumerate(animals) if a == zodiac]
                if len(indices) >= 2:
                    gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
                    avg_gap = sum(gaps) / len(gaps)
                    current_gap = len(animals) - indices[-1]
                    
                    if abs(current_gap - avg_gap) <= 2:
                        scores[zodiac] = 10.0
                    elif abs(current_gap - avg_gap) <= 5:
                        scores[zodiac] = 5.0
                    else:
                        scores[zodiac] = 0.0
                else:
                    scores[zodiac] = 0.0
            except:
                scores[zodiac] = 0.0
        
        return scores
    
    def _continuous_absence_penalty(self, animals):
        """连续不出现惩罚"""
        scores = {}
        
        for zodiac in self.zodiacs:
            try:
                last_idx = len(animals) - 1 - animals[::-1].index(zodiac)
                last_appearance = len(animals) - last_idx - 1
            except ValueError:
                last_appearance = len(animals)
            
            if last_appearance >= 20:
                scores[zodiac] = -3.0
            elif last_appearance >= 15:
                scores[zodiac] = -1.0
            elif last_appearance >= 10:
                scores[zodiac] = 3.0
            elif last_appearance >= 6:
                scores[zodiac] = 3.0
            elif last_appearance >= 4:
                scores[zodiac] = 1.0
            else:
                scores[zodiac] = -1.0
        
        return scores
    
    def _diversity_boost(self, animals):
        """多样性提升"""
        scores = {}
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        appeared = set(recent_20)
        
        for zodiac in self.zodiacs:
            if zodiac not in appeared:
                scores[zodiac] = 10.0
            else:
                count = recent_20.count(zodiac)
                if count == 1:
                    scores[zodiac] = 5.0
                else:
                    scores[zodiac] = -2.0
        
        return scores
    
    def _historical_similarity(self, animals):
        """历史相似性"""
        scores = {}
        
        if len(animals) < 5:
            return {z: 0 for z in self.zodiacs}
        
        recent_5 = animals[-5:]
        
        for zodiac in self.zodiacs:
            score = 0.0
            for i in range(len(animals) - 10):
                window = animals[i:i+5]
                similarity = sum(1 for a, b in zip(recent_5, window) if a == b)
                if similarity >= 3 and len(animals) > i + 5:
                    next_animal = animals[i + 5]
                    if next_animal == zodiac:
                        score += 5.0 * similarity
            
            scores[zodiac] = score
        
        return scores
    
    def _get_model_prediction(self, animals, model_weights, model_name=""):
        """获取单个模型的预测分数"""
        # 收集各策略评分
        strategies_scores = {
            'ultra_cold': self._ultra_cold_strategy(animals),
            'anti_hot': self._anti_hot_strategy(animals),
            'gap': self._gap_analysis(animals),
            'rotation': self._rotation_advanced(animals),
            'absence_penalty': self._continuous_absence_penalty(animals),
            'diversity': self._diversity_boost(animals),
            'similarity': self._historical_similarity(animals)
        }
        
        # 如果模型使用热门惯性策略
        if 'hot_momentum_20' in model_weights:
            strategies_scores['hot_momentum_20'] = self._hot_momentum_20_strategy(animals)
        
        # 加权融合
        final_scores = {}
        for zodiac in self.zodiacs:
            score = 0.0
            for strategy_name, strategy_scores in strategies_scores.items():
                weight = model_weights.get(strategy_name, 0)
                score += strategy_scores.get(zodiac, 0) * weight
            final_scores[zodiac] = score
        
        return final_scores
    
    def predict_from_history(self, animals, top_n=5, debug=False):
        """集成多模型预测"""
        
        # 获取三个模型的预测
        model_a_scores = self._get_model_prediction(animals, self.model_a_weights, "模型A-冷门")
        model_b_scores = self._get_model_prediction(animals, self.model_b_weights, "模型B-惯性")
        model_c_scores = self._get_model_prediction(animals, self.model_c_weights, "模型C-间隔")
        
        if debug:
            print(f"\n{'='*90}")
            print("多模型集成预测（v8.0）")
            print(f"{'='*90}")
            print("\n模型A - 冷门主导型:")
            for name, weight in self.model_a_weights.items():
                print(f"  {name:20s}: {weight*100:5.1f}%")
            print("\n模型B - 冷门+惯性平衡型:")
            for name, weight in self.model_b_weights.items():
                print(f"  {name:20s}: {weight*100:5.1f}%")
            print("\n模型C - 间隔主导型:")
            for name, weight in self.model_c_weights.items():
                print(f"  {name:20s}: {weight*100:5.1f}%")
            print(f"\n集成权重:")
            for name, weight in self.ensemble_weights.items():
                print(f"  {name:20s}: {weight*100:5.1f}%")
            print(f"{'='*90}\n")
        
        # 集成融合（加权平均）
        ensemble_scores = {}
        for zodiac in self.zodiacs:
            score = 0.0
            score += model_a_scores[zodiac] * self.ensemble_weights['model_a']
            score += model_b_scores[zodiac] * self.ensemble_weights['model_b']
            score += model_c_scores[zodiac] * self.ensemble_weights['model_c']
            ensemble_scores[zodiac] = score
        
        # 排序
        sorted_zodiacs = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
        top_zodiacs = [z for z, s in sorted_zodiacs[:top_n]]
        
        # 附加信息：各模型的TOP5
        model_a_top5 = sorted(model_a_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        model_b_top5 = sorted(model_b_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        model_c_top5 = sorted(model_c_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return {
            'ensemble_top5': top_zodiacs,
            'ensemble_scores': sorted_zodiacs,
            'model_a_top5': [z for z, s in model_a_top5],
            'model_b_top5': [z for z, s in model_b_top5],
            'model_c_top5': [z for z, s in model_c_top5]
        }
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_n=5):
        """标准预测接口"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        result = self.predict_from_history(animals, top_n, debug=True)
        
        # 推荐号码
        recommended_numbers = []
        for rank, (zodiac, score) in enumerate(result['ensemble_scores'][:top_n], 1):
            weight = top_n + 1 - rank
            for num in self.zodiac_numbers[zodiac]:
                recommended_numbers.append((num, weight))
        
        num_scores = {}
        for num, w in recommended_numbers:
            num_scores[num] = num_scores.get(num, 0) + w
        
        sorted_nums = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_nums[:15]]
        
        return {
            'model': '生肖集成预测器(多模型融合)',
            'version': '8.0',
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_animal': df.iloc[-1]['animal'],
            'ensemble_top5': result['ensemble_top5'],
            'model_a_top5': result['model_a_top5'],
            'model_b_top5': result['model_b_top5'],
            'model_c_top5': result['model_c_top5'],
            'top15_numbers': top_numbers,
            'all_scores': result['ensemble_scores']
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
            period = df.iloc[i]['period']
            date = df.iloc[i]['date']
            
            hit_rank = 0
            if actual in prediction['ensemble_top5']:
                hit_rank = prediction['ensemble_top5'].index(actual) + 1
            
            results.append({
                'period': period,
                'date': date,
                'actual': actual,
                'predicted_top5': prediction['ensemble_top5'],
                'hit': hit_rank > 0,
                'rank': hit_rank
            })
        
        return results

if __name__ == '__main__':
    predictor = ZodiacEnsemblePredictor()
    result = predictor.predict()
    
    print(f"\n模型: {result['model']}")
    print(f"版本: {result['version']}")
    print(f"总期数: {result['total_periods']}")
    print(f"最新期: {result['last_date']} - {result['last_animal']}")
    print(f"\n集成预测TOP5: {', '.join(result['ensemble_top5'])}")
    print(f"\n各模型预测对比:")
    print(f"  模型A(冷门): {', '.join(result['model_a_top5'])}")
    print(f"  模型B(惯性): {', '.join(result['model_b_top5'])}")
    print(f"  模型C(间隔): {', '.join(result['model_c_top5'])}")
    print(f"\n推荐号码TOP15: {result['top15_numbers']}")
