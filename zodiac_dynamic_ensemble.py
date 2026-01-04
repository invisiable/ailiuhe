"""
生肖预测器 v8.1 - 动态权重集成版
根据近期表现动态调整模型权重
"""

import pandas as pd
import numpy as np
from collections import Counter

class ZodiacDynamicEnsemble:
    """生肖动态集成预测器"""
    
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
        
        # v5.0最优配置（基准）
        self.base_weights = {
            'ultra_cold': 0.35,
            'anti_hot': 0.20,
            'gap': 0.18,
            'rotation': 0.12,
            'absence_penalty': 0.08,
            'hot_momentum_20': 0.00,  # 不使用
            'diversity': 0.04,
            'similarity': 0.03
        }
        
        # 热门惯性补偿权重
        self.momentum_weights = {
            'ultra_cold': 0.25,
            'anti_hot': 0.10,
            'gap': 0.20,
            'rotation': 0.12,
            'absence_penalty': 0.06,
            'hot_momentum_20': 0.20,  # 启用20期惯性
            'diversity': 0.04,
            'similarity': 0.03
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
        """20期热门惯性策略"""
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
    
    def _detect_hot_momentum(self, animals, window=20):
        """检测是否有明显的热门惯性"""
        if len(animals) < window:
            return False, 0.0
        
        recent = animals[-window:]
        counter = Counter(recent)
        
        # 计算集中度
        hot_count = sum(1 for count in counter.values() if count >= 2)
        concentration = hot_count / len(counter) if counter else 0
        
        # 如果有3个以上生肖出现2+次，且集中度>40%，认为有热门惯性
        has_momentum = hot_count >= 3 and concentration >= 0.40
        
        return has_momentum, concentration
    
    def predict_from_history(self, animals, top_n=5, debug=False):
        """动态权重预测"""
        
        # 检测热门惯性
        has_momentum, concentration = self._detect_hot_momentum(animals)
        
        # 动态选择权重
        if has_momentum:
            # 使用混合权重：70%基准 + 30%惯性
            weights = {}
            for key in self.base_weights:
                weights[key] = (
                    self.base_weights[key] * 0.70 +
                    self.momentum_weights.get(key, 0) * 0.30
                )
            mode = "混合模式(70%基准+30%惯性)"
        else:
            # 完全使用基准权重
            weights = self.base_weights.copy()
            mode = "基准模式(v5.0)"
        
        if debug:
            print(f"\n{'='*80}")
            print(f"动态权重预测（v8.1）")
            print(f"{'='*80}")
            print(f"热门惯性检测: {'是' if has_momentum else '否'} (集中度: {concentration*100:.1f}%)")
            print(f"采用模式: {mode}")
            print(f"\n权重配置:")
            for name, weight in weights.items():
                if weight > 0:
                    print(f"  {name:20s}: {weight*100:5.1f}%")
            print(f"{'='*80}\n")
        
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
        
        if weights.get('hot_momentum_20', 0) > 0:
            strategies_scores['hot_momentum_20'] = self._hot_momentum_20_strategy(animals)
        
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
        
        return top_zodiacs, sorted_zodiacs, mode
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_n=5):
        """标准预测接口"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        top_zodiacs, all_scores, mode = self.predict_from_history(animals, top_n, debug=True)
        
        # 推荐号码
        recommended_numbers = []
        for rank, (zodiac, score) in enumerate(all_scores[:top_n], 1):
            weight = top_n + 1 - rank
            for num in self.zodiac_numbers[zodiac]:
                recommended_numbers.append((num, weight))
        
        num_scores = {}
        for num, w in recommended_numbers:
            num_scores[num] = num_scores.get(num, 0) + w
        
        sorted_nums = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_nums[:15]]
        
        return {
            'model': '生肖动态集成预测器',
            'version': '8.1',
            'mode': mode,
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_animal': df.iloc[-1]['animal'],
            'top5': top_zodiacs,
            'top15_numbers': top_numbers
        }
    
    def get_recent_20_validation(self, csv_file='data/lucky_numbers.csv'):
        """获取最近20期验证结果"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        total = len(df)
        
        results = []
        for i in range(max(0, total - 20), total):
            animals = [str(a).strip() for a in df['animal'].values[:i]]
            top5, _, mode = self.predict_from_history(animals, top_n=5)
            
            actual = str(df['animal'].values[i]).strip()
            period = df.iloc[i]['period']
            date = df.iloc[i]['date']
            
            hit_rank = 0
            if actual in top5:
                hit_rank = top5.index(actual) + 1
            
            results.append({
                'period': period,
                'date': date,
                'actual': actual,
                'predicted_top5': top5,
                'hit': hit_rank > 0,
                'rank': hit_rank,
                'mode': mode
            })
        
        return results

if __name__ == '__main__':
    predictor = ZodiacDynamicEnsemble()
    result = predictor.predict()
    
    print(f"\n模型: {result['model']}")
    print(f"版本: {result['version']}")
    print(f"模式: {result['mode']}")
    print(f"总期数: {result['total_periods']}")
    print(f"最新期: {result['last_date']} - {result['last_animal']}")
    print(f"\n预测TOP5: {', '.join(result['top5'])}")
    print(f"推荐号码TOP15: {result['top15_numbers']}")
