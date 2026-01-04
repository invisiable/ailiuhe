"""
生肖预测器 v7.0 - 最终优化版
在冷门策略基础上增加热门惯性补偿
"""

import pandas as pd
import numpy as np
from collections import Counter

class ZodiacFinalPredictor:
    """生肖最终预测器 - 冷门策略 + 热门惯性平衡"""
    
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
        
        # 最优权重配置（基于v5.0 + 热门惯性补偿）
        self.weights = {
            'ultra_cold': 0.30,           # 略降冷门权重（原35%）
            'anti_hot': 0.15,             # 降低反热门权重（原20%）
            'gap': 0.18,                  # 保持间隔权重
            'rotation': 0.12,             # 保持轮转权重
            'absence_penalty': 0.08,      # 保持惩罚权重
            'hot_momentum': 0.10,         # 新增！热门惯性补偿
            'diversity': 0.04,            # 保持多样性
            'similarity': 0.03            # 保持相似性
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
    
    def _hot_momentum_strategy(self, animals):
        """
        热门惯性策略（新增）
        理念：近期频繁出现的生肖可能有"惯性"继续出现
        补偿冷门策略对热门生肖的过度压制
        """
        scores = {}
        
        # 多时间窗口综合评估
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        
        for zodiac in self.zodiacs:
            score = 0.0
            
            # 最近10期权重更高
            count_10 = recent_10.count(zodiac)
            if count_10 >= 3:
                score += 30.0  # 强惯性
            elif count_10 == 2:
                score += 18.0  # 中惯性
            elif count_10 == 1:
                score += 8.0   # 轻惯性
            
            # 最近20期作为补充
            count_20 = recent_20.count(zodiac)
            if count_20 >= 4:
                score += 10.0
            elif count_20 == 3:
                score += 5.0
            
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
    
    def predict_from_history(self, animals, top_n=5, debug=False):
        """从历史数据预测"""
        
        # 收集各策略评分
        strategies_scores = {
            'ultra_cold': self._ultra_cold_strategy(animals),
            'anti_hot': self._anti_hot_strategy(animals),
            'hot_momentum': self._hot_momentum_strategy(animals),  # 新增
            'gap': self._gap_analysis(animals),
            'rotation': self._rotation_advanced(animals),
            'absence_penalty': self._continuous_absence_penalty(animals),
            'diversity': self._diversity_boost(animals),
            'similarity': self._historical_similarity(animals)
        }
        
        if debug:
            print(f"\n{'='*80}")
            print("策略权重配置（v7.0 最终版）")
            print(f"{'='*80}")
            for name, weight in self.weights.items():
                print(f"  {name:20s}: {weight*100:5.1f}%")
            print(f"{'='*80}\n")
        
        # 加权融合
        final_scores = {}
        for zodiac in self.zodiacs:
            score = 0.0
            for strategy_name, strategy_scores in strategies_scores.items():
                weight = self.weights.get(strategy_name, 0)
                score += strategy_scores.get(zodiac, 0) * weight
            final_scores[zodiac] = score
        
        # 排序
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_zodiacs = [z for z, s in sorted_zodiacs[:top_n]]
        
        return top_zodiacs, sorted_zodiacs
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_n=5):
        """标准预测接口"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        top_zodiacs, all_scores = self.predict_from_history(animals, top_n, debug=True)
        
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
            'model': '生肖最终预测器(冷门+惯性平衡)',
            'version': '7.0',
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_animal': df.iloc[-1]['animal'],
            'top5': top_zodiacs,
            'top6': [z for z, s in all_scores[:6]],
            'top15_numbers': top_numbers,
            'all_scores': all_scores
        }
    
    def get_recent_20_validation(self, csv_file='data/lucky_numbers.csv'):
        """获取最近20期验证结果（GUI兼容）"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        total = len(df)
        
        results = []
        for i in range(max(0, total - 20), total):
            animals = [str(a).strip() for a in df['animal'].values[:i]]
            top5, _ = self.predict_from_history(animals, top_n=5)
            
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
                'rank': hit_rank
            })
        
        return results

if __name__ == '__main__':
    predictor = ZodiacFinalPredictor()
    result = predictor.predict()
    
    print(f"\n模型: {result['model']}")
    print(f"版本: {result['version']}")
    print(f"总期数: {result['total_periods']}")
    print(f"最新期: {result['last_date']} - {result['last_animal']}")
    print(f"\n预测TOP5: {', '.join(result['top5'])}")
    print(f"预测TOP6: {', '.join(result['top6'])}")
    print(f"推荐号码TOP15: {result['top15_numbers']}")
