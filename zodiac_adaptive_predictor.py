"""
自适应生肖预测器 v6.0
新增热门连续性检测机制，动态调整策略权重
"""

import pandas as pd
import numpy as np
from collections import Counter

class ZodiacAdaptivePredictor:
    """生肖自适应预测器 - 可根据周期特征动态调整"""
    
    def __init__(self):
        self.zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        # 生肖对应号码
        self.zodiac_numbers = {
            '鼠': [1, 13, 25, 37, 49], '牛': [2, 14, 26, 38],
            '虎': [3, 15, 27, 39], '兔': [4, 16, 28, 40],
            '龙': [5, 17, 29, 41], '蛇': [6, 18, 30, 42],
            '马': [7, 19, 31, 43], '羊': [8, 20, 32, 44],
            '猴': [9, 21, 33, 45], '鸡': [10, 22, 34, 46],
            '狗': [11, 23, 35, 47], '猪': [12, 24, 36, 48]
        }
        
        # 默认权重配置（冷门导向）
        self.default_weights = {
            'ultra_cold': 0.35,
            'anti_hot': 0.20,
            'gap': 0.18,
            'rotation': 0.12,
            'absence_penalty': 0.08,
            'diversity': 0.04,
            'similarity': 0.03
        }
        
        # 热门周期权重配置（优化版：保留一定冷门权重）
        self.hot_cycle_weights = {
            'ultra_cold': 0.20,      # 降低但保留冷门权重
            'anti_hot': 0.08,        # 降低反热门权重
            'gap': 0.20,             # 提升间隔权重
            'rotation': 0.15,        # 提升轮转权重
            'absence_penalty': 0.05, # 降低惩罚权重
            'hot_boost': 0.20,       # 热门提升策略
            'diversity': 0.08,       # 提升多样性权重
            'similarity': 0.04       # 提升相似性权重
        }
    
    def _detect_hot_cycle(self, animals, window=15, hot_threshold=3):
        """
        检测是否进入热门周期
        
        参数:
            animals: 历史生肖列表
            window: 检测窗口（最近N期）
            hot_threshold: 热门阈值（单个生肖最少出现次数才算热门）
        
        返回:
            (is_hot_cycle, hot_zodiacs, stats)
        """
        if len(animals) < window:
            return False, [], {}
        
        recent = animals[-window:]
        counter = Counter(recent)
        
        # 统计
        total_zodiacs = len(set(recent))
        
        # 找出真正的热门生肖（出现>=hot_threshold次的）
        hot_zodiacs = []
        for zodiac, count in counter.items():
            if count >= hot_threshold:
                hot_zodiacs.append((zodiac, count))
        
        # 计算集中度（热门生肖占比）
        if hot_zodiacs:
            hot_total = sum(count for _, count in hot_zodiacs)
            concentration = hot_total / window
        else:
            concentration = 0
        
        # 计算重复率（出现2次以上的生肖比例）
        repeated_count = sum(1 for count in counter.values() if count >= 2)
        repeat_rate = repeated_count / len(counter) if counter else 0
        
        # 判断是否为热门周期（严格条件）
        # 必须满足：有2个以上热门生肖(>=3次) 且 集中度>=40%
        is_hot = len(hot_zodiacs) >= 2 and concentration >= 0.40
        
        stats = {
            'window': window,
            'total_zodiacs': total_zodiacs,
            'hot_count': len(hot_zodiacs),
            'concentration': concentration,
            'repeat_rate': repeat_rate,
            'hot_zodiacs': sorted(hot_zodiacs, key=lambda x: x[1], reverse=True)
        }
        
        return is_hot, [z for z, _ in hot_zodiacs], stats
    
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
    
    def _hot_boost_strategy(self, animals):
        """热门提升策略（新增）"""
        scores = {}
        recent_15 = animals[-15:] if len(animals) >= 15 else animals
        
        for zodiac in self.zodiacs:
            count = recent_15.count(zodiac)
            if count >= 3:
                scores[zodiac] = 25.0  # 强热门
            elif count == 2:
                scores[zodiac] = 15.0  # 中热门
            elif count == 1:
                scores[zodiac] = 5.0   # 轻热门
            else:
                scores[zodiac] = 0.0   # 冷门
        
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
    
    def predict_with_adaptive_weights(self, animals, top_n=5, debug=False):
        """使用自适应权重预测"""
        
        # 检测周期类型
        is_hot_cycle, hot_zodiacs, stats = self._detect_hot_cycle(animals)
        
        # 选择权重配置
        if is_hot_cycle:
            weights = self.hot_cycle_weights.copy()
            cycle_type = "热门周期"
        else:
            weights = self.default_weights.copy()
            weights['hot_boost'] = 0.0  # 冷门周期不使用热门提升
            cycle_type = "冷门周期"
        
        if debug:
            print(f"\n{'='*80}")
            print(f"周期检测: {cycle_type}")
            print(f"{'='*80}")
            print(f"  检测窗口: 最近{stats['window']}期")
            print(f"  生肖多样性: {stats['total_zodiacs']}/12")
            print(f"  热门生肖数: {stats['hot_count']}")
            print(f"  热门集中度: {stats['concentration']*100:.1f}%")
            print(f"  重复出现率: {stats['repeat_rate']*100:.1f}%")
            if stats['hot_zodiacs']:
                print(f"  热门生肖: {', '.join([f'{z}({c}次)' for z, c in stats['hot_zodiacs']])}")
            print(f"\n  采用权重配置:")
            for name, weight in weights.items():
                print(f"    {name:20s}: {weight*100:5.1f}%")
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
        
        # 如果是热门周期，添加热门提升策略
        if is_hot_cycle:
            strategies_scores['hot_boost'] = self._hot_boost_strategy(animals)
        
        # 加权融合
        final_scores = {}
        for zodiac in self.zodiacs:
            score = 0.0
            for strategy_name, strategy_scores in strategies_scores.items():
                weight = weights.get(strategy_name, 0)
                score += strategy_scores.get(zodiac, 0) * weight
            final_scores[zodiac] = score
        
        # 排序
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_zodiacs = sorted_zodiacs[:top_n]
        
        result = {
            'cycle_type': cycle_type,
            'is_hot_cycle': is_hot_cycle,
            'hot_zodiacs': hot_zodiacs,
            'stats': stats,
            'weights': weights,
            'top5': [z for z, s in top_zodiacs],
            'top5_scores': top_zodiacs,
            'all_scores': sorted_zodiacs
        }
        
        return result
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_n=5):
        """标准预测接口"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        result = self.predict_with_adaptive_weights(animals, top_n, debug=True)
        
        # 推荐号码
        recommended_numbers = []
        for rank, (zodiac, score) in enumerate(result['top5_scores'], 1):
            weight = top_n + 1 - rank
            for num in self.zodiac_numbers[zodiac]:
                recommended_numbers.append((num, weight))
        
        num_scores = {}
        for num, w in recommended_numbers:
            num_scores[num] = num_scores.get(num, 0) + w
        
        sorted_nums = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_nums[:15]]
        
        return {
            'model': '生肖自适应预测器(动态权重)',
            'version': '6.0',
            'cycle_type': result['cycle_type'],
            'is_hot_cycle': result['is_hot_cycle'],
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_animal': df.iloc[-1]['animal'],
            'top5': result['top5'],
            'top6': [z for z, s in result['all_scores'][:6]],
            'top15_numbers': top_numbers,
            'all_scores': result['all_scores']
        }
    
    def get_recent_20_validation(self, csv_file='data/lucky_numbers.csv'):
        """获取最近20期验证结果（GUI兼容）"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        total = len(df)
        
        results = []
        for i in range(max(0, total - 20), total):
            animals = [str(a).strip() for a in df['animal'].values[:i]]
            prediction = self.predict_with_adaptive_weights(animals, top_n=5)
            
            actual = str(df['animal'].values[i]).strip()
            period = df.iloc[i]['period']
            date = df.iloc[i]['date']
            
            hit_rank = 0
            if actual in prediction['top5']:
                hit_rank = prediction['top5'].index(actual) + 1
            
            results.append({
                'period': period,
                'date': date,
                'actual': actual,
                'predicted_top5': prediction['top5'],
                'hit': hit_rank > 0,
                'rank': hit_rank,
                'cycle_type': prediction['cycle_type']
            })
        
        return results

if __name__ == '__main__':
    predictor = ZodiacAdaptivePredictor()
    result = predictor.predict()
    
    print(f"\n模型: {result['model']}")
    print(f"版本: {result['version']}")
    print(f"当前周期: {result['cycle_type']}")
    print(f"总期数: {result['total_periods']}")
    print(f"最新期: {result['last_date']} - {result['last_animal']}")
    print(f"\n预测TOP5: {', '.join(result['top5'])}")
    print(f"预测TOP6: {', '.join(result['top6'])}")
    print(f"推荐号码TOP15: {result['top15_numbers']}")
