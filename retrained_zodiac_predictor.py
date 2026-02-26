"""
重训练后的生肖预测器 v2.0
基于最近200期数据优化，使用热门感知策略
"""

import pandas as pd
import numpy as np
from collections import Counter
from zodiac_super_predictor import ZodiacSuperPredictor


class RetrainedZodiacPredictor:
    """重训练的生肖预测器 - v2.0"""
    
    def __init__(self):
        self.base_predictor = ZodiacSuperPredictor()
        self.zodiacs = self.base_predictor.zodiacs
        self.zodiac_numbers = self.base_predictor.zodiac_numbers
        
        # 基于最近200期数据的优化权重
        self.weights = {
            'cold_boost': 0.25,
            'anti_hot': 0.15,
            'gap_analysis': 0.25,
            'hot_momentum': 0.20,
            'rotation': 0.10,
            'diversity': 0.05
        }
        
        # 从分析中得到的热门/冷门生肖
        self.known_hot = {'蛇', '虎', '鼠'}
        self.known_cold = {'猴', '鸡'}
    
    def _cold_boost_strategy(self, animals):
        """冷号提升策略"""
        scores = {}
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        
        freq_30 = Counter(recent_30)
        freq_20 = Counter(recent_20)
        freq_10 = Counter(recent_10)
        
        for zodiac in self.zodiacs:
            score = 0.0
            
            # 30期内频率
            count_30 = freq_30.get(zodiac, 0)
            if count_30 == 0:
                score += 3.0  # 30期内未出现
            elif count_30 == 1:
                score += 2.0
            elif count_30 == 2:
                score += 1.0
            elif count_30 >= 5:
                score -= 1.5  # 高频降权
            
            # 20期内频率
            count_20 = freq_20.get(zodiac, 0)
            if count_20 == 0:
                score += 1.5
            elif count_20 >= 4:
                score -= 1.0
            
            # 10期内频率
            count_10 = freq_10.get(zodiac, 0)
            if count_10 == 0:
                score += 1.0
            elif count_10 >= 3:
                score -= 0.5
            
            scores[zodiac] = max(0, score)  # 确保非负
        
        return scores
    
    def _anti_hot_strategy(self, animals):
        """反热门策略"""
        scores = {}
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        freq = Counter(recent_30)
        
        avg_count = len(recent_30) / 12
        
        for zodiac in self.zodiacs:
            count = freq.get(zodiac, 0)
            if count == 0:
                score = 2.5
            elif count < avg_count:
                score = 2.0
            elif count == avg_count:
                score = 1.0
            else:
                score = 0.5 / count  # 高频强降权
            
            scores[zodiac] = score
        
        return scores
    
    def _gap_analysis(self, animals):
        """间隔分析"""
        scores = {}
        last_seen = {}
        
        for i, z in enumerate(animals):
            last_seen[z] = i
        
        current_pos = len(animals)
        
        for zodiac in self.zodiacs:
            if zodiac in last_seen:
                gap = current_pos - last_seen[zodiac]
                
                # 优化间隔评分
                if 1 <= gap <= 2:
                    score = 0.5  # 刚出现过
                elif 3 <= gap <= 8:
                    score = 2.5  # 最佳间隔
                elif 9 <= gap <= 15:
                    score = 2.0
                elif 16 <= gap <= 25:
                    score = 1.8
                else:
                    score = 1.5  # 超长间隔
            else:
                score = 2.0  # 从未出现
            
            scores[zodiac] = score
        
        return scores
    
    def _hot_momentum_strategy(self, animals):
        """热门惯性策略"""
        scores = {}
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        
        freq_20 = Counter(recent_20)
        freq_30 = Counter(recent_30)
        
        for zodiac in self.zodiacs:
            score = 0.0
            count_20 = freq_20.get(zodiac, 0)
            count_30 = freq_30.get(zodiac, 0)
            
            # 20期内热门
            if count_20 >= 4:
                score += 2.5
            elif count_20 == 3:
                score += 1.8
            elif count_20 == 2:
                score += 1.0
            elif count_20 == 1:
                score += 0.3
            
            # 30期内趋势
            if count_30 >= 6:
                score += 1.5
            elif count_30 >= 4:
                score += 0.8
            
            scores[zodiac] = score
        
        return scores
    
    def _rotation_strategy(self, animals):
        """轮转策略"""
        scores = {}
        recent_12 = animals[-12:] if len(animals) >= 12 else animals
        appeared = set(recent_12)
        
        for zodiac in self.zodiacs:
            if zodiac not in appeared:
                scores[zodiac] = 2.0
            else:
                scores[zodiac] = 0.5
        
        return scores
    
    def _diversity_strategy(self, animals):
        """多样性策略"""
        scores = {}
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        freq = Counter(recent_20)
        
        diversity = len(freq) / 12
        
        if diversity > 0.8:  # 高多样性，倾向出现过的
            for zodiac in self.zodiacs:
                if zodiac in freq:
                    scores[zodiac] = 1.0
                else:
                    scores[zodiac] = 0.5
        else:  # 低多样性，倾向未出现的
            for zodiac in self.zodiacs:
                if zodiac not in freq:
                    scores[zodiac] = 1.5
                else:
                    scores[zodiac] = 0.8
        
        return scores
    
    def predict_from_history(self, animals, top_n=4, debug=False):
        """预测TOP N生肖"""
        # 收集各策略评分
        strategies = {
            'cold_boost': self._cold_boost_strategy(animals),
            'anti_hot': self._anti_hot_strategy(animals),
            'gap_analysis': self._gap_analysis(animals),
            'hot_momentum': self._hot_momentum_strategy(animals),
            'rotation': self._rotation_strategy(animals),
            'diversity': self._diversity_strategy(animals)
        }
        
        # 加权融合
        final_scores = {}
        for zodiac in self.zodiacs:
            score = 0.0
            for strategy_name, strategy_scores in strategies.items():
                weight = self.weights[strategy_name]
                score += strategy_scores.get(zodiac, 0) * weight
            final_scores[zodiac] = score
        
        # 排序
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        if debug:
            print(f"\n【各生肖综合评分】")
            for zodiac, score in sorted_zodiacs:
                print(f"  {zodiac}: {score:.2f}")
        
        return {
            'top3': [z for z, s in sorted_zodiacs[:3]],
            'top4': [z for z, s in sorted_zodiacs[:4]],
            'top5': [z for z, s in sorted_zodiacs[:5]],
            'top6': [z for z, s in sorted_zodiacs[:6]],
            'all_scores': sorted_zodiacs,
            'selected_model': '重训练v2.0-热门感知策略'
        }
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_n=4):
        """标准预测接口"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        result = self.predict_from_history(animals, top_n, debug=True)
        
        # 推荐号码
        recommended_numbers = []
        for rank, zodiac in enumerate(result['top5'], 1):
            weight = 5 + 1 - rank
            for num in self.zodiac_numbers[zodiac]:
                recommended_numbers.append((num, weight))
        
        num_scores = {}
        for num, w in recommended_numbers:
            num_scores[num] = num_scores.get(num, 0) + w
        
        sorted_nums = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_nums[:15]]
        
        return {
            'model': '重训练生肖预测器',
            'version': '2.0',
            'selected_model': result['selected_model'],
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_animal': df.iloc[-1]['animal'],
            'top3': result['top3'],
            'top4': result['top4'],
            'top5': result['top5'],
            'top15_numbers': top_numbers,
            'weights': self.weights
        }


if __name__ == '__main__':
    print("="*80)
    print("重训练生肖预测器 v2.0 测试")
    print("="*80 + "\n")
    
    predictor = RetrainedZodiacPredictor()
    result = predictor.predict()
    
    print(f"\n模型: {result['model']} v{result['version']}")
    print(f"策略: {result['selected_model']}")
    print(f"数据: 第{result['total_periods']}期 ({result['last_date']})")
    print(f"上期生肖: {result['last_animal']}")
    
    print(f"\n【预测结果】")
    print(f"TOP3生肖: {', '.join(result['top3'])}")
    print(f"TOP4生肖: {', '.join(result['top4'])}")
    print(f"TOP5生肖: {', '.join(result['top5'])}")
    
    print(f"\nTOP15号码: {result['top15_numbers']}")
    
    print(f"\n权重配置:")
    for name, weight in result['weights'].items():
        print(f"  {name}: {weight}")
