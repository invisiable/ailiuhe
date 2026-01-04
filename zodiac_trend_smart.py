"""
生肖预测器 v11.0 - 实时趋势检测智能选择器
核心改进：基于当前数据实时检测冷门爆发和高频变化，动态切换模型
"""

import pandas as pd
import numpy as np
from collections import Counter
from zodiac_super_predictor import ZodiacSuperPredictor

class ZodiacTrendSmart:
    """v11.0 - 实时趋势检测智能预测器"""
    
    def __init__(self):
        self.base_predictor = ZodiacSuperPredictor()
        self.zodiacs = self.base_predictor.zodiacs
        self.zodiac_numbers = self.base_predictor.zodiac_numbers
    
    def _detect_trend_scenario(self, animals, debug=False):
        """
        实时趋势检测v2 - 基于当前数据动态判断
        改进：更激进地检测爆发，优先识别高频生肖
        """
        if len(animals) < 30:
            return 'normal', {}
        
        # 最近10期
        recent_10 = animals[-10:]
        # 前20期（用于对比基准）
        prev_20 = animals[-30:-10]
        
        recent_counter = Counter(recent_10)
        prev_counter = Counter(prev_20)
        
        # 检测趋势变化
        trend_analysis = {}
        burst_zodiacs = []  # 爆发生肖（冷→热）
        rising_zodiacs = []  # 上升生肖
        cooling_zodiacs = []  # 冷却生肖（热→冷）
        hot_zodiacs = []  # 持续热门
        
        # 首先：识别最近10期的高频生肖（无论历史如何）
        recent_hot = [z for z, c in recent_counter.items() if c >= 2]
        
        for zodiac in self.zodiacs:
            recent_count = recent_counter.get(zodiac, 0)
            prev_count = prev_counter.get(zodiac, 0)
            
            # 计算变化
            change = recent_count - prev_count
            
            trend_analysis[zodiac] = {
                'recent': recent_count,
                'prev': prev_count,
                'change': change
            }
            
            # 识别爆发（冷门突然高频）- 放宽条件
            if recent_count >= 2 and prev_count <= 1:
                burst_zodiacs.append(zodiac)
            # 识别上升（频率增加）
            elif recent_count >= 2 and change > 0:
                rising_zodiacs.append(zodiac)
            
            # 识别冷却（热门突然冷门）
            if prev_count >= 3 and recent_count <= 1:
                cooling_zodiacs.append(zodiac)
            
            # 识别持续热门
            if prev_count >= 2 and recent_count >= 2:
                hot_zodiacs.append(zodiac)
        
        # 决策逻辑
        scenario_info = {
            'burst_zodiacs': burst_zodiacs,
            'rising_zodiacs': rising_zodiacs,
            'cooling_zodiacs': cooling_zodiacs,
            'hot_zodiacs': hot_zodiacs,
            'recent_hot': recent_hot,
            'trend_analysis': trend_analysis
        }
        
        if debug:
            print(f"\n【趋势检测】")
            print(f"  最近10期高频: {recent_hot}")
            if burst_zodiacs:
                print(f"  爆发生肖: {burst_zodiacs}")
                for z in burst_zodiacs:
                    info = trend_analysis[z]
                    print(f"    {z}: 前20期{info['prev']}次 → 最近10期{info['recent']}次 (+{info['change']})")
            if rising_zodiacs:
                print(f"  上升生肖: {rising_zodiacs}")
            if hot_zodiacs:
                print(f"  持续热门: {hot_zodiacs}")
            if cooling_zodiacs:
                print(f"  冷却生肖: {cooling_zodiacs}")
        
        # 场景判断 - 优先级调整：爆发 > 上升 > 持续热门 > 冷却
        if len(burst_zodiacs) >= 1 or len(recent_hot) >= 3:
            # 存在爆发生肖 OR 最近高频生肖多 → 使用追热策略
            return 'burst_trend', scenario_info
        elif len(rising_zodiacs) >= 2 or len(hot_zodiacs) >= 2:
            # 多个上升或持续热门 → 使用热门感知
            return 'hot_stable', scenario_info
        elif len(cooling_zodiacs) >= 2 and len(recent_hot) <= 1:
            # 热门冷却且当前无明显高频 → 使用极致冷门
            return 'cooling_trend', scenario_info
        else:
            # 标准场景
            return 'normal', scenario_info
    
    def _hot_momentum_strategy(self, animals):
        """热门惯性策略 - 强化版，更关注最近10期"""
        scores = {}
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        
        for zodiac in self.zodiacs:
            score = 0.0
            
            # 最近10期权重更高
            count_10 = recent_10.count(zodiac)
            if count_10 >= 3:
                score += 60.0  # 强化最近表现
            elif count_10 == 2:
                score += 35.0
            elif count_10 == 1:
                score += 12.0
            
            # 最近20期作为辅助
            count_20 = recent_20.count(zodiac)
            if count_20 >= 4:
                score += 20.0
            elif count_20 >= 3:
                score += 10.0
            
            scores[zodiac] = score
        return scores
    
    def _burst_boost_strategy(self, animals, burst_zodiacs):
        """爆发加成策略 - 对检测到的爆发生肖大幅加分"""
        scores = {z: 0.0 for z in self.zodiacs}
        
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        
        for zodiac in self.zodiacs:
            count_10 = recent_10.count(zodiac)
            
            # 爆发生肖获得大幅加成
            if zodiac in burst_zodiacs:
                scores[zodiac] = 80.0 + count_10 * 15.0
            else:
                # 非爆发生肖按正常频率评分
                if count_10 >= 2:
                    scores[zodiac] = 30.0
                elif count_10 == 1:
                    scores[zodiac] = 10.0
        
        return scores
    
    def predict_from_history(self, animals, top_n=5, debug=False):
        """实时趋势检测预测"""
        
        # 实时检测趋势场景
        scenario, info = self._detect_trend_scenario(animals, debug=debug)
        
        if scenario == 'burst_trend':
            # 爆发趋势 → 追热+爆发加成
            weights = {
                'ultra_cold': 0.15, 'anti_hot': 0.05, 'gap': 0.15,
                'rotation': 0.10, 'absence_penalty': 0.05,
                'hot_momentum': 0.30, 'burst_boost': 0.15, 'diversity': 0.05
            }
            selected_model = '爆发追踪模型'
            
        elif scenario == 'hot_stable':
            # 持续热门 → 热门感知
            weights = {
                'ultra_cold': 0.20, 'anti_hot': 0.08, 'gap': 0.22,
                'rotation': 0.15, 'absence_penalty': 0.05,
                'hot_momentum': 0.22, 'diversity': 0.05, 'similarity': 0.03
            }
            selected_model = '热门稳定模型'
            
        elif scenario == 'cooling_trend':
            # 冷却趋势 → 极致冷门
            weights = {
                'ultra_cold': 0.45, 'anti_hot': 0.25, 'gap': 0.15,
                'rotation': 0.08, 'absence_penalty': 0.04,
                'diversity': 0.02, 'similarity': 0.01
            }
            selected_model = '冷却追踪模型'
            
        else:  # normal
            # 标准场景 → v5.0
            weights = {
                'ultra_cold': 0.35, 'anti_hot': 0.20, 'gap': 0.18,
                'rotation': 0.12, 'absence_penalty': 0.08,
                'diversity': 0.04, 'similarity': 0.03
            }
            selected_model = 'v5.0平衡模型'
        
        # 收集策略评分
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
        
        if 'burst_boost' in weights:
            strategies_scores['burst_boost'] = self._burst_boost_strategy(animals, info.get('burst_zodiacs', []))
        
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
        
        if debug:
            print(f"  场景判断: {scenario}")
            print(f"  选择模型: {selected_model}")
        
        return {
            'top5': top_zodiacs,
            'all_scores': sorted_zodiacs,
            'selected_model': selected_model,
            'scenario': scenario,
            'scenario_info': info
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
            'model': '实时趋势检测智能选择器',
            'version': '11.0',
            'selected_model': result['selected_model'],
            'scenario': result['scenario'],
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_animal': df.iloc[-1]['animal'],
            'top5': result['top5'],
            'top15_numbers': top_numbers,
            'scenario_info': result['scenario_info']
        }
    
    def get_recent_20_validation(self, csv_file='data/lucky_numbers.csv'):
        """获取最近20期验证结果"""
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
                'model': prediction['selected_model'],
                'scenario': prediction['scenario']
            })
        
        return results

if __name__ == '__main__':
    predictor = ZodiacTrendSmart()
    result = predictor.predict()
    
    print(f"\n模型: {result['model']}")
    print(f"版本: {result['version']}")
    print(f"场景: {result['scenario']}")
    print(f"选择的模型: {result['selected_model']}")
    print(f"总期数: {result['total_periods']}")
    print(f"最新期: {result['last_date']} - {result['last_animal']}")
    print(f"\n预测TOP5: {', '.join(result['top5'])}")
    print(f"推荐号码TOP15: {result['top15_numbers']}")
    
    # 显示场景信息
    info = result['scenario_info']
    if info.get('burst_zodiacs'):
        print(f"\n检测到爆发生肖: {info['burst_zodiacs']}")
    if info.get('hot_zodiacs'):
        print(f"检测到持续热门: {info['hot_zodiacs']}")
