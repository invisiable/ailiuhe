"""
自适应生肖预测器 v3.0
核心特性:
1. 动态训练窗口（最近30-50期）
2. 自动策略切换（冷门/热门/平衡）
3. 异常检测与紧急重训练
4. 实时性能监控
"""

import pandas as pd
import numpy as np
from collections import Counter, deque
from datetime import datetime


class AdaptiveZodiacPredictor:
    """自适应生肖预测器"""
    
    def __init__(self, train_window=50, monitor_window=10):
        self.zodiacs = ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪']
        self.zodiac_numbers = {
            '鼠': [6, 18, 30, 42], '牛': [2, 14, 26, 38],
            '虎': [4, 16, 28, 40], '兔': [3, 15, 27, 39],
            '龙': [8, 20, 32, 44], '蛇': [1, 13, 25, 37, 49],
            '马': [5, 17, 29, 41], '羊': [7, 19, 31, 43],
            '猴': [11, 23, 35, 47], '鸡': [9, 21, 33, 45],
            '狗': [10, 22, 34, 46], '猪': [12, 24, 36, 48]
        }
        
        self.train_window = train_window  # 训练窗口大小
        self.monitor_window = monitor_window  # 监控窗口大小
        
        # 性能监控
        self.recent_performance = deque(maxlen=monitor_window)
        self.weights = None
        self.strategy_type = None
        self.last_retrain_period = 0
        
    def detect_data_pattern(self, animals):
        """检测当前数据模式"""
        recent = animals[-30:] if len(animals) >= 30 else animals
        freq = Counter(recent)
        
        # 关键指标
        diversity = len(freq) / 12  # 多样性
        disappeared = sum(1 for z in self.zodiacs if freq.get(z, 0) == 0)  # 消失生肖数
        hot_count = sum(1 for c in freq.values() if c >= 4)  # 热门生肖数
        max_freq = max(freq.values()) if freq else 0  # 最高频次
        
        # 模式判断
        if disappeared >= 3:
            return 'extreme_concentrated'  # 极端集中
        elif hot_count >= 3 or max_freq >= 5:
            return 'hot_dominant'  # 热门主导
        elif diversity >= 0.9:
            return 'diverse'  # 高度多样
        elif diversity <= 0.6:
            return 'concentrated'  # 集中
        else:
            return 'balanced'  # 平衡
    
    def auto_adjust_weights(self, animals):
        """根据数据模式自动调整权重"""
        pattern = self.detect_data_pattern(animals)
        
        if pattern == 'extreme_concentrated':
            # 极端集中：强调短期模式
            self.strategy_type = '极端集中模式'
            self.weights = {
                'recent_hot': 0.40,      # 强调最近热门
                'short_gap': 0.30,       # 短期间隔
                'momentum': 0.20,        # 惯性
                'anti_cold': 0.10        # 反冷门
            }
        elif pattern == 'hot_dominant':
            # 热门主导：跟随热点
            self.strategy_type = '热门主导模式'
            self.weights = {
                'recent_hot': 0.35,
                'momentum': 0.30,
                'short_gap': 0.20,
                'rotation': 0.15
            }
        elif pattern == 'diverse':
            # 高度多样：轮转优先
            self.strategy_type = '高度多样模式'
            self.weights = {
                'rotation': 0.35,
                'gap_balanced': 0.30,
                'cold_moderate': 0.20,
                'diversity': 0.15
            }
        elif pattern == 'concentrated':
            # 集中模式：冷号补充
            self.strategy_type = '集中模式'
            self.weights = {
                'cold_boost': 0.35,
                'gap_balanced': 0.30,
                'rotation': 0.20,
                'anti_hot': 0.15
            }
        else:  # balanced
            # 平衡模式：综合策略
            self.strategy_type = '平衡模式'
            self.weights = {
                'cold_boost': 0.25,
                'gap_balanced': 0.25,
                'momentum': 0.20,
                'rotation': 0.15,
                'diversity': 0.15
            }
        
        return pattern
    
    def _recent_hot_strategy(self, animals, window=15):
        """最近热门策略（短时间窗口）"""
        scores = {}
        recent = animals[-window:] if len(animals) >= window else animals
        freq = Counter(recent)
        
        for zodiac in self.zodiacs:
            count = freq.get(zodiac, 0)
            if count >= 3:
                scores[zodiac] = 3.0
            elif count == 2:
                scores[zodiac] = 2.0
            elif count == 1:
                scores[zodiac] = 1.0
            else:
                scores[zodiac] = 0.2
        
        return scores
    
    def _short_gap_strategy(self, animals):
        """短期间隔策略（优先3-8期间隔）"""
        scores = {}
        last_seen = {}
        
        for i, z in enumerate(animals):
            last_seen[z] = i
        
        current_pos = len(animals)
        
        for zodiac in self.zodiacs:
            if zodiac in last_seen:
                gap = current_pos - last_seen[zodiac]
                if 3 <= gap <= 8:
                    scores[zodiac] = 3.0
                elif gap == 2:
                    scores[zodiac] = 2.0
                elif 9 <= gap <= 12:
                    scores[zodiac] = 1.8
                elif gap == 1:
                    scores[zodiac] = 0.5
                else:
                    scores[zodiac] = 1.0
            else:
                scores[zodiac] = 2.5
        
        return scores
    
    def _momentum_strategy(self, animals):
        """惯性策略（跟随趋势）"""
        scores = {}
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        
        freq_10 = Counter(recent_10)
        freq_20 = Counter(recent_20)
        
        for zodiac in self.zodiacs:
            score = 0.0
            c10 = freq_10.get(zodiac, 0)
            c20 = freq_20.get(zodiac, 0)
            
            # 10期内频次
            if c10 >= 3:
                score += 2.5
            elif c10 == 2:
                score += 1.5
            elif c10 == 1:
                score += 0.5
            
            # 20期趋势
            if c20 >= 5:
                score += 1.0
            elif c20 >= 3:
                score += 0.5
            
            scores[zodiac] = score
        
        return scores
    
    def _cold_boost_strategy(self, animals):
        """冷号提升策略"""
        scores = {}
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        freq = Counter(recent_30)
        
        for zodiac in self.zodiacs:
            count = freq.get(zodiac, 0)
            if count == 0:
                scores[zodiac] = 3.0
            elif count == 1:
                scores[zodiac] = 2.0
            elif count == 2:
                scores[zodiac] = 1.0
            else:
                scores[zodiac] = 0.3
        
        return scores
    
    def _rotation_strategy(self, animals):
        """轮转策略"""
        scores = {}
        recent = animals[-12:] if len(animals) >= 12 else animals
        appeared = set(recent)
        
        for zodiac in self.zodiacs:
            if zodiac not in appeared:
                scores[zodiac] = 2.5
            else:
                scores[zodiac] = 0.5
        
        return scores
    
    def _gap_balanced_strategy(self, animals):
        """平衡间隔策略"""
        scores = {}
        last_seen = {}
        
        for i, z in enumerate(animals):
            last_seen[z] = i
        
        current_pos = len(animals)
        avg_gap = len(animals) / 12
        
        for zodiac in self.zodiacs:
            if zodiac in last_seen:
                gap = current_pos - last_seen[zodiac]
                diff = abs(gap - avg_gap)
                scores[zodiac] = max(1.0, 3.0 - diff * 0.2)
            else:
                scores[zodiac] = 2.5
        
        return scores
    
    def _anti_hot_strategy(self, animals):
        """反热门策略"""
        scores = {}
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        freq = Counter(recent_20)
        
        for zodiac in self.zodiacs:
            count = freq.get(zodiac, 0)
            if count == 0:
                scores[zodiac] = 3.0
            elif count <= 2:
                scores[zodiac] = 2.0
            else:
                scores[zodiac] = 1.0 / count
        
        return scores
    
    def _anti_cold_strategy(self, animals):
        """反冷门策略（跟随出现过的）"""
        scores = {}
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        freq = Counter(recent_20)
        
        for zodiac in self.zodiacs:
            count = freq.get(zodiac, 0)
            if count >= 2:
                scores[zodiac] = 2.5
            elif count == 1:
                scores[zodiac] = 1.5
            else:
                scores[zodiac] = 0.5
        
        return scores
    
    def _diversity_strategy(self, animals):
        """多样性策略"""
        scores = {}
        recent_15 = animals[-15:] if len(animals) >= 15 else animals
        freq = Counter(recent_15)
        
        # 如果多样性高，倾向已出现的
        diversity = len(freq) / 12
        
        for zodiac in self.zodiacs:
            if diversity > 0.8:
                scores[zodiac] = 1.5 if zodiac in freq else 0.8
            else:
                scores[zodiac] = 1.8 if zodiac not in freq else 1.0
        
        return scores
    
    def predict_from_history(self, animals, top_n=4, debug=False):
        """自适应预测"""
        # 使用最近的数据进行训练（动态窗口）
        train_data = animals[-self.train_window:] if len(animals) >= self.train_window else animals
        
        # 自动调整权重
        pattern = self.auto_adjust_weights(train_data)
        
        # 收集策略评分
        strategies = {}
        for strategy_name in self.weights.keys():
            if strategy_name == 'recent_hot':
                strategies[strategy_name] = self._recent_hot_strategy(train_data)
            elif strategy_name == 'short_gap':
                strategies[strategy_name] = self._short_gap_strategy(train_data)
            elif strategy_name == 'momentum':
                strategies[strategy_name] = self._momentum_strategy(train_data)
            elif strategy_name == 'cold_boost':
                strategies[strategy_name] = self._cold_boost_strategy(train_data)
            elif strategy_name == 'rotation':
                strategies[strategy_name] = self._rotation_strategy(train_data)
            elif strategy_name == 'gap_balanced':
                strategies[strategy_name] = self._gap_balanced_strategy(train_data)
            elif strategy_name == 'anti_hot':
                strategies[strategy_name] = self._anti_hot_strategy(train_data)
            elif strategy_name == 'anti_cold':
                strategies[strategy_name] = self._anti_cold_strategy(train_data)
            elif strategy_name == 'diversity':
                strategies[strategy_name] = self._diversity_strategy(train_data)
            elif strategy_name == 'cold_moderate':
                strategies[strategy_name] = self._cold_boost_strategy(train_data)
        
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
            print(f"\n检测模式: {pattern}")
            print(f"策略类型: {self.strategy_type}")
            print(f"\n权重配置:")
            for name, weight in self.weights.items():
                print(f"  {name}: {weight}")
            print(f"\n各生肖评分:")
            for zodiac, score in sorted_zodiacs:
                print(f"  {zodiac}: {score:.2f}")
        
        return {
            'top3': [z for z, s in sorted_zodiacs[:3]],
            'top4': [z for z, s in sorted_zodiacs[:4]],
            'top5': [z for z, s in sorted_zodiacs[:5]],
            'all_scores': sorted_zodiacs,
            'pattern': pattern,
            'strategy': self.strategy_type,
            'weights': self.weights.copy()
        }
    
    def update_performance(self, is_hit):
        """更新性能监控"""
        self.recent_performance.append(1 if is_hit else 0)
    
    def check_need_retrain(self):
        """检查是否需要紧急重训练"""
        if len(self.recent_performance) < 3:
            return False
        
        # 最近3期命中率
        recent_3_rate = sum(list(self.recent_performance)[-3:]) / 3
        
        if recent_3_rate < 0.33:  # 低于33%
            return True
        
        return False
    
    def get_performance_stats(self):
        """获取性能统计"""
        if not self.recent_performance:
            return None
        
        total = len(self.recent_performance)
        hits = sum(self.recent_performance)
        rate = hits / total * 100
        
        return {
            'recent_hits': hits,
            'recent_total': total,
            'recent_rate': rate,
            'need_retrain': self.check_need_retrain()
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
            'model': '自适应生肖预测器',
            'version': '3.0',
            'strategy': result['strategy'],
            'pattern': result['pattern'],
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_animal': df.iloc[-1]['animal'],
            'top3': result['top3'],
            'top4': result['top4'],
            'top5': result['top5'],
            'top15_numbers': top_numbers,
            'weights': result['weights']
        }


if __name__ == '__main__':
    print("="*80)
    print("自适应生肖预测器 v3.0 测试")
    print("="*80 + "\n")
    
    predictor = AdaptiveZodiacPredictor(train_window=50, monitor_window=10)
    result = predictor.predict()
    
    print(f"\n{'='*80}")
    print(f"【预测结果】")
    print(f"{'='*80}")
    print(f"模型: {result['model']} v{result['version']}")
    print(f"数据模式: {result['pattern']}")
    print(f"策略类型: {result['strategy']}")
    print(f"数据: 第{result['total_periods']}期 ({result['last_date']})")
    print(f"上期生肖: {result['last_animal']}")
    
    print(f"\nTOP3生肖: {', '.join(result['top3'])}")
    print(f"TOP4生肖: {', '.join(result['top4'])}")
    print(f"TOP5生肖: {', '.join(result['top5'])}")
    
    print(f"\nTOP15号码: {result['top15_numbers']}")
