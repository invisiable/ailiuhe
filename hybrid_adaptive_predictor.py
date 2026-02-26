"""
混合自适应预测器 v3.1
结合固定策略的稳定性和自适应策略的灵活性

核心思路：
1. 默认使用重训练v2.0的固定策略（已验证47%命中率）
2. 检测到数据模式剧变时，自动切换到自适应模式
3. 剧变判断标准：最近10期命中率<30% 或 数据分布异常
"""

import pandas as pd
import numpy as np
from collections import Counter, deque
from retrained_zodiac_predictor import RetrainedZodiacPredictor
from adaptive_zodiac_predictor import AdaptiveZodiacPredictor


class HybridAdaptivePredictor:
    """混合自适应预测器"""
    
    def __init__(self):
        self.zodiacs = ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪']
        self.zodiac_numbers = {
            '鼠': [6, 18, 30, 42], '牛': [2, 14, 26, 38],
            '虎': [4, 16, 28, 40], '兔': [3, 15, 27, 39],
            '龙': [8, 20, 32, 44], '蛇': [1, 13, 25, 37, 49],
            '马': [5, 17, 29, 41], '羊': [7, 19, 31, 43],
            '猴': [11, 23, 35, 47], '鸡': [9, 21, 33, 45],
            '狗': [10, 22, 34, 46], '猪': [12, 24, 36, 48]
        }
        
        # 两个核心预测器
        self.stable_predictor = RetrainedZodiacPredictor()  # 稳定策略
        self.adaptive_predictor = AdaptiveZodiacPredictor(train_window=50)  # 自适应策略
        
        # 性能监控
        self.recent_performance = deque(maxlen=10)
        self.current_mode = 'stable'  # 'stable' 或 'adaptive'
        self.mode_switch_cooldown = 0  # 切换冷却期，避免频繁切换
    
    def detect_data_anomaly(self, animals):
        """检测数据异常（模式剧变）"""
        if len(animals) < 30:
            return False
        
        recent_20 = animals[-20:]
        prev_20 = animals[-40:-20] if len(animals) >= 40 else animals[-20:]
        
        freq_recent = Counter(recent_20)
        freq_prev = Counter(prev_20)
        
        # 指标1: 消失生肖数量突增
        disappeared_recent = sum(1 for z in self.zodiacs if freq_recent.get(z, 0) == 0)
        disappeared_prev = sum(1 for z in self.zodiacs if freq_prev.get(z, 0) == 0)
        
        if disappeared_recent >= 3 and disappeared_recent > disappeared_prev:
            return True
        
        # 指标2: 爆发生肖（单生肖占比>20%）
        max_ratio_recent = max(freq_recent.values()) / 20 if freq_recent else 0
        if max_ratio_recent >= 0.25:  # 25%以上
            return True
        
        # 指标3: 分布剧变（前后差异大）
        distribution_change = 0
        for zodiac in self.zodiacs:
            diff = abs(freq_recent.get(zodiac, 0) - freq_prev.get(zodiac, 0))
            distribution_change += diff
        
        if distribution_change >= 16:  # 平均每个生肖变化>1.3次
            return True
        
        return False
    
    def check_performance_drop(self):
        """检查最近性能是否下降"""
        if len(self.recent_performance) < 5:
            return False
        
        # 最近5期命中率
        recent_5_rate = sum(list(self.recent_performance)[-5:]) / 5
        
        if recent_5_rate < 0.30:  # 低于30%
            return True
        
        return False
    
    def decide_mode(self, animals):
        """决定使用哪种模式"""
        # 冷却期内不切换
        if self.mode_switch_cooldown > 0:
            self.mode_switch_cooldown -= 1
            return self.current_mode
        
        # 检测异常
        has_anomaly = self.detect_data_anomaly(animals)
        has_perf_drop = self.check_performance_drop()
        
        # 切换逻辑
        if self.current_mode == 'stable':
            # 稳定模式下，如果检测到异常或性能下降，切换到自适应
            if has_anomaly or has_perf_drop:
                self.current_mode = 'adaptive'
                self.mode_switch_cooldown = 5  # 切换后5期内不再切换
                return 'adaptive'
        else:  # adaptive
            # 自适应模式下，如果数据恢复正常且性能提升，切回稳定
            if not has_anomaly and not has_perf_drop and len(self.recent_performance) >= 5:
                recent_5_rate = sum(list(self.recent_performance)[-5:]) / 5
                if recent_5_rate >= 0.40:  # 最近表现良好
                    # 检查数据是否恢复稳定
                    recent_20 = animals[-20:]
                    diversity = len(Counter(recent_20)) / 12
                    if diversity >= 0.80:  # 多样性恢复
                        self.current_mode = 'stable'
                        self.mode_switch_cooldown = 10  # 切回后10期内不再切换
                        return 'stable'
        
        return self.current_mode
    
    def predict_from_history(self, animals, top_n=4, debug=False):
        """混合预测"""
        # 决定使用哪种模式
        mode = self.decide_mode(animals)
        
        if mode == 'stable':
            result = self.stable_predictor.predict_from_history(animals, top_n, debug=False)
            result['predictor'] = '稳定策略(重训练v2.0)'
            result['mode'] = 'stable'
        else:
            result = self.adaptive_predictor.predict_from_history(animals, top_n, debug=False)
            result['predictor'] = f'自适应策略({result.get("strategy", "未知")})'
            result['mode'] = 'adaptive'
        
        if debug:
            print(f"\n当前模式: {mode}")
            print(f"使用预测器: {result['predictor']}")
            if mode == 'adaptive':
                print(f"检测到数据模式: {result.get('pattern', '未知')}")
        
        return result
    
    def update_performance(self, is_hit):
        """更新性能监控"""
        self.recent_performance.append(1 if is_hit else 0)
        # 同时更新自适应预测器的监控
        self.adaptive_predictor.update_performance(is_hit)
    
    def get_status(self):
        """获取当前状态"""
        if not self.recent_performance:
            return {
                'mode': self.current_mode,
                'recent_rate': 0,
                'cooldown': self.mode_switch_cooldown
            }
        
        total = len(self.recent_performance)
        hits = sum(self.recent_performance)
        rate = hits / total * 100
        
        return {
            'mode': self.current_mode,
            'recent_hits': hits,
            'recent_total': total,
            'recent_rate': rate,
            'cooldown': self.mode_switch_cooldown
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
        
        status = self.get_status()
        
        return {
            'model': '混合自适应预测器',
            'version': '3.1',
            'predictor': result['predictor'],
            'mode': result['mode'],
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_animal': df.iloc[-1]['animal'],
            'top3': result['top3'],
            'top4': result['top4'],
            'top5': result['top5'],
            'top15_numbers': top_numbers,
            'status': status
        }


if __name__ == '__main__':
    print("="*80)
    print("混合自适应预测器 v3.1 测试")
    print("="*80 + "\n")
    
    predictor = HybridAdaptivePredictor()
    result = predictor.predict()
    
    print(f"\n{'='*80}")
    print(f"【预测结果】")
    print(f"{'='*80}")
    print(f"模型: {result['model']} v{result['version']}")
    print(f"当前模式: {result['mode']}")
    print(f"使用预测器: {result['predictor']}")
    print(f"数据: 第{result['total_periods']}期 ({result['last_date']})")
    print(f"上期生肖: {result['last_animal']}")
    
    print(f"\nTOP3生肖: {', '.join(result['top3'])}")
    print(f"TOP4生肖: {', '.join(result['top4'])}")
    print(f"TOP5生肖: {', '.join(result['top5'])}")
    
    print(f"\nTOP15号码: {result['top15_numbers']}")
    
    print(f"\n状态监控:")
    status = result['status']
    print(f"  当前模式: {status['mode']}")
    print(f"  切换冷却: {status['cooldown']}期")
    if status.get('recent_total'):
        print(f"  最近{status['recent_total']}期: {status['recent_rate']:.1f}%")
