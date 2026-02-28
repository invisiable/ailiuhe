"""
精准生肖TOP4预测器 - 风险控制优化版
目标：将最大连续不中期数降低到4期以内
"""

import numpy as np
from collections import Counter, defaultdict, deque


class PreciseZodiacTop4Predictor:
    """精准生肖TOP4预测器 - 专注于降低连续不中风险"""
    
    def __init__(self):
        """初始化预测器"""
        # 12生肖定义
        self.all_zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        # 性能跟踪
        self.consecutive_misses = 0
        self.recent_predictions = deque(maxlen=20)  # 最近20期预测
        self.recent_actuals = deque(maxlen=20)  # 最近20期实际
        self.miss_patterns = defaultdict(int)  # 记录哪些生肖经常预测错误
        
    def predict_top4(self, animals):
        """
        预测TOP4生肖 - 优化版（简化策略，强调趋势）
        
        Args:
            animals: 历史生肖列表
            
        Returns:
            list: TOP4生肖列表
        """
        if len(animals) < 10:
            # 数据不足时使用长期频率
            return self._get_top_by_frequency(animals, 4)
        
        # 简化策略：强调最近趋势和频率
        scores = defaultdict(float)
        
        # 策略1: 短期热度（最近20期）- 50%权重
        recent_20 = animals[-20:]
        freq_20 = Counter(recent_20)
        for zodiac in self.all_zodiacs:
            scores[zodiac] += (freq_20.get(zodiac, 0) / 20) * 0.50
        
        # 策略2: 中期热度（最近50期）- 30%权重
        if len(animals) >= 50:
            recent_50 = animals[-50:]
            freq_50 = Counter(recent_50)
            for zodiac in self.all_zodiacs:
                scores[zodiac] += (freq_50.get(zodiac, 0) / 50) * 0.30
        else:
            # 用全历史代替
            freq_all = Counter(animals)
            for zodiac in self.all_zodiacs:
                scores[zodiac] += (freq_all.get(zodiac, 0) / len(animals)) * 0.30
        
        # 策略3: 避免刚出现的（降低连续重复概率）- 20%权重
        last_zodiac = animals[-1] if animals else None
        last_2_zodiacs = set(animals[-2:]) if len(animals) >= 2 else set()
        
        for zodiac in self.all_zodiacs:
            if zodiac == last_zodiac:
                # 刚出现过，大幅降权
                scores[zodiac] *= 0.3
            elif zodiac in last_2_zodiacs:
                # 最近2期出现过，中度降权
                scores[zodiac] *= 0.6
        
        # 按总分排序并选择TOP4
        sorted_zodiacs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top4 = [zodiac for zodiac, _ in sorted_zodiacs[:4]]
        
        # 确保返回4个不同的生肖
        if len(top4) < 4:
            remaining = [z for z in self.all_zodiacs if z not in top4]
            top4.extend(remaining[:4 - len(top4)])
        
        return top4[:4]
    
    def _method_precision_frequency(self, animals):
        """
        多窗口精准频率分析
        融合短期和长期频率，提高预测精度
        """
        scores = defaultdict(float)
        
        # 长期窗口（最近50期或全部）
        long_window = animals[-50:] if len(animals) > 50 else animals
        long_freq = Counter(long_window)
        long_total = len(long_window)
        
        # 短期窗口（最近15期）
        short_window = animals[-15:] if len(animals) >= 15 else animals[-10:]
        short_freq = Counter(short_window)
        short_total = len(short_window)
        
        # 融合长短期频率
        for zodiac in self.all_zodiacs:
            long_rate = long_freq.get(zodiac, 0) / long_total
            short_rate = short_freq.get(zodiac, 0) / short_total
            
            # 短期权重更高（60% vs 40%）
            scores[zodiac] = short_rate * 0.60 + long_rate * 0.40
        
        # 归一化到0-1
        max_score = max(scores.values()) if scores else 1
        if max_score > 0:
            for zodiac in scores:
                scores[zodiac] /= max_score
        
        return scores
    
    def _method_zone_concentration(self, animals):
        """
        区域集中度优化
        分析生肖在不同区域的分布，选择热门区域
        """
        scores = defaultdict(float)
        
        # 将12生肖分为3个区域（每区4个）
        zones = {
            'zone1': ['鼠', '牛', '虎', '兔'],
            'zone2': ['龙', '蛇', '马', '羊'],
            'zone3': ['猴', '鸡', '狗', '猪']
        }
        
        # 分析最近30期的区域热度
        recent = animals[-30:] if len(animals) >= 30 else animals
        zone_counts = defaultdict(int)
        
        for animal in recent:
            for zone_name, zone_animals in zones.items():
                if animal in zone_animals:
                    zone_counts[zone_name] += 1
        
        # 找出最热门的2个区域
        sorted_zones = sorted(zone_counts.items(), key=lambda x: x[1], reverse=True)
        hot_zones = [zone for zone, _ in sorted_zones[:2]]
        
        # 给热门区域的生肖加分
        for zone_name in hot_zones:
            for zodiac in zones[zone_name]:
                scores[zodiac] = 1.0
        
        # 其他区域降权
        for zone_name in zones:
            if zone_name not in hot_zones:
                for zodiac in zones[zone_name]:
                    if zodiac not in scores:
                        scores[zodiac] = 0.3
        
        return scores
    
    def _method_gap_analysis(self, animals):
        """
        间隔分析
        优先选择合适间隔（3-10期）的生肖
        """
        scores = defaultdict(float)
        
        # 计算每个生肖距离上次出现的间隔
        for zodiac in self.all_zodiacs:
            # 找到最后一次出现的位置
            last_pos = -1
            for i in range(len(animals) - 1, -1, -1):
                if animals[i] == zodiac:
                    last_pos = i
                    break
            
            if last_pos == -1:
                # 从未出现过，给中等分数
                gap = 999
                scores[zodiac] = 0.5
            else:
                gap = len(animals) - last_pos
                
                # 最佳间隔：3-10期
                if 3 <= gap <= 10:
                    scores[zodiac] = 1.0
                elif 2 <= gap <= 15:
                    scores[zodiac] = 0.7
                elif gap == 1:
                    # 刚出现过，降低权重
                    scores[zodiac] = 0.2
                else:
                    # 间隔太长或太短
                    scores[zodiac] = 0.4
        
        return scores
    
    def _method_avoid_recent_misses(self, animals):
        """
        避免历史错误生肖
        学习哪些生肖经常被错误预测，降低其权重
        """
        scores = defaultdict(lambda: 1.0)
        
        if len(self.recent_predictions) < 5:
            return scores
        
        # 统计最近预测中经常出现但实际不中的生肖
        for i in range(len(self.recent_predictions)):
            predicted = self.recent_predictions[i]
            actual = self.recent_actuals[i]
            
            # 记录预测错误的生肖
            if actual not in predicted:
                for zodiac in predicted:
                    if zodiac != actual:
                        self.miss_patterns[zodiac] += 1
        
        # 降低经常预测错误生肖的权重
        if self.miss_patterns:
            max_misses = max(self.miss_patterns.values())
            for zodiac in self.all_zodiacs:
                miss_count = self.miss_patterns.get(zodiac, 0)
                if miss_count > 0 and max_misses > 0:
                    # 错误次数越多，权重越低
                    penalty = miss_count / max_misses
                    scores[zodiac] = 1.0 - (penalty * 0.5)  # 最多降50%
        
        return scores
    
    def _get_top_by_frequency(self, animals, n):
        """
        简单频率法（数据不足时使用）
        """
        freq = Counter(animals)
        # 补充未出现过的生肖
        for zodiac in self.all_zodiacs:
            if zodiac not in freq:
                freq[zodiac] = 0
        
        top_n = [zodiac for zodiac, _ in freq.most_common(n)]
        return top_n
    
    def update_performance(self, predicted, actual):
        """
        更新性能跟踪
        
        Args:
            predicted: 预测的TOP4生肖列表
            actual: 实际开出的生肖
        """
        # 记录预测和实际结果
        self.recent_predictions.append(predicted)
        self.recent_actuals.append(actual)
        
        # 更新连续不中计数
        if actual in predicted:
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
    
    def get_performance_stats(self):
        """获取性能统计"""
        if not self.recent_actuals:
            return {
                'recent_hit_rate': 0,
                'consecutive_misses': self.consecutive_misses,
                'total_predictions': 0
            }
        
        hits = sum(1 for i in range(len(self.recent_actuals)) 
                   if self.recent_actuals[i] in self.recent_predictions[i])
        total = len(self.recent_actuals)
        
        return {
            'recent_hit_rate': hits / total if total > 0 else 0,
            'consecutive_misses': self.consecutive_misses,
            'total_predictions': total,
            'recent_hits': hits
        }


def test_precise_predictor():
    """测试精准预测器"""
    import pandas as pd
    
    print("="*70)
    print("精准生肖TOP4预测器测试")
    print("="*70 + "\n")
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    predictor = PreciseZodiacTop4Predictor()
    
    # 测试最近10期
    test_periods = 10
    start_idx = len(df) - test_periods
    
    print(f"测试最近{test_periods}期:\n")
    
    for i in range(start_idx, len(df)):
        period = i + 1
        history = df['animal'].iloc[:i].tolist()
        
        # 预测
        top4 = predictor.predict_top4(history)
        
        # 实际
        actual = df.iloc[i]['animal']
        is_hit = actual in top4
        
        # 更新性能
        predictor.update_performance(top4, actual)
        
        # 显示结果
        status = "✓ 命中" if is_hit else "✗ 未中"
        print(f"第{period}期: 预测{top4} | 实际:{actual} | {status}")
    
    # 显示统计
    stats = predictor.get_performance_stats()
    print(f"\n命中率: {stats['recent_hit_rate']*100:.2f}%")
    print(f"当前连续不中: {stats['consecutive_misses']}期")


if __name__ == '__main__':
    test_precise_predictor()
