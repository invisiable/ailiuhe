"""
生肖TOP5new预测器 - 基于统计分析优化版
使用冷热号分析、遗漏值、周期性、趋势等多维度统计方法提升预测准确率

核心特征：
1. 近期热度分析（最近10期）- 25%
2. 中期频率分析（最近30期）- 15%
3. 遗漏值分析（冷号回补）- 25%
4. 周期规律分析 - 15%
5. 趋势分析 - 5%
6. 连续性分析 - 5%
7. 配对模式分析 - 10%
"""
import numpy as np
import pandas as pd
from collections import Counter

class ZodiacTOP5NewPredictor:
    """基于统计分析的生肖TOP5预测器"""
    
    def __init__(self):
        # 12生肖列表
        self.zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        # 分析窗口大小
        self.recent_window = 10    # 最近期分析
        self.hot_window = 30       # 热号分析窗口
        self.warm_window = 60      # 温号分析窗口
        
        # 权重配置（优化后）
        self.weights = {
            'recent_hot': 0.25,         # 近期热号
            'medium_freq': 0.15,        # 中期频率
            'omission': 0.25,           # 遗漏值（冷号回补）
            'cycle_pattern': 0.15,      # 周期规律
            'trend': 0.05,              # 趋势
            'consecutive': 0.05,        # 连续性
            'pair_pattern': 0.10,       # 配对模式
        }
    
    def analyze_frequency(self, history, window):
        """分析指定窗口的出现频率"""
        recent = history[-window:] if len(history) > window else history
        freq = Counter(recent)
        
        # 归一化频率
        total = len(recent)
        freq_normalized = {zodiac: freq.get(zodiac, 0) / total for zodiac in self.zodiacs}
        
        return freq_normalized
    
    def analyze_omission(self, history):
        """分析遗漏值 - 距离上次出现的期数"""
        omission_score = {}
        
        for zodiac in self.zodiacs:
            # 找到该生肖最后一次出现的位置
            last_idx = -1
            for i in range(len(history) - 1, -1, -1):
                if history[i] == zodiac:
                    last_idx = i
                    break
            
            if last_idx == -1:
                omission = len(history)
            else:
                omission = len(history) - 1 - last_idx
            
            # 遗漏值评分：理论上每个生肖平均12期出现一次
            # 遗漏8-20期的给高分（冷号回补）
            if omission <= 3:
                score = 0.1  # 刚出现，不太可能再出
            elif omission <= 7:
                score = 0.3 + (omission - 3) * 0.1  # 逐渐增加
            elif omission <= 15:
                score = 0.7 + (omission - 7) * 0.03  # 最佳区间
            elif omission <= 25:
                score = 0.9 - (omission - 15) * 0.02  # 开始降低
            else:
                score = 0.5  # 超冷号
            
            omission_score[zodiac] = min(1.0, max(0.0, score))
        
        return omission_score
    
    def analyze_trend(self, history, window=40):
        """分析趋势：比较前半段和后半段的频率变化"""
        if len(history) < window:
            window = len(history)
        
        recent = history[-window:]
        half = window // 2
        
        first_half = Counter(recent[:half])
        second_half = Counter(recent[half:])
        
        trend_score = {}
        for zodiac in self.zodiacs:
            freq1 = first_half.get(zodiac, 0) / max(half, 1)
            freq2 = second_half.get(zodiac, 0) / max(half, 1)
            
            # 趋势分数：正值表示上升趋势
            trend_score[zodiac] = freq2 - freq1
        
        # 归一化到0-1
        min_trend = min(trend_score.values())
        max_trend = max(trend_score.values())
        
        if max_trend - min_trend > 0:
            trend_normalized = {
                zodiac: (score - min_trend) / (max_trend - min_trend)
                for zodiac, score in trend_score.items()
            }
        else:
            trend_normalized = {zodiac: 0.5 for zodiac in self.zodiacs}
        
        return trend_normalized
    
    def analyze_cycle(self, history):
        """分析周期性规律"""
        cycle_score = {}
        
        for zodiac in self.zodiacs:
            positions = [i for i, z in enumerate(history) if z == zodiac]
            
            if len(positions) < 2:
                cycle_score[zodiac] = 0.5
                continue
            
            # 计算间隔序列
            intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            
            if len(intervals) == 0:
                cycle_score[zodiac] = 0.5
                continue
            
            # 计算平均间隔和标准差
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals) if len(intervals) > 1 else 0
            
            # 当前距离上次出现的期数
            current_interval = len(history) - 1 - positions[-1]
            
            # 如果当前间隔接近平均间隔，说明可能即将出现
            if std_interval > 0:
                z_score = abs(current_interval - avg_interval) / std_interval
                cycle_score[zodiac] = np.exp(-z_score)
            else:
                diff = abs(current_interval - avg_interval)
                cycle_score[zodiac] = 1.0 if diff < 2 else 0.5
        
        return cycle_score
    
    def analyze_consecutive(self, history, window=20):
        """分析连续性 - 是否有连续出现的模式"""
        consecutive_score = {}
        
        recent = history[-window:] if len(history) > window else history
        
        for zodiac in self.zodiacs:
            # 统计最近是否有连续出现
            max_consecutive = 0
            current_consecutive = 0
            
            for animal in reversed(recent):
                if animal == zodiac:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            # 如果最近有连续出现，给予一定分数
            if max_consecutive >= 2:
                consecutive_score[zodiac] = 0.8
            elif max_consecutive == 1:
                # 检查是否有连续出现的历史
                consecutive_pairs = 0
                for i in range(len(recent) - 1):
                    if recent[i] == zodiac and recent[i+1] == zodiac:
                        consecutive_pairs += 1
                
                consecutive_score[zodiac] = min(0.6, consecutive_pairs * 0.2)
            else:
                consecutive_score[zodiac] = 0.3
        
        return consecutive_score
    
    def analyze_pair_pattern(self, history, window=30):
        """分析配对规律 - 某些生肖倾向于在相近期数出现"""
        pair_score = {}
        
        recent = history[-window:] if len(history) > window else history
        
        # 统计每个生肖前后最常跟随的生肖
        following_pairs = {zodiac: Counter() for zodiac in self.zodiacs}
        
        for i in range(len(recent) - 1):
            current = recent[i]
            next_animal = recent[i + 1]
            following_pairs[current][next_animal] += 1
        
        # 基于最近一期预测下一期
        if len(history) > 0:
            last_animal = history[-1]
            pairs_count = following_pairs[last_animal]
            
            max_count = max(pairs_count.values()) if pairs_count else 0
            
            for zodiac in self.zodiacs:
                count = pairs_count.get(zodiac, 0)
                if max_count > 0:
                    pair_score[zodiac] = count / max_count
                else:
                    pair_score[zodiac] = 0.5
        else:
            pair_score = {zodiac: 0.5 for zodiac in self.zodiacs}
        
        return pair_score
    
    def calculate_composite_score(self, history):
        """计算综合分数"""
        # 各项分析
        recent_hot = self.analyze_frequency(history, self.recent_window)
        medium_freq = self.analyze_frequency(history, self.hot_window)
        omission = self.analyze_omission(history)
        cycle = self.analyze_cycle(history)
        trend = self.analyze_trend(history, window=40)
        consecutive = self.analyze_consecutive(history)
        pair_pattern = self.analyze_pair_pattern(history)
        
        # 加权综合评分
        composite_score = {}
        for zodiac in self.zodiacs:
            score = (
                self.weights['recent_hot'] * recent_hot[zodiac] +
                self.weights['medium_freq'] * medium_freq[zodiac] +
                self.weights['omission'] * omission[zodiac] +
                self.weights['cycle_pattern'] * cycle[zodiac] +
                self.weights['trend'] * trend[zodiac] +
                self.weights['consecutive'] * consecutive[zodiac] +
                self.weights['pair_pattern'] * pair_pattern[zodiac]
            )
            composite_score[zodiac] = score
        
        # 调试信息
        debug_info = {
            'recent_hot': recent_hot,
            'medium_freq': medium_freq,
            'omission': omission,
            'cycle': cycle,
            'trend': trend,
            'consecutive': consecutive,
            'pair_pattern': pair_pattern,
            'composite': composite_score
        }
        
        return composite_score, debug_info
    
    def predict_top5(self, history, debug=False):
        """预测TOP5生肖"""
        if len(history) < 20:
            # 数据太少，使用频率统计
            freq = self.analyze_frequency(history, len(history))
            sorted_zodiacs = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            top5 = [z[0] for z in sorted_zodiacs[:5]]
            
            if debug:
                print("数据不足，使用简单频率统计")
                print(f"TOP5: {top5}")
            
            return top5, {}
        
        # 计算综合分数
        scores, debug_info = self.calculate_composite_score(history)
        
        # 排序获取TOP5
        sorted_zodiacs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top5 = [z[0] for z in sorted_zodiacs[:5]]
        
        if debug:
            print("\n" + "="*80)
            print("生肖TOP5new预测分析")
            print("="*80)
            print(f"\n历史数据: {len(history)}期")
            print(f"\n最近10期: {history[-10:]}")
            
            print("\n各项分析得分（TOP5）:")
            for zodiac in top5:
                print(f"{zodiac}:")
                print(f"  近期热度({self.recent_window}期): {debug_info['recent_hot'][zodiac]:.3f}")
                print(f"  中期频率({self.hot_window}期): {debug_info['medium_freq'][zodiac]:.3f}")
                print(f"  遗漏值分数: {debug_info['omission'][zodiac]:.3f}")
                print(f"  周期规律: {debug_info['cycle'][zodiac]:.3f}")
                print(f"  趋势分数: {debug_info['trend'][zodiac]:.3f}")
                print(f"  连续性: {debug_info['consecutive'][zodiac]:.3f}")
                print(f"  配对模式: {debug_info['pair_pattern'][zodiac]:.3f}")
                print(f"  综合得分: {scores[zodiac]:.3f}")
            
            print(f"\n预测TOP5: {top5}")
            print("="*80)
        
        return top5, debug_info
    
    def predict_from_history(self, history, top_n=5, debug=False):
        """从历史数据预测（兼容接口）"""
        top5, debug_info = self.predict_top5(history, debug=debug)
        
        return {
            'top5': top5[:top_n],
            'selected_model': 'zodiac_top5_new',
            'scores': debug_info.get('composite', {}),
            'debug_info': debug_info
        }
    
    def validate(self, history, test_periods=100, debug=False):
        """验证预测准确率"""
        if len(history) < test_periods + 50:
            test_periods = max(10, len(history) - 50)
        
        hits_top5 = 0
        total_predictions = 0
        
        hit_details = []
        
        start_idx = len(history) - test_periods
        
        for i in range(start_idx, len(history)):
            train_data = history[:i]
            actual = history[i]
            
            top5, _ = self.predict_top5(train_data, debug=False)
            
            hit = actual in top5
            if hit:
                hits_top5 += 1
            
            total_predictions += 1
            
            hit_details.append({
                'period': i + 1,
                'actual': actual,
                'predicted_top5': top5,
                'hit': hit
            })
        
        hit_rate = hits_top5 / total_predictions if total_predictions > 0 else 0
        
        if debug:
            print("\n" + "="*80)
            print(f"生肖TOP5new验证结果 - 最近{test_periods}期")
            print("="*80)
            print(f"TOP5命中率: {hit_rate*100:.2f}% ({hits_top5}/{total_predictions})")
            print("="*80)
        
        return {
            'hit_rate': hit_rate,
            'hits': hits_top5,
            'total': total_predictions,
            'details': hit_details
        }


def main():
    """测试预测器"""
    print("生肖TOP5new预测器 - 统计分析优化版")
    print("="*80)
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"数据加载: {len(df)}期")
    
    all_animals = df['animal'].tolist()
    
    # 创建预测器
    predictor = ZodiacTOP5NewPredictor()
    
    # 验证不同期数
    print("\n" + "="*80)
    print("多期验证测试")
    print("="*80)
    
    for test_periods in [50, 100, 150, 200]:
        result = predictor.validate(all_animals, test_periods=test_periods, debug=False)
        print(f"最近{test_periods:3}期: 命中率 {result['hit_rate']*100:.2f}% ({result['hits']}/{result['total']})")
    
    # 预测下一期
    print("\n下期预测：")
    prediction = predictor.predict_from_history(all_animals, debug=True)
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


if __name__ == "__main__":
    main()
