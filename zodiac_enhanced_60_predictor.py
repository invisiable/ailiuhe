"""
增强版生肖预测器 - 融合多模型
目标：60%成功率

融合策略：
1. 生肖智能预测（52%基础）
2. 统计分布模型（泊松、正态、卡方）
3. Top15频率分析
4. 集成投票机制
5. 共识加权
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from scipy.stats import poisson, norm, chi2
import warnings
warnings.filterwarnings('ignore')

# 导入现有预测器
from zodiac_balanced_smart import ZodiacBalancedSmart


class ZodiacEnhanced60Predictor:
    """增强版生肖预测器 - 目标60%成功率"""
    
    def __init__(self):
        # 基础生肖预测器
        self.zodiac_base = ZodiacBalancedSmart()
        
        # 生肖映射
        self.zodiac_numbers = {
            '鼠': [4, 16, 28, 40],
            '牛': [5, 17, 29, 41],
            '虎': [6, 18, 30, 42],
            '兔': [7, 19, 31, 43],
            '龙': [8, 20, 32, 44],
            '蛇': [9, 21, 33, 45],
            '马': [10, 22, 34, 46],
            '羊': [11, 23, 35, 47],
            '猴': [12, 24, 36, 48],
            '鸡': [1, 13, 25, 37, 49],
            '狗': [2, 14, 26, 38],
            '猪': [3, 15, 27, 39]
        }
        
        self.number_to_zodiac = {}
        for zodiac, nums in self.zodiac_numbers.items():
            for n in nums:
                self.number_to_zodiac[n] = zodiac
    
    def predict_top5(self, numbers=None, recent_periods=100):
        """预测Top5生肖 - 增强版"""
        if numbers is None:
            # 读取数据
            df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
            numbers = df['number'].values
        
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        if len(numbers_list) < 30:
            return ['鸡', '鼠', '龙', '马', '狗']
        
        # 综合评分
        zodiac_scores = defaultdict(float)
        
        # 方法1：基础生肖预测（25%）
        try:
            base_top6 = self.zodiac_base.predict_top6(recent_periods=recent_periods)
            for i, zodiac in enumerate(base_top6, 1):
                score = (7 - i) * 10
                zodiac_scores[zodiac] += score * 0.25
        except:
            pass
        
        # 方法2：统计分布分析（35%）- 最科学的方法，提高权重
        stat_scores = self._statistical_analysis(numbers_list)
        for zodiac, score in stat_scores.items():
            zodiac_scores[zodiac] += score * 0.35
        
        # 方法3：频率回补分析（25%）
        freq_scores = self._frequency_rebound_analysis(numbers_list)
        for zodiac, score in freq_scores.items():
            zodiac_scores[zodiac] += score * 0.25
        
        # 方法4：趋势分析（15%）
        trend_scores = self._trend_analysis(numbers_list)
        for zodiac, score in trend_scores.items():
            zodiac_scores[zodiac] += score * 0.15
        
        # 方法5：最近期共识加权 - 多维度交叉验证
        recent_30 = numbers_list[-30:]
        recent_20 = numbers_list[-20:]
        recent_10 = numbers_list[-10:]
        
        # 统计最近30期生肖分布
        recent_zodiac_count = defaultdict(int)
        for num in recent_30:
            z = self.number_to_zodiac.get(num)
            if z:
                recent_zodiac_count[z] += 1
        
        # 给出现1-2次的生肖加分（处于适中状态）
        for zodiac, count in recent_zodiac_count.items():
            if count == 1:
                zodiac_scores[zodiac] += 25
            elif count == 2:
                zodiac_scores[zodiac] += 20
            elif count >= 4:  # 出现太多次，降权
                zodiac_scores[zodiac] -= 15
        
        # 给最近10期未出现的生肖额外加分
        recent_10_zodiacs = set()
        for num in recent_10:
            z = self.number_to_zodiac.get(num)
            if z:
                recent_10_zodiacs.add(z)
        
        for zodiac in self.zodiac_numbers.keys():
            if zodiac not in recent_10_zodiacs:
                zodiac_scores[zodiac] += 30
        
        # 排序返回Top5
        sorted_zodiacs = sorted(zodiac_scores.items(), key=lambda x: x[1], reverse=True)
        return [zodiac for zodiac, _ in sorted_zodiacs[:5]]
    
    def _statistical_analysis(self, numbers):
        """统计分布分析"""
        recent_100 = numbers[-100:] if len(numbers) >= 100 else numbers
        recent_50 = numbers[-50:]
        
        # 统计每个生肖的出现次数
        zodiac_freq = defaultdict(int)
        for num in recent_100:
            zodiac = self.number_to_zodiac.get(num)
            if zodiac:
                zodiac_freq[zodiac] += 1
        
        scores = {}
        
        # 1. 泊松分布分析
        total_periods = len(recent_100)
        expected_freq = total_periods / 12  # 每个生肖期望出现次数
        
        for zodiac in self.zodiac_numbers.keys():
            observed = zodiac_freq.get(zodiac, 0)
            
            # 如果观察值低于期望，说明该生肖"欠债"
            if observed < expected_freq:
                debt_ratio = (expected_freq - observed) / expected_freq
                scores[zodiac] = debt_ratio * 100
            else:
                # 已经超额出现
                excess_ratio = (observed - expected_freq) / expected_freq
                scores[zodiac] = max(0, 50 - excess_ratio * 30)
        
        # 2. 卡方检验加分
        observed_list = [zodiac_freq.get(z, 0) for z in self.zodiac_numbers.keys()]
        expected_list = [expected_freq] * 12
        
        try:
            chi2_stat, p_value = stats.chisquare(observed_list, expected_list)
            
            # 如果分布不均匀，给低频生肖加分
            if p_value < 0.05:
                for zodiac in self.zodiac_numbers.keys():
                    obs = zodiac_freq.get(zodiac, 0)
                    if obs < expected_freq:
                        scores[zodiac] += 20
        except:
            pass
        
        # 3. 正态分布异常检测
        if zodiac_freq:
            mean_freq = np.mean(list(zodiac_freq.values()))
            std_freq = np.std(list(zodiac_freq.values()))
            
            for zodiac in self.zodiac_numbers.keys():
                obs = zodiac_freq.get(zodiac, 0)
                z_score = abs(obs - mean_freq) / (std_freq + 1e-10)
                
                # 异常值（远离均值）给予加分
                if z_score > 1.5 and obs < mean_freq:
                    scores[zodiac] += 30
        
        return scores
    
    def _frequency_rebound_analysis(self, numbers):
        """频率回补分析"""
        recent_100 = numbers[-100:] if len(numbers) >= 100 else numbers
        recent_50 = numbers[-50:]
        recent_30 = numbers[-30:]
        recent_20 = numbers[-20:]
        recent_10 = numbers[-10:]
        
        scores = defaultdict(float)
        
        # 统计各周期的生肖频率
        freq_100 = self._count_zodiac_freq(recent_100)
        freq_50 = self._count_zodiac_freq(recent_50)
        freq_30 = self._count_zodiac_freq(recent_30)
        freq_20 = self._count_zodiac_freq(recent_20)
        freq_10 = self._count_zodiac_freq(recent_10)
        
        for zodiac in self.zodiac_numbers.keys():
            f100 = freq_100.get(zodiac, 0)
            f50 = freq_50.get(zodiac, 0)
            f30 = freq_30.get(zodiac, 0)
            f20 = freq_20.get(zodiac, 0)
            f10 = freq_10.get(zodiac, 0)
            
            # 策略1：长期出现，短期消失 -> 回补机会
            if f100 >= 5 and f30 <= 1:
                scores[zodiac] += 90
            elif f50 >= 3 and f20 == 0:
                scores[zodiac] += 80
            
            # 策略2：完全未出现 -> 超高分
            if f100 == 0:
                scores[zodiac] += 100
            elif f50 == 0:
                scores[zodiac] += 95
            elif f30 == 0:
                scores[zodiac] += 85
            elif f20 == 0:
                scores[zodiac] += 70
            
            # 策略3：适中频率加分
            if 2 <= f30 <= 4:
                scores[zodiac] += 55
            
            # 策略4：最近10期强惩罚
            if f10 >= 3:
                scores[zodiac] -= 50
            elif f10 >= 2:
                scores[zodiac] -= 35
        
        return scores
    
    def _trend_analysis(self, numbers):
        """趋势分析"""
        recent_20 = numbers[-20:]
        recent_10 = numbers[-10:]
        
        scores = defaultdict(float)
        
        # 1. 数字区间趋势
        avg_recent = np.mean(recent_10)
        
        for zodiac, nums in self.zodiac_numbers.items():
            avg_zodiac = np.mean(nums)
            
            # 如果最近趋势接近该生肖的平均值
            distance = abs(avg_zodiac - avg_recent)
            if distance <= 10:
                scores[zodiac] += 60
            elif distance <= 20:
                scores[zodiac] += 40
        
        # 2. 间隔分析
        last_seen = {}
        for i, num in enumerate(recent_20):
            zodiac = self.number_to_zodiac.get(num)
            if zodiac and zodiac not in last_seen:
                last_seen[zodiac] = len(recent_20) - i
        
        for zodiac in self.zodiac_numbers.keys():
            gap = last_seen.get(zodiac, 20)
            
            # 最佳间隔区间
            if 3 <= gap <= 10:
                scores[zodiac] += 70
            elif 10 < gap <= 15:
                scores[zodiac] += 50
            elif gap > 15:
                scores[zodiac] += 60
        
        return scores
    
    def _count_zodiac_freq(self, numbers):
        """统计生肖频率"""
        freq = defaultdict(int)
        for num in numbers:
            zodiac = self.number_to_zodiac.get(num)
            if zodiac:
                freq[zodiac] += 1
        return freq
    
    def predict_numbers(self, numbers=None, recent_periods=100, top_n=20):
        """预测具体数字 - 基于生肖预测"""
        # 获取Top5生肖
        top5_zodiacs = self.predict_top5(numbers, recent_periods)
        
        if numbers is None:
            df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
            numbers = df['number'].values
        
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        # 为每个生肖的数字打分
        number_scores = defaultdict(float)
        
        # 生肖权重
        for i, zodiac in enumerate(top5_zodiacs, 1):
            zodiac_weight = (6 - i) * 15
            nums = self.zodiac_numbers[zodiac]
            
            for num in nums:
                number_scores[num] += zodiac_weight
        
        # 额外的数字级别分析
        recent_50 = numbers_list[-50:]
        recent_30 = numbers_list[-30:]
        recent_10 = numbers_list[-10:]
        recent_5 = set(numbers_list[-5:])
        
        freq_50 = Counter(recent_50)
        freq_30 = Counter(recent_30)
        
        for num in range(1, 50):
            # 频率分析
            f50 = freq_50.get(num, 0)
            f30 = freq_30.get(num, 0)
            
            if f50 >= 2 and f30 == 0:
                number_scores[num] += 30
            elif f30 == 0:
                number_scores[num] += 20
            
            # 惩罚最近5期
            if num in recent_5:
                number_scores[num] *= 0.2
        
        # 排序返回Top N
        sorted_nums = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:top_n]]


def validate():
    """验证预测器性能"""
    print("=" * 80)
    print("增强版生肖预测器 - 目标60%成功率验证")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    predictor = ZodiacEnhanced60Predictor()
    
    # 100期回测
    test_periods = 100
    hits_top5 = 0
    hits_top20 = 0
    total = 0
    
    print("\n开始100期回测验证...")
    print(f"{'期数':<8} {'实际':<6} {'生肖':<6} {'Top5':<8} {'Top20':<8}")
    print("-" * 80)
    
    for i in range(len(numbers) - test_periods, len(numbers)):
        actual_num = numbers[i]
        actual_zodiac = predictor.number_to_zodiac.get(actual_num, '未知')
        history = numbers[:i]
        
        if len(history) < 30:
            continue
        
        # 预测Top5生肖
        top5_zodiacs = predictor.predict_top5(history, recent_periods=100)
        
        # 预测Top20数字
        top20_numbers = predictor.predict_numbers(history, recent_periods=100, top_n=20)
        
        # 检查命中
        hit_top5 = actual_zodiac in top5_zodiacs
        hit_top20 = actual_num in top20_numbers
        
        if hit_top5:
            hits_top5 += 1
        if hit_top20:
            hits_top20 += 1
        
        total += 1
        
        status_5 = "Y" if hit_top5 else "N"
        status_20 = "Y" if hit_top20 else "N"
        
        if i % 10 == 0 or hit_top5:
            print(f"第{i+1:<5}期 {actual_num:<6} {actual_zodiac:<6} {status_5:<8} {status_20:<8}")
    
    # 统计结果
    print("-" * 80)
    print(f"\n验证完成!")
    print(f"测试期数: {total} 期")
    print(f"Top5生肖命中: {hits_top5} 期")
    print(f"Top5生肖成功率: {hits_top5/total*100:.1f}%")
    print(f"Top20数字命中: {hits_top20} 期")
    print(f"Top20数字成功率: {hits_top20/total*100:.1f}%")
    
    # 预测下一期
    print("\n【下一期预测】")
    top5 = predictor.predict_top5(numbers)
    top20 = predictor.predict_numbers(numbers, top_n=20)
    
    print(f"Top5生肖: {top5}")
    print(f"Top20数字: {top20}")
    
    # 显示Top5生肖对应的所有数字
    print("\n【Top5生肖数字】")
    for i, zodiac in enumerate(top5, 1):
        nums = predictor.zodiac_numbers[zodiac]
        print(f"  {i}. {zodiac}: {nums}")


if __name__ == '__main__':
    validate()
