"""
Top15 + 生肖混合预测器
策略：结合生肖预测(52%成功率)和统计分析，提升Top15到60%+
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from zodiac_balanced_smart import ZodiacBalancedSmart


class Top15ZodiacHybridPredictor:
    """Top15生肖混合预测器"""
    
    def __init__(self):
        self.zodiac_predictor = ZodiacBalancedSmart()
        
        # 生肖到号码映射
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
        
        # 反向映射
        self.number_to_zodiac = {}
        for zodiac, nums in self.zodiac_numbers.items():
            for n in nums:
                self.number_to_zodiac[n] = zodiac
    
    def predict(self, numbers):
        """核心预测方法 - 生肖+统计混合"""
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        if len(numbers_list) < 30:
            return list(range(1, 16))
        
        # ========== 第一步：生肖预测 (40%权重) ==========
        zodiac_top5 = self.zodiac_predictor.predict_top5(recent_periods=100)
        
        # 生肖对应的所有号码
        zodiac_candidates = []
        for zodiac in zodiac_top5:
            zodiac_candidates.extend(self.zodiac_numbers[zodiac])
        
        # ========== 第二步：统计分析 (60%权重) ==========
        recent_100 = numbers_list[-100:] if len(numbers_list) >= 100 else numbers_list
        recent_50 = numbers_list[-50:]
        recent_30 = numbers_list[-30:]
        recent_20 = numbers_list[-20:]
        recent_10 = numbers_list[-10:]
        recent_5 = set(numbers_list[-5:])
        
        scores = defaultdict(float)
        
        # 2.1 频率分析
        freq_100 = Counter(recent_100)
        freq_50 = Counter(recent_50)
        freq_30 = Counter(recent_30)
        freq_20 = Counter(recent_20)
        freq_10 = Counter(recent_10)
        
        for n in range(1, 50):
            scores[n] += freq_100.get(n, 0) * 0.5
            scores[n] += freq_50.get(n, 0) * 1.0
            scores[n] += freq_30.get(n, 0) * 1.8
            scores[n] += freq_20.get(n, 0) * 2.5
            scores[n] += freq_10.get(n, 0) * 1.5
        
        # 2.2 间隔分析
        last_seen = {}
        for i, n in enumerate(recent_100):
            last_seen[n] = len(recent_100) - i
        
        for n in range(1, 50):
            gap = last_seen.get(n, 100)
            
            if 3 <= gap <= 15:
                scores[n] += 12.0
            elif 16 <= gap <= 25:
                scores[n] += 10.0
            elif 26 <= gap <= 35:
                scores[n] += 8.0
            elif gap > 35:
                scores[n] += 9.0
            elif gap == 2:
                scores[n] += 3.0
        
        # 2.3 周期性
        for period in range(3, 16):
            if len(numbers_list) > period:
                scores[numbers_list[-period]] += 3.0
            if len(numbers_list) > period * 2:
                scores[numbers_list[-period * 2]] += 2.0
        
        # ========== 第三步：生肖加权 (关键!) ==========
        # 生肖预测的号码获得大幅加分
        for n in zodiac_candidates:
            # 根据生肖排名给予不同加分
            zodiac = self.number_to_zodiac.get(n)
            if zodiac:
                zodiac_rank = zodiac_top5.index(zodiac) + 1
                if zodiac_rank == 1:
                    scores[n] += 25.0  # 第1生肖
                elif zodiac_rank == 2:
                    scores[n] += 20.0  # 第2生肖
                elif zodiac_rank == 3:
                    scores[n] += 16.0  # 第3生肖
                elif zodiac_rank == 4:
                    scores[n] += 12.0  # 第4生肖
                else:
                    scores[n] += 8.0   # 第5生肖
        
        # ========== 第四步：热号补充策略 ==========
        # 100期超热但10期未出现的号码
        top_hot = [n for n, _ in freq_100.most_common(20)]
        for n in top_hot:
            if n not in recent_10:
                scores[n] += 5.0
        
        # ========== 第五步：同生肖协同效应 ==========
        # 如果某个生肖最近出现过，其他号码也可能出现
        recent_zodiacs = set()
        for n in recent_10:
            if n in self.number_to_zodiac:
                recent_zodiacs.add(self.number_to_zodiac[n])
        
        for zodiac in recent_zodiacs:
            if zodiac in zodiac_top5:
                for n in self.zodiac_numbers[zodiac]:
                    if n not in recent_5:
                        scores[n] += 3.0
        
        # ========== 最终调整 ==========
        # 最近5期降权
        for n in recent_5:
            if n in scores:
                scores[n] *= 0.45
        
        # 基础分
        for n in range(1, 50):
            if n not in scores:
                scores[n] = 0.01
        
        # 排序返回Top15
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:15]]
    
    def get_analysis(self, numbers):
        """获取详细分析"""
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        # 生肖预测
        zodiac_top5 = self.zodiac_predictor.predict_top5(recent_periods=100)
        
        # Top15预测
        top15 = self.predict(numbers)
        
        # 分析Top15中的生肖分布
        zodiac_dist = defaultdict(list)
        for n in top15:
            if n in self.number_to_zodiac:
                zodiac = self.number_to_zodiac[n]
                zodiac_dist[zodiac].append(n)
        
        return {
            'top15': top15,
            'zodiac_top5': zodiac_top5,
            'zodiac_distribution': dict(zodiac_dist),
            'recent_10': numbers_list[-10:],
            'strategy': 'Top15生肖混合预测 (生肖52% + 统计分析)'
        }


def validate_hybrid():
    """验证混合预测器"""
    print("=" * 80)
    print("Top15 + 生肖混合预测器 - 100期回测验证")
    print("=" * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    predictor = Top15ZodiacHybridPredictor()
    
    test_periods = min(100, len(numbers) - 50)
    hits = 0
    total = 0
    
    details = []
    
    print("\n期数\t实际\t命中\t预测生肖")
    print("-" * 60)
    
    for i in range(len(numbers) - test_periods, len(numbers)):
        actual = numbers[i]
        history = numbers[:i]
        
        if len(history) < 30:
            continue
        
        top15 = predictor.predict(history)
        hit = actual in top15
        
        if hit:
            hits += 1
        
        # 获取生肖预测
        zodiac_top5 = predictor.zodiac_predictor.predict_top5(recent_periods=100)
        
        mark = 'V' if hit else 'X'
        print(f"{i+1}\t{actual}\t{mark}\t{zodiac_top5[:3]}")
        
        total += 1
        details.append({
            'period': i+1,
            'actual': actual,
            'hit': hit,
            'top15': top15,
            'zodiac_top5': zodiac_top5
        })
    
    rate = hits / total * 100 if total > 0 else 0
    
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    
    print(f"\n总期数: {total}")
    print(f"命中数: {hits}")
    print(f"成功率: {rate:.1f}%")
    
    if rate >= 60:
        print(f"\n*** [成功] 达成60%目标! ***")
    elif rate >= 55:
        print(f"\n[接近目标] 还差 {60-rate:.1f}%")
    elif rate >= 50:
        print(f"\n[良好] 达到50%以上")
    else:
        print(f"\n[需优化] 还差 {60-rate:.1f}%")
    
    # 最近20期详情
    print("\n最近20期表现:")
    recent_20 = details[-20:]
    recent_hits = sum(1 for d in recent_20 if d['hit'])
    print(f"  命中: {recent_hits}/20 = {recent_hits/20*100:.1f}%")
    
    # 保存结果
    df_result = pd.DataFrame(details)
    df_result.to_csv('top15_zodiac_hybrid_validation.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: top15_zodiac_hybrid_validation.csv")
    
    return rate


def test_prediction():
    """测试当前预测"""
    print("\n" + "=" * 80)
    print("当前预测示例")
    print("=" * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    predictor = Top15ZodiacHybridPredictor()
    analysis = predictor.get_analysis(numbers)
    
    print(f"\n最近10期: {analysis['recent_10']}")
    print(f"\n生肖Top5预测: {analysis['zodiac_top5']}")
    print(f"\nTop15预测: {analysis['top15']}")
    
    print(f"\nTop15中的生肖分布:")
    for zodiac, nums in analysis['zodiac_distribution'].items():
        print(f"  {zodiac}: {nums}")


if __name__ == '__main__':
    # 验证
    final_rate = validate_hybrid()
    
    print("\n" + "=" * 80)
    print(f"最终成功率: {final_rate:.1f}%")
    print("=" * 80)
    
    # 测试当前预测
    test_prediction()
