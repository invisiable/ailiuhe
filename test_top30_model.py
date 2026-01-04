"""
Top30预测模型测试
基于混合策略扩展到Top30预测，并输出每期预测结果和成功率
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime


class Top30Predictor:
    """Top30预测器 - 扩展混合策略"""
    
    def _analyze_recent_10(self, numbers, elements):
        """分析最近10期的模式"""
        recent_10 = numbers[-10:] if len(numbers) >= 10 else numbers
        recent_5 = numbers[-5:] if len(numbers) >= 5 else numbers
        
        num_freq = Counter(recent_10)
        
        avg = np.mean(recent_10)
        is_extreme = avg < 15 or avg > 35
        
        return {
            'recent_10': set(recent_10),
            'recent_5': set(recent_5),
            'num_freq': num_freq,
            'avg': avg,
            'is_extreme': is_extreme
        }
    
    def _predict_strategy_a(self, numbers):
        """策略A: 全部历史数据（稳定）"""
        freq = Counter(numbers)
        recent_30 = set(numbers[-30:])
        
        candidates = []
        for num in range(1, 50):
            score = 0
            count = freq.get(num, 0)
            
            if count > 0:
                score += count * 2
            
            if num not in recent_30:
                score += 10
            
            if 15 <= num <= 35:
                score += 5
            
            candidates.append((num, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in candidates[:30]]
    
    def _predict_strategy_b(self, numbers, elements):
        """策略B: 最近10期数据（精准）"""
        recent_numbers = numbers[-10:] if len(numbers) >= 10 else numbers
        recent_elements = elements[-10:] if len(elements) >= 10 else elements
        
        pattern = self._analyze_recent_10(recent_numbers, recent_elements)
        recent_10 = pattern['recent_10']
        recent_5 = pattern['recent_5']
        freq = pattern['num_freq']
        
        # 方法1: 频率优先
        weighted = {}
        for n in range(1, 50):
            weight = 1.0
            
            if n in recent_10:
                appearances = freq.get(n, 0)
                weight *= (1 + appearances * 1.5)
            
            if n in recent_5:
                weight *= 0.3
            
            if pattern['is_extreme']:
                if n <= 10 or n >= 40:
                    weight *= 2.0
                else:
                    weight *= 0.5
            else:
                if 15 <= n <= 35:
                    weight *= 1.5
            
            weighted[n] = weight
        
        # 方法2: 热号策略
        hot_nums = []
        for n, count in freq.items():
            if count >= 2 and n not in recent_5:
                hot_nums.append((n, count))
        hot_nums.sort(key=lambda x: x[1], reverse=True)
        hot_nums = [n for n, _ in hot_nums[:10]]
        
        warm_nums = [n for n, count in freq.items() if count == 1 and n not in recent_5]
        
        cold_nums = []
        for n in range(1, 50):
            if n not in recent_10:
                if pattern['is_extreme']:
                    if n <= 10 or n >= 40:
                        cold_nums.append(n)
                else:
                    if 15 <= n <= 35:
                        cold_nums.append(n)
        
        np.random.seed(42)
        np.random.shuffle(warm_nums)
        np.random.shuffle(cold_nums)
        
        hot_candidates = hot_nums + warm_nums[:6] + cold_nums[:4]
        
        # 综合评分
        scores = {}
        method1 = sorted(weighted.items(), key=lambda x: x[1], reverse=True)[:30]
        method1 = [num for num, _ in method1]
        
        for i, num in enumerate(method1):
            scores[num] = scores.get(num, 0) + (30 - i) * 0.6
        
        for i, num in enumerate(hot_candidates[:30]):
            scores[num] = scores.get(num, 0) + (30 - i) * 0.4
        
        sorted_predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_predictions[:30]]
    
    def predict_top30(self, numbers, elements):
        """
        预测Top30
        - TOP 1-5:   策略B（最近10期）
        - TOP 6-15:  策略A（全部历史）
        - TOP 16-30: 策略B和策略A交替补充
        """
        strategy_a = self._predict_strategy_a(numbers)
        strategy_b = self._predict_strategy_b(numbers, elements)
        
        top30_predictions = []
        
        # TOP 1-5: 策略B
        for num in strategy_b[:5]:
            if num not in top30_predictions:
                top30_predictions.append(num)
        
        # TOP 6-15: 策略A
        for num in strategy_a:
            if num not in top30_predictions:
                top30_predictions.append(num)
            if len(top30_predictions) >= 15:
                break
        
        # TOP 16-30: 交替补充
        remaining_b = [n for n in strategy_b if n not in top30_predictions]
        remaining_a = [n for n in strategy_a if n not in top30_predictions]
        
        j = 0
        while len(top30_predictions) < 30:
            if j < len(remaining_b):
                num = remaining_b[j]
                if num not in top30_predictions:
                    top30_predictions.append(num)
            if len(top30_predictions) >= 30:
                break
            if j < len(remaining_a):
                num = remaining_a[j]
                if num not in top30_predictions:
                    top30_predictions.append(num)
            if len(top30_predictions) >= 30:
                break
            j += 1
        
        return top30_predictions[:30]
    
    def predict_top20(self, numbers, elements):
        """
        预测Top20
        - TOP 1-5:   策略B（最近10期）
        - TOP 6-15:  策略A（全部历史）
        - TOP 16-20: 策略B和策略A交替补充
        返回: Top20预测列表
        """
        strategy_a = self._predict_strategy_a(numbers)
        strategy_b = self._predict_strategy_b(numbers, elements)
        
        top20_predictions = []
        
        # TOP 1-5: 策略B
        for num in strategy_b[:5]:
            if num not in top20_predictions:
                top20_predictions.append(num)
        
        # TOP 6-15: 策略A
        for num in strategy_a:
            if num not in top20_predictions:
                top20_predictions.append(num)
            if len(top20_predictions) >= 15:
                break
        
        # TOP 16-20: 交替补充
        remaining_b = [n for n in strategy_b if n not in top20_predictions]
        remaining_a = [n for n in strategy_a if n not in top20_predictions]
        
        j = 0
        while len(top20_predictions) < 20:
            if j < len(remaining_b):
                num = remaining_b[j]
                if num not in top20_predictions:
                    top20_predictions.append(num)
            if len(top20_predictions) >= 20:
                break
            if j < len(remaining_a):
                num = remaining_a[j]
                if num not in top20_predictions:
                    top20_predictions.append(num)
            if len(top20_predictions) >= 20:
                break
            j += 1
        
        return top20_predictions[:20]
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_k=20):
        """
        通用预测接口
        参数:
            csv_file: 数据文件路径
            top_k: 返回Top K预测，支持5/10/15/20/30
        返回:
            预测数字列表
        """
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        numbers = df['number'].values
        elements = df['element'].values
        
        if top_k == 20:
            return self.predict_top20(numbers, elements)
        elif top_k == 30:
            return self.predict_top30(numbers, elements)
        else:
            # 对于其他值，返回top30的前top_k个
            top30 = self.predict_top30(numbers, elements)
            return top30[:top_k]


def test_top30_predictions(csv_file='data/lucky_numbers.csv', periods=50):
    """测试Top30预测模型"""
    
    predictor = Top30Predictor()
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    if len(df) < periods + 1:
        print(f"数据不足，需要至少 {periods+1} 期数据")
        return
    
    print("=" * 80)
    print(f"Top30预测模型测试 - 最近{periods}期")
    print("=" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"验证规则: 使用当期数据预测下一期，与实际结果比对")
    print(f"策略说明:")
    print(f"  TOP 1-5:   最近10期数据策略（精准预测）")
    print(f"  TOP 6-15:  全部历史数据策略（稳定覆盖）")
    print(f"  TOP 16-30: 策略B+策略A交替补充")
    print("=" * 80)
    print()
    
    results = {
        'top5': [],
        'top10': [],
        'top15': [],
        'top20': [],
        'top30': [],
        'details': []
    }
    
    # 从后往前验证最近N期
    for i in range(periods):
        next_index = len(df) - periods + i
        
        # 训练数据：当期之前的所有数据
        train_data = df.iloc[:next_index]
        current_date = train_data.iloc[-1]['date']
        current_period = next_index
        
        # 测试数据：下一期
        next_actual = df.iloc[next_index]['number']
        next_date = df.iloc[next_index]['date']
        next_period = next_index + 1
        
        numbers = train_data['number'].values
        elements = train_data['element'].values
        
        # 生成Top30预测
        top30_pred = predictor.predict_top30(numbers, elements)
        
        top5 = top30_pred[:5]
        top10 = top30_pred[:10]
        top15 = top30_pred[:15]
        top20 = top30_pred[:20]
        top30 = top30_pred[:30]
        
        # 检查命中情况
        hit_top5 = next_actual in top5
        hit_top10 = next_actual in top10
        hit_top15 = next_actual in top15
        hit_top20 = next_actual in top20
        hit_top30 = next_actual in top30
        
        results['top5'].append(hit_top5)
        results['top10'].append(hit_top10)
        results['top15'].append(hit_top15)
        results['top20'].append(hit_top20)
        results['top30'].append(hit_top30)
        
        # 确定命中等级
        rank = None
        hit_level = "未命中"
        if hit_top5:
            rank = top5.index(next_actual) + 1
            hit_level = f"TOP5 (#{rank})"
        elif hit_top10:
            rank = top10.index(next_actual) + 1
            hit_level = f"TOP10 (#{rank})"
        elif hit_top15:
            rank = top15.index(next_actual) + 1
            hit_level = f"TOP15 (#{rank})"
        elif hit_top20:
            rank = top20.index(next_actual) + 1
            hit_level = f"TOP20 (#{rank})"
        elif hit_top30:
            rank = top30.index(next_actual) + 1
            hit_level = f"TOP30 (#{rank})"
        
        # 输出每期预测结果
        status_icon = "✅" if hit_top15 else ("○" if hit_top30 else "❌")
        print(f"第{i+1:>2}期 | {next_date} | 实际: {next_actual:>2} | {status_icon} {hit_level:>12} | Top30: {top30}")
        
        results['details'].append({
            'period': i + 1,
            'date': next_date,
            'actual': next_actual,
            'rank': rank,
            'hit_level': hit_level,
            'top30': top30
        })
    
    # 统计成功率
    print(f"\n{'='*80}")
    print("预测成功率统计")
    print(f"{'='*80}\n")
    
    total = len(results['top5'])
    top5_success = sum(results['top5'])
    top10_success = sum(results['top10'])
    top15_success = sum(results['top15'])
    top20_success = sum(results['top20'])
    top30_success = sum(results['top30'])
    
    top5_rate = (top5_success / total) * 100
    top10_rate = (top10_success / total) * 100
    top15_rate = (top15_success / total) * 100
    top20_rate = (top20_success / total) * 100
    top30_rate = (top30_success / total) * 100
    
    print(f"验证期数: {total} 期\n")
    print(f"成功率统计:")
    print(f"  TOP 5  命中: {top5_success:>2}/{total} 期 = {top5_rate:>5.1f}%")
    print(f"  TOP 10 命中: {top10_success:>2}/{total} 期 = {top10_rate:>5.1f}%")
    print(f"  TOP 15 命中: {top15_success:>2}/{total} 期 = {top15_rate:>5.1f}%")
    print(f"  TOP 20 命中: {top20_success:>2}/{total} 期 = {top20_rate:>5.1f}%")
    print(f"  TOP 30 命中: {top30_success:>2}/{total} 期 = {top30_rate:>5.1f}%")
    
    print(f"\n{'='*80}")
    print("结论")
    print(f"{'='*80}\n")
    print(f"✅ Top30预测模型成功率: {top30_rate:.1f}%")
    print(f"   相比Top15 ({top15_rate:.1f}%)，Top30提升了 {top30_rate - top15_rate:.1f} 个百分点")
    
    return {
        'periods': total,
        'top5_rate': top5_rate,
        'top10_rate': top10_rate,
        'top15_rate': top15_rate,
        'top20_rate': top20_rate,
        'top30_rate': top30_rate,
        'results': results
    }


if __name__ == '__main__':
    print("\n🔮 Top30预测模型测试\n")
    results = test_top30_predictions(periods=50)
