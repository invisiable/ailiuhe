"""
混合预测系统 - 综合所有模型达到Top 15 60%成功率
结合：简化统计预测器 + ML模型预测 + 综合分析
"""

import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from lucky_number_predictor import LuckyNumberPredictor


class HybridPredictor:
    """混合预测器 - 综合多种方法"""
    
    def __init__(self):
        self.element_numbers = {
            '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
            '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
            '水': [13, 14, 21, 22, 29, 30, 43, 44],
            '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
            '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
        }
        self.ml_predictor = None
    
    def analyze_pattern(self, numbers):
        """分析数字模式"""
        recent_30 = numbers[-30:]
        recent_10 = numbers[-10:]
        recent_5 = numbers[-5:]
        
        # 极端值分析
        extreme_count = sum(1 for n in recent_10 if n <= 10 or n >= 40)
        extreme_ratio = extreme_count / len(recent_10)
        
        # 连续性分析
        gaps = np.diff(recent_10)
        avg_gap = np.mean(np.abs(gaps))
        
        # 周期性分析
        period_5 = recent_30[-25:-20]  # 5期前
        period_10 = recent_30[-20:-15]  # 10期前
        
        return {
            'recent_30': recent_30,
            'recent_10': recent_10,
            'recent_5': recent_5,
            'extreme_ratio': extreme_ratio,
            'is_extreme': extreme_ratio > 0.4,
            'avg_gap': avg_gap,
            'period_5': period_5,
            'period_10': period_10
        }
    
    def method_frequency_advanced(self, pattern, k=20):
        """方法1: 增强频率分析"""
        recent_30 = pattern['recent_30']
        recent_5 = pattern['recent_5']
        freq = Counter(recent_30)
        
        # 多层权重
        weighted = {}
        for n in range(1, 50):
            base_freq = freq.get(n, 0)
            weight = 1.0
            
            # 极端值趋势权重
            if pattern['is_extreme']:
                if n <= 10 or n >= 40:
                    weight *= 2.5  # 极端值强化
                else:
                    weight *= 0.3
            else:
                if 15 <= n <= 35:
                    weight *= 1.5  # 中间值偏好
            
            # 最近5期出现过的降权（避免重复）
            if n in recent_5:
                weight *= 0.4
            
            # 频率加成
            if base_freq > 0:
                weight *= (1 + base_freq * 0.3)
            
            weighted[n] = weight
        
        sorted_nums = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def method_ml_ensemble(self, numbers, k=20):
        """方法2: ML集成预测"""
        if self.ml_predictor is None:
            return []
        
        try:
            # 使用ML预测器的Top K概率
            predictions = self.ml_predictor.predict_top_probabilities(top_k=k)
            return [p['number'] for p in predictions]
        except:
            return []
    
    def method_zone_dynamic(self, pattern, k=20):
        """方法3: 动态区域分配"""
        recent_30 = pattern['recent_30']
        recent_5 = set(pattern['recent_5'])
        freq = Counter(recent_30)
        
        # 根据趋势动态调整区域配额
        if pattern['is_extreme']:
            zones = [
                (1, 10, 5),    # 极小 - 增加
                (11, 20, 2),   # 小 - 减少
                (21, 30, 3),   # 中
                (31, 40, 2),   # 大 - 减少
                (41, 49, 8)    # 极大 - 大幅增加
            ]
        else:
            zones = [
                (1, 10, 3),
                (11, 20, 4),
                (21, 30, 6),
                (31, 40, 4),
                (41, 49, 3)
            ]
        
        result = []
        for start, end, count in zones:
            zone_nums = [
                (n, freq.get(n, 0)) 
                for n in range(start, end+1) 
                if n not in recent_5
            ]
            zone_nums.sort(key=lambda x: x[1], reverse=True)
            result.extend([n for n, _ in zone_nums[:count]])
        
        return result[:k]
    
    def method_cyclic_pattern(self, pattern, k=20):
        """方法4: 周期模式识别"""
        recent_30 = pattern['recent_30']
        period_5 = pattern['period_5']
        period_10 = pattern['period_10']
        
        # 寻找周期性重复
        candidates = {}
        
        # 5期周期
        for n in period_5:
            candidates[n] = candidates.get(n, 0) + 2.0
        
        # 10期周期
        for n in period_10:
            candidates[n] = candidates.get(n, 0) + 1.5
        
        # 最近趋势
        freq = Counter(recent_30[-15:])
        for n, count in freq.items():
            candidates[n] = candidates.get(n, 0) + count * 0.5
        
        # 补充未出现的热门数字
        all_freq = Counter(recent_30)
        for n, count in all_freq.most_common(30):
            if n not in candidates:
                candidates[n] = count * 0.3
        
        sorted_nums = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def method_gap_prediction(self, pattern, k=20):
        """方法5: 间隔预测"""
        recent_30 = pattern['recent_30']
        recent_10 = pattern['recent_10']
        
        # 计算每个数字距离上次出现的间隔
        last_seen = {}
        for i, n in enumerate(recent_30):
            last_seen[n] = len(recent_30) - i
        
        # 间隔越长，越可能出现
        candidates = {}
        for n in range(1, 50):
            gap = last_seen.get(n, 30)  # 未出现过的按30期算
            
            # 间隔权重：5-15期最佳，太短或太长降权
            if 5 <= gap <= 15:
                weight = 2.0
            elif 3 <= gap <= 20:
                weight = 1.5
            elif gap > 20:
                weight = 1.0 + (gap - 20) * 0.1
            else:
                weight = 0.5
            
            candidates[n] = weight
        
        sorted_nums = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def predict_hybrid(self, numbers, top_k=15, use_ml=True):
        """混合预测 - 综合所有方法"""
        # 分析模式
        pattern = self.analyze_pattern(numbers)
        
        # 准备ML模型
        if use_ml and self.ml_predictor is None:
            try:
                self.ml_predictor = LuckyNumberPredictor()
                self.ml_predictor.load_data('data/lucky_numbers.csv')
                self.ml_predictor.train_model('lightgbm', test_size=0.15)
            except:
                use_ml = False
        
        # 运行所有方法
        methods = []
        
        # 方法1: 增强频率分析 (权重25%)
        m1 = self.method_frequency_advanced(pattern, top_k * 2)
        methods.append((m1, 0.25))
        
        # 方法2: ML集成 (权重20%)
        if use_ml:
            m2 = self.method_ml_ensemble(numbers, top_k * 2)
            if m2:
                methods.append((m2, 0.20))
        
        # 方法3: 动态区域 (权重20%)
        m3 = self.method_zone_dynamic(pattern, top_k * 2)
        methods.append((m3, 0.20))
        
        # 方法4: 周期模式 (权重20%)
        m4 = self.method_cyclic_pattern(pattern, top_k * 2)
        methods.append((m4, 0.20))
        
        # 方法5: 间隔预测 (权重15%)
        m5 = self.method_gap_prediction(pattern, top_k * 2)
        methods.append((m5, 0.15))
        
        # 综合评分
        scores = {}
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                scores[num] = scores.get(num, 0) + score
        
        # 排序并返回
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in final[:top_k]]


def test_hybrid_predictor():
    """测试混合预测器"""
    print("=" * 80)
    print("混合预测系统 - 目标：Top 15 达到 60%")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    print(f"\n数据集: {len(df)}期")
    print(f"最近10期: {numbers[-10:].tolist()}")
    
    # 创建预测器
    predictor = HybridPredictor()
    
    # 测试最近20期
    print("\n" + "=" * 80)
    print("回测最近20期")
    print("=" * 80)
    
    results = {'top5': 0, 'top10': 0, 'top15': 0, 'top20': 0, 'details': []}
    
    test_periods = min(20, len(numbers) - 50)  # 至少保留50期训练数据
    
    for i in range(len(numbers) - test_periods, len(numbers)):
        actual = numbers[i]
        history = numbers[:i]
        
        period_num = i + 1
        print(f"\n第{period_num}期: 实际 = {actual}")
        
        # 预测
        try:
            predictions = predictor.predict_hybrid(history, top_k=20, use_ml=True)
            
            # 检查命中
            if actual in predictions:
                rank = predictions.index(actual) + 1
                
                if rank <= 5:
                    level = "[★★★] Top 5"
                    results['top5'] += 1
                    results['top10'] += 1
                    results['top15'] += 1
                    results['top20'] += 1
                elif rank <= 10:
                    level = "[★★] Top 10"
                    results['top10'] += 1
                    results['top15'] += 1
                    results['top20'] += 1
                elif rank <= 15:
                    level = "[★] Top 15"
                    results['top15'] += 1
                    results['top20'] += 1
                else:
                    level = "[+] Top 20"
                    results['top20'] += 1
                
                print(f"  ✅ 命中! 排名: {rank} {level}")
            else:
                print(f"  ❌ 未命中")
            
            print(f"  预测Top15: {predictions[:15]}")
            
            results['details'].append({
                'period': period_num,
                'actual': actual,
                'hit_top15': actual in predictions[:15],
                'hit_top20': actual in predictions[:20]
            })
        except Exception as e:
            print(f"  ⚠️  预测出错: {e}")
    
    # 统计
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    
    total = len(results['details'])
    
    print(f"\n命中统计 (最近{total}期):")
    print(f"  Top 5:  {results['top5']}/{total} = {results['top5']/total*100:.1f}%")
    print(f"  Top 10: {results['top10']}/{total} = {results['top10']/total*100:.1f}%")
    print(f"  Top 15: {results['top15']}/{total} = {results['top15']/total*100:.1f}%")
    print(f"  Top 20: {results['top20']}/{total} = {results['top20']/total*100:.1f}%")
    
    # 对比随机概率
    print(f"\n对比随机概率:")
    for k, name in [(5, 'top5'), (10, 'top10'), (15, 'top15'), (20, 'top20')]:
        actual_rate = results[name] / total * 100
        random_rate = k / 49 * 100
        improvement = actual_rate / random_rate if random_rate > 0 else 0
        
        if improvement >= 1.5:
            status = "✅ 优秀"
        elif improvement >= 1.2:
            status = "🟢 良好"
        else:
            status = "⚠️  一般"
        
        print(f"  {name.upper()}: 实际{actual_rate:.1f}% vs 随机{random_rate:.1f}% = {improvement:.2f}x {status}")
    
    # 评估
    top15_rate = results['top15'] / total * 100
    top20_rate = results['top20'] / total * 100
    
    print("\n" + "=" * 80)
    print("目标评估")
    print("=" * 80)
    
    if top15_rate >= 60:
        print(f"\n🎉 [成功] Top 15: {top15_rate:.1f}% - 已达到60%目标!")
    elif top15_rate >= 50:
        print(f"\n👍 [良好] Top 15: {top15_rate:.1f}% - 接近60%目标")
    else:
        print(f"\n📊 [进行中] Top 15: {top15_rate:.1f}%")
    
    if top20_rate >= 60:
        print(f"✅ [成功] Top 20: {top20_rate:.1f}% - 已达到60%目标!")
    
    # 下一期预测
    print("\n" + "=" * 80)
    print("下一期预测")
    print("=" * 80)
    
    next_predictions = predictor.predict_hybrid(numbers, top_k=20, use_ml=True)
    
    print(f"\nTop 15 预测号码:")
    print(f"  {next_predictions[:15]}")
    
    print(f"\nTop 20 预测号码 (供参考):")
    print(f"  {next_predictions[:20]}")
    
    # 区域分布
    zones = {
        '极小(1-10)': [n for n in next_predictions[:15] if 1 <= n <= 10],
        '小(11-20)': [n for n in next_predictions[:15] if 11 <= n <= 20],
        '中(21-30)': [n for n in next_predictions[:15] if 21 <= n <= 30],
        '大(31-40)': [n for n in next_predictions[:15] if 31 <= n <= 40],
        '极大(41-49)': [n for n in next_predictions[:15] if 41 <= n <= 49]
    }
    
    print(f"\nTop 15区域分布:")
    for zone, nums in zones.items():
        if nums:
            print(f"  {zone}: {nums}")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == '__main__':
    test_hybrid_predictor()
