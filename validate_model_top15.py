"""
模型验证脚本 - TOP15预测成功率验证
规则：
1. 使用当天之前的所有数据进行训练
2. 生成TOP15预测
3. 与当天实际开出的号码对比
4. 统计最近10期的预测成功率
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class Top15Validator:
    """TOP15预测验证器"""
    
    def __init__(self):
        self.element_numbers = {
            '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
            '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
            '水': [13, 14, 21, 22, 29, 30, 43, 44],
            '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
            '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
        }
    
    def analyze_pattern(self, numbers):
        """分析数字模式"""
        recent_30 = numbers[-30:] if len(numbers) >= 30 else numbers
        recent_10 = numbers[-10:] if len(numbers) >= 10 else numbers
        recent_5 = numbers[-5:] if len(numbers) >= 5 else numbers
        
        # 极端值分析
        extreme_count = sum(1 for n in recent_10 if n <= 10 or n >= 40)
        extreme_ratio = extreme_count / len(recent_10) if len(recent_10) > 0 else 0
        
        # 连续性分析
        if len(recent_10) > 1:
            gaps = np.diff(recent_10)
            avg_gap = np.mean(np.abs(gaps))
        else:
            avg_gap = 0
        
        return {
            'recent_30': recent_30,
            'recent_10': recent_10,
            'recent_5': recent_5,
            'extreme_ratio': extreme_ratio,
            'is_extreme': extreme_ratio > 0.4,
            'avg_gap': avg_gap
        }
    
    def method_frequency_advanced(self, pattern, k=20):
        """方法1: 增强频率分析"""
        recent_30 = pattern['recent_30']
        recent_5 = pattern['recent_5']
        freq = Counter(recent_30)
        
        weighted = {}
        for n in range(1, 50):
            base_freq = freq.get(n, 0)
            weight = 1.0
            
            # 极端值趋势权重
            if pattern['is_extreme']:
                if n <= 10 or n >= 40:
                    weight *= 2.5
                else:
                    weight *= 0.3
            else:
                if 15 <= n <= 35:
                    weight *= 1.5
            
            # 最近5期出现过的降权
            if n in recent_5:
                weight *= 0.4
            
            # 频率加成
            if base_freq > 0:
                weight *= (1 + base_freq * 0.3)
            
            weighted[n] = weight
        
        sorted_nums = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def method_zone_dynamic(self, pattern, k=20):
        """方法2: 动态区域分配"""
        recent_30 = pattern['recent_30']
        recent_5 = set(pattern['recent_5'])
        freq = Counter(recent_30)
        
        if pattern['is_extreme']:
            zones = [
                (1, 10, 5),
                (11, 20, 2),
                (21, 30, 3),
                (31, 40, 3),
                (41, 49, 5)
            ]
        else:
            zones = [
                (1, 10, 3),
                (11, 20, 4),
                (21, 30, 5),
                (31, 40, 4),
                (41, 49, 3)
            ]
        
        candidates = []
        for start, end, quota in zones:
            zone_nums = []
            for n in range(start, end + 1):
                if n not in recent_5:
                    score = freq.get(n, 0) + np.random.random() * 0.5
                    zone_nums.append((n, score))
            
            zone_nums.sort(key=lambda x: x[1], reverse=True)
            candidates.extend([n for n, _ in zone_nums[:quota]])
        
        return candidates[:k]
    
    def method_element_balance(self, pattern, elements, k=20):
        """方法3: 五行平衡"""
        recent_30 = pattern['recent_30']
        recent_5 = set(pattern['recent_5'])
        freq = Counter(recent_30)
        
        element_freq = {}
        for num in recent_30:
            for elem, nums in self.element_numbers.items():
                if num in nums:
                    element_freq[elem] = element_freq.get(elem, 0) + 1
                    break
        
        # 选择出现频率较低的五行
        sorted_elements = sorted(element_freq.items(), key=lambda x: x[1])
        target_elements = [elem for elem, _ in sorted_elements[:3]]
        
        candidates = []
        for elem in target_elements:
            elem_nums = []
            for n in self.element_numbers[elem]:
                if n not in recent_5:
                    score = freq.get(n, 0) + np.random.random()
                    elem_nums.append((n, score))
            
            elem_nums.sort(key=lambda x: x[1], reverse=True)
            candidates.extend([n for n, _ in elem_nums[:7]])
        
        return candidates[:k]
    
    def method_hot_cold_balance(self, pattern, k=20):
        """方法4: 冷热号平衡"""
        recent_30 = pattern['recent_30']
        recent_5 = set(pattern['recent_5'])
        freq = Counter(recent_30)
        
        # 热号：出现3次以上
        hot_nums = [(n, c) for n, c in freq.items() if c >= 3 and n not in recent_5]
        hot_nums.sort(key=lambda x: x[1], reverse=True)
        hot_nums = [n for n, _ in hot_nums[:8]]
        
        # 冷号：出现0-1次
        cold_nums = []
        for n in range(1, 50):
            if freq.get(n, 0) <= 1 and n not in recent_5:
                cold_nums.append(n)
        
        np.random.shuffle(cold_nums)
        cold_nums = cold_nums[:8]
        
        # 温号：出现2次
        warm_nums = [(n, c) for n, c in freq.items() if c == 2 and n not in recent_5]
        np.random.shuffle(warm_nums)
        warm_nums = [n for n, _ in warm_nums[:4]]
        
        candidates = hot_nums + cold_nums + warm_nums
        return candidates[:k]
    
    def predict_top15(self, train_numbers, train_elements):
        """生成TOP15预测"""
        pattern = self.analyze_pattern(train_numbers)
        
        # 使用4种方法生成预测
        method1 = self.method_frequency_advanced(pattern, k=20)
        method2 = self.method_zone_dynamic(pattern, k=20)
        method3 = self.method_element_balance(pattern, train_elements, k=20)
        method4 = self.method_hot_cold_balance(pattern, k=20)
        
        # 综合评分
        scores = {}
        methods = [method1, method2, method3, method4]
        weights = [0.3, 0.25, 0.25, 0.2]
        
        for method, weight in zip(methods, weights):
            for i, num in enumerate(method):
                score = (len(method) - i) * weight
                scores[num] = scores.get(num, 0) + score
        
        # 排序返回TOP15
        sorted_predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top15 = [num for num, _ in sorted_predictions[:15]]
        
        return top15
    
    def validate_recent_periods(self, csv_file, periods=10):
        """验证最近N期的预测成功率"""
        print("=" * 80)
        print("模型验证 - TOP15预测成功率分析")
        print("=" * 80)
        print(f"验证规则：")
        print(f"  1. 使用当天之前的所有历史数据进行训练")
        print(f"  2. 生成TOP15预测号码")
        print(f"  3. 与当天实际开出的号码进行对比验证")
        print(f"  4. 统计预测成功率")
        print("=" * 80)
        
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        total_records = len(df)
        
        print(f"\n总数据量: {total_records}期")
        print(f"验证期数: 最近{periods}期")
        print(f"训练数据: 每次使用验证期之前的所有数据\n")
        
        # 统计变量
        top5_success = 0
        top10_success = 0
        top15_success = 0
        
        results = []
        
        # 验证最近N期
        for i in range(periods):
            # 计算当前验证的期数索引
            test_index = total_records - periods + i
            
            # 获取训练数据（当天之前的所有数据）
            train_df = df.iloc[:test_index]
            train_numbers = train_df['number'].tolist()
            train_elements = train_df['element'].tolist()
            
            # 获取当天实际数据
            actual_row = df.iloc[test_index]
            actual_number = actual_row['number']
            actual_date = actual_row['date']
            
            print(f"\n{'='*80}")
            print(f"验证第 {i+1}/{periods} 期")
            print(f"日期: {actual_date}")
            print(f"使用前 {test_index} 期数据进行训练")
            
            # 生成预测
            top15_predictions = self.predict_top15(train_numbers, train_elements)
            
            # 提取TOP5和TOP10
            top5_predictions = top15_predictions[:5]
            top10_predictions = top15_predictions[:10]
            
            print(f"\nTOP 5 预测:  {top5_predictions}")
            print(f"TOP 10 预测: {top10_predictions}")
            print(f"TOP 15 预测: {top15_predictions}")
            print(f"\n实际开出: {actual_number}")
            
            # 验证预测结果
            hit_level = None
            rank = None
            
            if actual_number in top5_predictions:
                rank = top5_predictions.index(actual_number) + 1
                hit_level = "TOP 5"
                top5_success += 1
                top10_success += 1
                top15_success += 1
                status = f"✅ TOP 5 命中! (排名第 {rank})"
            elif actual_number in top10_predictions:
                rank = top10_predictions.index(actual_number) + 1
                hit_level = "TOP 10"
                top10_success += 1
                top15_success += 1
                status = f"✓ TOP 10 命中 (排名第 {rank})"
            elif actual_number in top15_predictions:
                rank = top15_predictions.index(actual_number) + 1
                hit_level = "TOP 15"
                top15_success += 1
                status = f"○ TOP 15 命中 (排名第 {rank})"
            else:
                status = "❌ 未命中"
            
            print(f"结果: {status}")
            
            # 记录结果
            results.append({
                'period': i + 1,
                'date': actual_date,
                'actual': actual_number,
                'top15': top15_predictions,
                'hit_level': hit_level,
                'rank': rank,
                'status': status
            })
        
        # 输出统计结果
        print(f"\n{'='*80}")
        print("验证结果统计")
        print("=" * 80)
        
        top5_rate = (top5_success / periods) * 100
        top10_rate = (top10_success / periods) * 100
        top15_rate = (top15_success / periods) * 100
        
        print(f"\n验证期数: {periods} 期")
        print(f"\nTOP 5  命中: {top5_success} 期, 成功率: {top5_rate:.1f}%")
        print(f"TOP 10 命中: {top10_success} 期, 成功率: {top10_rate:.1f}%")
        print(f"TOP 15 命中: {top15_success} 期, 成功率: {top15_rate:.1f}%")
        
        # 详细结果列表
        print(f"\n{'='*80}")
        print("详细验证结果")
        print("=" * 80)
        print(f"{'期数':<6} {'日期':<12} {'实际':<6} {'命中级别':<10} {'排名':<6} {'状态':<20}")
        print("-" * 80)
        
        for r in results:
            period_str = f"第{r['period']}期"
            hit_level_str = r['hit_level'] if r['hit_level'] else "-"
            rank_str = str(r['rank']) if r['rank'] else "-"
            print(f"{period_str:<6} {r['date']:<12} {r['actual']:<6} {hit_level_str:<10} {rank_str:<6} {r['status']:<20}")
        
        print("=" * 80)
        
        return {
            'periods': periods,
            'top5_success': top5_success,
            'top10_success': top10_success,
            'top15_success': top15_success,
            'top5_rate': top5_rate,
            'top10_rate': top10_rate,
            'top15_rate': top15_rate,
            'results': results
        }


def main():
    """主函数"""
    validator = Top15Validator()
    
    # 验证最近30期
    results = validator.validate_recent_periods('data/lucky_numbers.csv', periods=30)
    
    print(f"\n最终结论:")
    print(f"在最近{results['periods']}期的验证中:")
    print(f"  - TOP 15 预测成功率: {results['top15_rate']:.1f}%")
    print(f"  - TOP 10 预测成功率: {results['top10_rate']:.1f}%")
    print(f"  - TOP 5  预测成功率: {results['top5_rate']:.1f}%")
    
    if results['top15_rate'] >= 60:
        print(f"\n✅ TOP15预测成功率达到 {results['top15_rate']:.1f}%，达到60%目标！")
    else:
        print(f"\n⚠️ TOP15预测成功率为 {results['top15_rate']:.1f}%，未达到60%目标")


if __name__ == '__main__':
    main()
