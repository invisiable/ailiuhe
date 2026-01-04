"""
基于最近10期数据预测策略验证
规则：
1. 只使用当天之前最近10期的数据进行预测（而非全部历史数据）
2. 生成TOP15预测
3. 与当天实际开出的号码对比
4. 统计最近10期的预测成功率
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class Recent10Predictor:
    """基于最近10期数据的预测器"""
    
    def __init__(self):
        self.element_numbers = {
            '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
            '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
            '水': [13, 14, 21, 22, 29, 30, 43, 44],
            '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
            '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
        }
    
    def analyze_recent_pattern(self, numbers, elements):
        """分析最近10期的模式"""
        recent_10 = numbers[-10:] if len(numbers) >= 10 else numbers
        recent_5 = numbers[-5:] if len(numbers) >= 5 else numbers
        recent_elements = elements[-10:] if len(elements) >= 10 else elements
        
        # 极端值分析
        extreme_count = sum(1 for n in recent_10 if n <= 10 or n >= 40)
        extreme_ratio = extreme_count / len(recent_10) if len(recent_10) > 0 else 0
        
        # 五行分析
        element_freq = Counter(recent_elements)
        
        # 数字频率
        num_freq = Counter(recent_10)
        
        return {
            'recent_10': recent_10,
            'recent_5': recent_5,
            'extreme_ratio': extreme_ratio,
            'is_extreme': extreme_ratio > 0.4,
            'element_freq': element_freq,
            'num_freq': num_freq
        }
    
    def method_frequency_recent(self, pattern, k=20):
        """方法1: 最近期频率优先"""
        recent_10 = pattern['recent_10']
        recent_5 = pattern['recent_5']
        freq = pattern['num_freq']
        
        weighted = {}
        for n in range(1, 50):
            weight = 1.0
            
            # 最近10期出现过的号码权重提升
            if n in recent_10:
                appearances = freq.get(n, 0)
                weight *= (1 + appearances * 1.5)
            
            # 最近5期出现过的降权（避免连续重复）
            if n in recent_5:
                weight *= 0.3
            
            # 极端值趋势
            if pattern['is_extreme']:
                if n <= 10 or n >= 40:
                    weight *= 2.0
                else:
                    weight *= 0.5
            else:
                if 15 <= n <= 35:
                    weight *= 1.5
            
            weighted[n] = weight
        
        sorted_nums = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def method_hot_numbers(self, pattern, k=20):
        """方法2: 热号策略（最近10期出现2次以上）"""
        recent_10 = pattern['recent_10']
        recent_5 = set(pattern['recent_5'])
        freq = pattern['num_freq']
        
        # 热号：最近10期出现2次以上
        hot_nums = []
        for n, count in freq.items():
            if count >= 2 and n not in recent_5:
                hot_nums.append((n, count))
        
        hot_nums.sort(key=lambda x: x[1], reverse=True)
        hot_nums = [n for n, _ in hot_nums[:10]]
        
        # 温号：出现1次
        warm_nums = []
        for n, count in freq.items():
            if count == 1 and n not in recent_5:
                warm_nums.append(n)
        
        # 冷号：未出现但符合趋势
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
        
        candidates = hot_nums + warm_nums[:6] + cold_nums[:4]
        return candidates[:k]
    
    def method_element_pattern(self, pattern, k=20):
        """方法3: 五行平衡策略"""
        recent_5 = set(pattern['recent_5'])
        element_freq = pattern['element_freq']
        num_freq = pattern['num_freq']
        
        # 找出最近10期出现最少的五行
        all_elements = ['金', '木', '水', '火', '土']
        element_counts = [(elem, element_freq.get(elem, 0)) for elem in all_elements]
        element_counts.sort(key=lambda x: x[1])
        
        # 选择出现较少的3个五行
        target_elements = [elem for elem, _ in element_counts[:3]]
        
        candidates = []
        for elem in target_elements:
            elem_nums = []
            for n in self.element_numbers[elem]:
                if n not in recent_5:
                    score = num_freq.get(n, 0) + np.random.random() * 0.5
                    elem_nums.append((n, score))
            
            elem_nums.sort(key=lambda x: x[1], reverse=True)
            candidates.extend([n for n, _ in elem_nums[:7]])
        
        return candidates[:k]
    
    def method_zone_recent(self, pattern, k=20):
        """方法4: 区域动态分配（基于最近趋势）"""
        recent_10 = pattern['recent_10']
        recent_5 = set(pattern['recent_5'])
        
        # 统计最近10期各区域出现次数
        zone_counts = {
            'zone1_10': sum(1 for n in recent_10 if 1 <= n <= 10),
            'zone11_20': sum(1 for n in recent_10 if 11 <= n <= 20),
            'zone21_30': sum(1 for n in recent_10 if 21 <= n <= 30),
            'zone31_40': sum(1 for n in recent_10 if 31 <= n <= 40),
            'zone41_49': sum(1 for n in recent_10 if 41 <= n <= 49)
        }
        
        # 根据出现频率动态分配配额（出现多的区域多分配）
        total = sum(zone_counts.values())
        if total > 0:
            zones = [
                (1, 10, max(2, int(zone_counts['zone1_10'] / total * 15))),
                (11, 20, max(2, int(zone_counts['zone11_20'] / total * 15))),
                (21, 30, max(2, int(zone_counts['zone21_30'] / total * 15))),
                (31, 40, max(2, int(zone_counts['zone31_40'] / total * 15))),
                (41, 49, max(2, int(zone_counts['zone41_49'] / total * 15)))
            ]
        else:
            zones = [(1, 10, 3), (11, 20, 3), (21, 30, 4), (31, 40, 3), (41, 49, 2)]
        
        candidates = []
        for start, end, quota in zones:
            zone_nums = []
            for n in range(start, end + 1):
                if n not in recent_5:
                    in_recent = 1 if n in recent_10 else 0
                    score = in_recent * 2 + np.random.random()
                    zone_nums.append((n, score))
            
            zone_nums.sort(key=lambda x: x[1], reverse=True)
            candidates.extend([n for n, _ in zone_nums[:quota]])
        
        return candidates[:k]
    
    def predict_top15(self, train_numbers, train_elements):
        """生成TOP15预测（基于最近10期数据）"""
        # 只使用最近10期数据
        recent_numbers = train_numbers[-10:] if len(train_numbers) >= 10 else train_numbers
        recent_elements = train_elements[-10:] if len(train_elements) >= 10 else train_elements
        
        pattern = self.analyze_recent_pattern(recent_numbers, recent_elements)
        
        # 使用4种方法生成预测
        method1 = self.method_frequency_recent(pattern, k=20)
        method2 = self.method_hot_numbers(pattern, k=20)
        method3 = self.method_element_pattern(pattern, k=20)
        method4 = self.method_zone_recent(pattern, k=20)
        
        # 综合评分
        scores = {}
        methods = [method1, method2, method3, method4]
        weights = [0.35, 0.30, 0.20, 0.15]  # 更重视频率和热号
        
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
        print("最近10期数据预测策略 - TOP15成功率验证")
        print("=" * 80)
        print(f"策略说明：")
        print(f"  1. 只使用当天之前最近10期的数据进行预测（而非全部历史数据）")
        print(f"  2. 基于最近趋势生成TOP15预测号码")
        print(f"  3. 与当天实际开出的号码进行对比验证")
        print(f"  4. 统计预测成功率")
        print("=" * 80)
        
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        total_records = len(df)
        
        print(f"\n总数据量: {total_records}期")
        print(f"验证期数: 最近{periods}期")
        print(f"预测策略: 每次只使用验证期之前最近10期数据\n")
        
        # 统计变量
        top5_success = 0
        top10_success = 0
        top15_success = 0
        
        results = []
        
        # 验证最近N期
        for i in range(periods):
            # 计算当前验证的期数索引
            test_index = total_records - periods + i
            
            # 确保至少有10期历史数据
            if test_index < 10:
                print(f"跳过第 {i+1} 期（历史数据不足10期）")
                continue
            
            # 获取训练数据（只使用最近的数据，不是全部）
            train_df = df.iloc[:test_index]
            train_numbers = train_df['number'].tolist()
            train_elements = train_df['element'].tolist()
            
            # 获取当天实际数据
            actual_row = df.iloc[test_index]
            actual_number = actual_row['number']
            actual_date = actual_row['date']
            
            # 显示使用的数据范围
            used_data = train_df.tail(10)
            start_date = used_data.iloc[0]['date']
            end_date = used_data.iloc[-1]['date']
            
            print(f"\n{'='*80}")
            print(f"验证第 {i+1}/{periods} 期")
            print(f"预测日期: {actual_date}")
            print(f"使用数据: {start_date} 至 {end_date} (最近10期)")
            print(f"最近10期号码: {train_numbers[-10:]}")
            
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
        print("验证结果统计 - 最近10期数据预测策略")
        print("=" * 80)
        
        valid_periods = len(results)
        top5_rate = (top5_success / valid_periods) * 100 if valid_periods > 0 else 0
        top10_rate = (top10_success / valid_periods) * 100 if valid_periods > 0 else 0
        top15_rate = (top15_success / valid_periods) * 100 if valid_periods > 0 else 0
        
        print(f"\n验证期数: {valid_periods} 期")
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
            'periods': valid_periods,
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
    predictor = Recent10Predictor()
    
    print("\n" + "="*80)
    print("对比测试：最近10期数据 vs 全部历史数据")
    print("="*80 + "\n")
    
    # 验证最近10期
    results = predictor.validate_recent_periods('data/lucky_numbers.csv', periods=10)
    
    print(f"\n{'='*80}")
    print("最终结论 - 最近10期数据预测策略")
    print("="*80)
    print(f"\n在最近{results['periods']}期的验证中:")
    print(f"  - TOP 15 预测成功率: {results['top15_rate']:.1f}%")
    print(f"  - TOP 10 预测成功率: {results['top10_rate']:.1f}%")
    print(f"  - TOP 5  预测成功率: {results['top5_rate']:.1f}%")
    
    if results['top15_rate'] >= 60:
        print(f"\n✅ TOP15预测成功率达到 {results['top15_rate']:.1f}%，达到60%目标！")
    elif results['top15_rate'] >= 50:
        print(f"\n✓ TOP15预测成功率为 {results['top15_rate']:.1f}%，接近60%目标")
    else:
        print(f"\n⚠️ TOP15预测成功率为 {results['top15_rate']:.1f}%，需要进一步优化")
    
    print(f"\n{'='*80}")
    print("策略对比建议")
    print("="*80)
    print("建议运行 validate_model_top15.py 进行对比：")
    print("  - 如果最近10期策略成功率更高，说明短期趋势更有效")
    print("  - 如果全部历史数据策略更高，说明长期模式更稳定")
    print("="*80)


if __name__ == '__main__':
    main()
