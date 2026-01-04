"""
加权最近5期预测策略
重点考虑最近5期数据的权重分布，提升下一期预测成功率

权重分配：
- 最近1期（昨天）：权重 1.0
- 最近2期：权重 1.5
- 最近3期：权重 2.0
- 最近4期：权重 2.5
- 最近5期：权重 3.0 (最高权重)
- 6-10期：权重 1.0
- 11-20期：权重 0.5
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class WeightedRecent5Predictor:
    """基于最近5期加权的预测器"""
    
    def __init__(self):
        self.element_numbers = {
            '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
            '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
            '水': [13, 14, 21, 22, 29, 30, 43, 44],
            '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
            '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
        }
    
    def get_weighted_frequency(self, numbers, elements):
        """计算加权频率"""
        if len(numbers) < 5:
            return Counter(numbers), Counter(elements)
        
        # 权重配置：越近期权重越高（优化版）
        weights = {
            1: 1.0,  # 最近第1期 - 降权避免立即重复
            2: 2.5,  # 最近第2期
            3: 3.5,  # 最近第3期
            4: 4.5,  # 最近第4期
            5: 5.5,  # 最近第5期（最高权重）
        }
        
        # 对6-10期给予中等权重
        for i in range(6, min(11, len(numbers) + 1)):
            weights[i] = 2.0
        
        # 对11-20期给予较低权重
        for i in range(11, min(21, len(numbers) + 1)):
            weights[i] = 1.0
        
        # 20期以外的权重很低
        for i in range(21, len(numbers) + 1):
            weights[i] = 0.5
        
        # 计算加权频率
        weighted_nums = {}
        weighted_elems = {}
        
        for i in range(len(numbers)):
            position = len(numbers) - i  # 从后往前，1表示最近一期
            weight = weights.get(position, 0.1)
            
            num = numbers[i]
            elem = elements[i]
            
            weighted_nums[num] = weighted_nums.get(num, 0) + weight
            weighted_elems[elem] = weighted_elems.get(elem, 0) + weight
        
        return weighted_nums, weighted_elems
    
    def analyze_pattern(self, numbers, elements):
        """分析数字模式（加权版本）"""
        recent_20 = numbers[-20:] if len(numbers) >= 20 else numbers
        recent_10 = numbers[-10:] if len(numbers) >= 10 else numbers
        recent_5 = numbers[-5:] if len(numbers) >= 5 else numbers
        recent_3 = numbers[-3:] if len(numbers) >= 3 else numbers
        recent_1 = numbers[-1:] if len(numbers) >= 1 else []
        
        # 加权频率
        weighted_nums, weighted_elems = self.get_weighted_frequency(numbers, elements)
        
        # 极端值分析（最近10期）
        extreme_count = sum(1 for n in recent_10 if n <= 10 or n >= 40)
        extreme_ratio = extreme_count / len(recent_10) if len(recent_10) > 0 else 0
        
        # 区域分布（最近5期）
        zone_dist = {
            'small': sum(1 for n in recent_5 if n <= 10),
            'mid_low': sum(1 for n in recent_5 if 11 <= n <= 20),
            'mid': sum(1 for n in recent_5 if 21 <= n <= 30),
            'mid_high': sum(1 for n in recent_5 if 31 <= n <= 40),
            'large': sum(1 for n in recent_5 if n >= 41)
        }
        
        return {
            'recent_20': recent_20,
            'recent_10': recent_10,
            'recent_5': recent_5,
            'recent_3': recent_3,
            'recent_1': recent_1,
            'weighted_nums': weighted_nums,
            'weighted_elems': weighted_elems,
            'extreme_ratio': extreme_ratio,
            'is_extreme': extreme_ratio > 0.4,
            'zone_dist': zone_dist
        }
    
    def method_weighted_frequency(self, pattern, k=20):
        """方法1: 加权频率（核心方法）"""
        weighted_nums = pattern['weighted_nums']
        recent_1 = set(pattern['recent_1'])
        recent_3 = set(pattern['recent_3'])
        recent_5 = pattern['recent_5']
        
        scores = {}
        for n in range(1, 50):
            score = weighted_nums.get(n, 0.1)
            
            # 最近1期出现的降权（避免立即重复）
            if n in recent_1:
                score *= 0.2
            # 最近3期出现的适度降权
            elif n in recent_3:
                score *= 0.5
            
            # 如果在最近5期出现2次以上，说明是热号，给予加权
            appearances_in_5 = recent_5.count(n)
            if appearances_in_5 >= 2 and n not in recent_1:
                score *= (1 + appearances_in_5 * 0.5)
            
            # 极端值趋势调整
            if pattern['is_extreme']:
                if n <= 10 or n >= 40:
                    score *= 1.6
                else:
                    score *= 0.7
            else:
                if 15 <= n <= 35:
                    score *= 1.2
            
            scores[n] = score
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def method_recent_trend(self, pattern, k=20):
        """方法2: 最近趋势跟随"""
        recent_5 = pattern['recent_5']
        recent_3 = set(pattern['recent_3'])
        recent_1 = set(pattern['recent_1'])
        
        # 找出最近5期的相邻数字
        candidates = set()
        for num in recent_5:
            if num not in recent_1:
                # 相邻数字
                if num > 1:
                    candidates.add(num - 1)
                if num < 49:
                    candidates.add(num + 1)
                # 同个位数字
                for n in range(num % 10, 50, 10):
                    if n > 0 and n not in recent_3:
                        candidates.add(n)
        
        # 评分
        scores = {}
        weighted_nums = pattern['weighted_nums']
        for n in candidates:
            score = weighted_nums.get(n, 0.5) + np.random.random()
            scores[n] = score
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def method_element_weighted(self, pattern, k=20):
        """方法3: 加权五行平衡"""
        weighted_elems = pattern['weighted_elems']
        recent_3 = set(pattern['recent_3'])
        weighted_nums = pattern['weighted_nums']
        
        # 找出权重最低的3个五行（需要补充）
        elem_list = sorted(weighted_elems.items(), key=lambda x: x[1])
        target_elements = [elem for elem, _ in elem_list[:3]]
        
        candidates = []
        for elem in target_elements:
            elem_nums = []
            for n in self.element_numbers[elem]:
                if n not in recent_3:
                    score = weighted_nums.get(n, 0) + np.random.random() * 2
                    elem_nums.append((n, score))
            
            elem_nums.sort(key=lambda x: x[1], reverse=True)
            candidates.extend([n for n, _ in elem_nums[:7]])
        
        return candidates[:k]
    
    def method_zone_weighted(self, pattern, k=20):
        """方法4: 区域加权分配"""
        zone_dist = pattern['zone_dist']
        recent_3 = set(pattern['recent_3'])
        weighted_nums = pattern['weighted_nums']
        
        # 根据最近5期区域分布动态分配
        total = sum(zone_dist.values())
        if total > 0:
            zones = [
                (1, 10, max(2, min(6, int(zone_dist['small'] / total * 20)))),
                (11, 20, max(2, min(5, int(zone_dist['mid_low'] / total * 20)))),
                (21, 30, max(2, min(5, int(zone_dist['mid'] / total * 20)))),
                (31, 40, max(2, min(5, int(zone_dist['mid_high'] / total * 20)))),
                (41, 49, max(2, min(6, int(zone_dist['large'] / total * 20))))
            ]
        else:
            zones = [(1, 10, 3), (11, 20, 3), (21, 30, 4), (31, 40, 3), (41, 49, 3)]
        
        candidates = []
        for start, end, quota in zones:
            zone_nums = []
            for n in range(start, end + 1):
                if n not in recent_3:
                    score = weighted_nums.get(n, 0) + np.random.random()
                    zone_nums.append((n, score))
            
            zone_nums.sort(key=lambda x: x[1], reverse=True)
            candidates.extend([n for n, _ in zone_nums[:quota]])
        
        return candidates[:k]
    
    def predict_top15(self, train_numbers, train_elements):
        """生成TOP15预测"""
        pattern = self.analyze_pattern(train_numbers, train_elements)
        
        # 使用4种方法生成预测
        method1 = self.method_weighted_frequency(pattern, k=20)
        method2 = self.method_recent_trend(pattern, k=20)
        method3 = self.method_element_weighted(pattern, k=20)
        method4 = self.method_zone_weighted(pattern, k=20)
        
        # 综合评分（更均衡的权重分配）
        scores = {}
        methods = [method1, method2, method3, method4]
        weights = [0.40, 0.30, 0.18, 0.12]  # 更重视加权频率和趋势
        
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
        print("加权最近5期预测策略 - TOP15成功率验证")
        print("=" * 80)
        print(f"策略说明：")
        print(f"  1. 对最近5期数据赋予更高权重（5期>4期>3期>2期>1期）")
        print(f"  2. 最近1期降权避免立即重复")
        print(f"  3. 综合考虑加权频率、趋势、五行、区域")
        print(f"  4. 与当天实际开出的号码进行对比验证")
        print("=" * 80)
        
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        total_records = len(df)
        
        print(f"\n总数据量: {total_records}期")
        print(f"验证期数: 最近{periods}期\n")
        
        # 统计变量
        top5_success = 0
        top10_success = 0
        top15_success = 0
        
        results = []
        
        # 验证最近N期
        for i in range(periods):
            test_index = total_records - periods + i
            
            if test_index < 5:
                print(f"跳过第 {i+1} 期（历史数据不足5期）")
                continue
            
            # 获取训练数据
            train_df = df.iloc[:test_index]
            train_numbers = train_df['number'].tolist()
            train_elements = train_df['element'].tolist()
            
            # 获取当天实际数据
            actual_row = df.iloc[test_index]
            actual_number = actual_row['number']
            actual_date = actual_row['date']
            
            print(f"\n{'='*80}")
            print(f"验证第 {i+1}/{periods} 期")
            print(f"预测日期: {actual_date}")
            print(f"最近5期: {train_numbers[-5:]}")
            print(f"最近1期: {train_numbers[-1]} (将降权避免重复)")
            
            # 生成预测
            top15_predictions = self.predict_top15(train_numbers, train_elements)
            
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
        print("验证结果统计 - 加权最近5期策略")
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
    predictor = WeightedRecent5Predictor()
    
    print("\n加权最近5期策略核心设计：")
    print("  权重分配: 5期(5.5) > 4期(4.5) > 3期(3.5) > 2期(2.5) > 1期(1.0)")
    print("  避重机制: 最近1期权重x0.2, 最近3期权重x0.5")
    print("  热号识别: 5期内出现2次以上的号码额外加权")
    print("  多维融合: 加权频率(40%) + 趋势(30%) + 五行(18%) + 区域(12%)\n")
    
    results = predictor.validate_recent_periods('data/lucky_numbers.csv', periods=10)
    
    print(f"\n{'='*80}")
    print("最终结论 - 加权最近5期策略")
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
        print(f"\n⚠️ TOP15预测成功率为 {results['top15_rate']:.1f}%，继续优化中")


if __name__ == '__main__':
    main()
