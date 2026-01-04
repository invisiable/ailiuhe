"""
验证 TOP5 策略B - 最近100期预测成功率
策略B定义：使用最近10期数据进行精准预测
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime


class Top5StrategyBValidator:
    """TOP5策略B验证器"""
    
    def predict_strategy_b(self, numbers, elements):
        """
        策略B: 基于最近10期数据的精准预测
        使用多种方法组合评分
        """
        # 只使用最近10期数据
        recent_numbers = numbers[-10:] if len(numbers) >= 10 else numbers
        recent_elements = elements[-10:] if len(elements) >= 10 else elements
        
        # 方法1: 频率分析（最近10期出现次数）
        freq_counter = Counter(recent_numbers)
        
        # 方法2: 最近5期去重分析（更近的历史）
        recent_5 = set(numbers[-5:]) if len(numbers) >= 5 else set(numbers)
        
        # 方法3: 五行分布分析
        element_counter = Counter(recent_elements)
        
        # 方法4: 号码区间分析
        zone_dist = {
            'low': sum(1 for n in recent_numbers if 1 <= n <= 16),
            'mid': sum(1 for n in recent_numbers if 17 <= n <= 33),
            'high': sum(1 for n in recent_numbers if 34 <= n <= 49)
        }
        
        # 综合评分
        scores = {}
        for num in range(1, 50):
            score = 0.0
            
            # 频率得分（权重40%）
            freq_count = freq_counter.get(num, 0)
            if freq_count > 0:
                score += freq_count * 4.0  # 出现1次=4分，2次=8分
            
            # 避重得分（权重30%）- 最近5期出现过的号码降权
            if num not in recent_5:
                score += 3.0
            else:
                score -= 2.0  # 惩罚最近出现的
            
            # 五行平衡得分（权重20%）
            num_element = self._get_element(num)
            element_freq = element_counter.get(num_element, 0)
            if element_freq < 3:  # 最近10期出现少于3次的五行
                score += 2.0
            elif element_freq > 5:  # 最近10期出现超过5次的五行
                score -= 1.0
            
            # 区间平衡得分（权重10%）
            zone = self._get_zone(num)
            if zone == 'low' and zone_dist['low'] < 3:
                score += 1.0
            elif zone == 'mid' and zone_dist['mid'] < 4:
                score += 1.0
            elif zone == 'high' and zone_dist['high'] < 3:
                score += 1.0
            
            scores[num] = score
        
        # 排序并返回TOP15
        sorted_predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top15 = [num for num, _ in sorted_predictions[:15]]
        
        return top15
    
    def _get_element(self, num):
        """获取号码对应的五行"""
        element_map = {
            '金': [4, 5, 12, 13, 20, 21, 28, 29, 36, 37, 44, 45],
            '木': [1, 2, 9, 10, 17, 18, 25, 26, 33, 34, 41, 42, 49],
            '水': [6, 7, 14, 15, 22, 23, 30, 31, 38, 39, 46, 47],
            '火': [3, 11, 19, 27, 35, 43],
            '土': [8, 16, 24, 32, 40, 48]
        }
        
        for element, nums in element_map.items():
            if num in nums:
                return element
        return '未知'
    
    def _get_zone(self, num):
        """获取号码所属区间"""
        if 1 <= num <= 16:
            return 'low'
        elif 17 <= num <= 33:
            return 'mid'
        elif 34 <= num <= 49:
            return 'high'
        return 'unknown'
    
    def validate_recent_100_periods(self, csv_file='data/lucky_numbers.csv'):
        """验证TOP5策略B最近100期的预测成功率"""
        
        print("=" * 80)
        print("TOP5 策略B - 最近100期验证")
        print("=" * 80)
        print("\n策略说明：")
        print("  - 策略B使用最近10期数据进行精准预测")
        print("  - 综合考虑频率、避重、五行平衡、区间分布")
        print("  - 重点验证TOP5的命中情况")
        
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        total_records = len(df)
        
        # 确保有足够的数据
        if total_records < 101:
            print(f"\n错误：数据不足100期（当前只有{total_records}期）")
            return
        
        print(f"\n数据信息:")
        print(f"  总记录数: {total_records}")
        print(f"  验证期数: 100期")
        print(f"  验证范围: 第{total_records-100+1}期 到 第{total_records}期")
        
        # 统计结果
        top5_hits = 0
        top10_hits = 0
        top15_hits = 0
        total_tests = 0
        
        details = []
        hit_ranks = []  # 记录命中时的排名
        
        print(f"\n{'='*80}")
        print(f"开始验证...")
        print(f"{'='*80}\n")
        
        # 对最近100期进行验证
        for i in range(100):
            # 使用前N期数据预测第N+1期
            current_idx = total_records - 100 + i
            
            # 获取当期之前的所有数据（包括当期）
            train_data = df.iloc[:current_idx + 1]
            
            # 下一期的实际数字
            if current_idx + 1 < total_records:
                next_actual = int(df.iloc[current_idx + 1]['number'])
                next_date = df.iloc[current_idx + 1]['date']
                current_date = df.iloc[current_idx]['date']
                period_num = current_idx + 2  # 被预测的期数
            else:
                break
            
            # 使用训练数据进行预测
            numbers = train_data['number'].values
            elements = train_data['element'].values
            
            # 使用策略B预测
            top15_predictions = self.predict_strategy_b(numbers, elements)
            top10_predictions = top15_predictions[:10]
            top5_predictions = top15_predictions[:5]
            
            # 检查命中情况
            hit_top5 = next_actual in top5_predictions
            hit_top10 = next_actual in top10_predictions
            hit_top15 = next_actual in top15_predictions
            
            if hit_top5:
                top5_hits += 1
                top10_hits += 1
                top15_hits += 1
                rank = top5_predictions.index(next_actual) + 1
                hit_ranks.append(rank)
                hit_level = f"✅ TOP5 (#{rank})"
            elif hit_top10:
                top10_hits += 1
                top15_hits += 1
                rank = top10_predictions.index(next_actual) + 1
                hit_ranks.append(rank)
                hit_level = f"✓ TOP10 (#{rank})"
            elif hit_top15:
                top15_hits += 1
                rank = top15_predictions.index(next_actual) + 1
                hit_ranks.append(rank)
                hit_level = f"○ TOP15 (#{rank})"
            else:
                hit_level = "✗ 未命中"
                rank = None
            
            total_tests += 1
            
            # 记录详细信息
            detail = {
                '期数': period_num,
                '日期': next_date,
                '实际号码': next_actual,
                '命中情况': hit_level,
                'TOP5': '✓' if hit_top5 else '',
                'TOP10': '✓' if hit_top10 else '',
                'TOP15': '✓' if hit_top15 else '',
                '排名': rank if rank else '-',
                'TOP5预测': str(top5_predictions),
                'TOP10预测': str(top10_predictions)
            }
            details.append(detail)
            
            # 每20期输出一次进度
            if (i + 1) % 20 == 0:
                current_top5_rate = (top5_hits / total_tests) * 100
                print(f"已验证 {i+1}/100 期，当前TOP5成功率: {current_top5_rate:.2f}%")
        
        # 计算成功率
        top5_rate = (top5_hits / total_tests) * 100
        top10_rate = (top10_hits / total_tests) * 100
        top15_rate = (top15_hits / total_tests) * 100
        
        print(f"\n{'='*80}")
        print("验证结果统计")
        print(f"{'='*80}\n")
        
        print(f"总验证期数: {total_tests}")
        print(f"\n成功率统计:")
        print(f"  ⭐ TOP 5  成功率: {top5_rate:.2f}% ({top5_hits}/{total_tests})")
        print(f"     TOP 10 成功率: {top10_rate:.2f}% ({top10_hits}/{total_tests})")
        print(f"     TOP 15 成功率: {top15_rate:.2f}% ({top15_hits}/{total_tests})")
        
        # 命中质量分析
        if hit_ranks:
            avg_rank = sum(hit_ranks) / len(hit_ranks)
            print(f"\n命中质量分析:")
            print(f"  总命中次数: {len(hit_ranks)}")
            print(f"  平均排名: {avg_rank:.2f}")
            print(f"  最佳排名: {min(hit_ranks)}")
            print(f"  最差排名: {max(hit_ranks)}")
        
        # 分段统计（每25期一个区间）
        print(f"\n{'='*80}")
        print("分段成功率分析（每25期）")
        print(f"{'='*80}\n")
        
        for segment in range(4):
            start = segment * 25
            end = start + 25
            segment_details = details[start:end]
            
            seg_top5 = sum(1 for d in segment_details if d['TOP5'] == '✓')
            seg_top10 = sum(1 for d in segment_details if d['TOP10'] == '✓')
            seg_top15 = sum(1 for d in segment_details if d['TOP15'] == '✓')
            
            start_period = segment_details[0]['期数']
            end_period = segment_details[-1]['期数']
            
            print(f"第{segment+1}段（第{start_period}-{end_period}期）:")
            print(f"  TOP 5:  {seg_top5/25*100:.1f}% ({seg_top5}/25)")
            print(f"  TOP 10: {seg_top10/25*100:.1f}% ({seg_top10}/25)")
            print(f"  TOP 15: {seg_top15/25*100:.1f}% ({seg_top15}/25)\n")
        
        # 输出详细结果表格
        print(f"{'='*80}")
        print("详细验证记录（显示TOP5命中详情）")
        print(f"{'='*80}\n")
        
        print(f"{'期数':<8} {'日期':<12} {'实际':<6} {'命中情况':<18} {'排名':<6} {'TOP5预测'}")
        print("-" * 80)
        
        for detail in details:
            if detail['TOP5'] == '✓':  # 只显示TOP5命中的记录
                print(f"{detail['期数']:<8} {detail['日期']:<12} {detail['实际号码']:<6} "
                      f"{detail['命中情况']:<18} {detail['排名']:<6} {detail['TOP5预测']}")
        
        # 保存结果到CSV文件
        result_file = 'validate_top5_strategy_b_100periods_results.csv'
        result_df = pd.DataFrame(details)
        result_df.to_csv(result_file, index=False, encoding='utf-8-sig')
        
        print(f"\n详细结果已保存至: {result_file}")
        
        # 结论
        print(f"\n{'='*80}")
        print("结论")
        print(f"{'='*80}\n")
        
        if top5_rate >= 25:
            status = "✅ 优秀"
        elif top5_rate >= 20:
            status = "✓ 良好"
        elif top5_rate >= 15:
            status = "○ 一般"
        else:
            status = "✗ 需改进"
        
        print(f"TOP5策略B成功率: {top5_rate:.2f}% - {status}")
        print(f"\n评价标准:")
        print(f"  - ≥25%: 优秀 ✅")
        print(f"  - ≥20%: 良好 ✓")
        print(f"  - ≥15%: 一般 ○")
        print(f"  - <15%: 需改进 ✗")
        
        return {
            'total_tests': total_tests,
            'top5_hits': top5_hits,
            'top10_hits': top10_hits,
            'top15_hits': top15_hits,
            'top5_rate': top5_rate,
            'top10_rate': top10_rate,
            'top15_rate': top15_rate,
            'details': details
        }


def main():
    """主函数"""
    validator = Top5StrategyBValidator()
    results = validator.validate_recent_100_periods('data/lucky_numbers.csv')
    
    print(f"\n{'='*80}")
    print("验证完成!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
