"""
验证 TOP5 策略B (无五行) - 最近100期预测成功率
测试去掉五行权重后的效果
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime


class Top5StrategyBNoElementValidator:
    """TOP5策略B验证器 - 去除五行权重"""
    
    def predict_strategy_b_no_element(self, numbers, elements):
        """
        策略B: 基于最近10期数据的精准预测（无五行权重）
        只使用：频率、避重、区间分布
        """
        # 只使用最近10期数据
        recent_numbers = numbers[-10:] if len(numbers) >= 10 else numbers
        
        # 方法1: 频率分析（最近10期出现次数）
        freq_counter = Counter(recent_numbers)
        
        # 方法2: 最近5期去重分析（更近的历史）
        recent_5 = set(numbers[-5:]) if len(numbers) >= 5 else set(numbers)
        
        # 方法3: 号码区间分析
        zone_dist = {
            'low': sum(1 for n in recent_numbers if 1 <= n <= 16),
            'mid': sum(1 for n in recent_numbers if 17 <= n <= 33),
            'high': sum(1 for n in recent_numbers if 34 <= n <= 49)
        }
        
        # 综合评分（去除五行，重新分配权重）
        scores = {}
        for num in range(1, 50):
            score = 0.0
            
            # 频率得分（权重50%，原40%+五行20%的一半）
            freq_count = freq_counter.get(num, 0)
            if freq_count > 0:
                score += freq_count * 5.0  # 出现1次=5分，2次=10分
            
            # 避重得分（权重35%，原30%+五行20%的一半的一部分）
            if num not in recent_5:
                score += 3.5
            else:
                score -= 2.0  # 惩罚最近出现的
            
            # 区间平衡得分（权重15%，原10%+五行20%的剩余部分）
            zone = self._get_zone(num)
            if zone == 'low' and zone_dist['low'] < 3:
                score += 1.5
            elif zone == 'mid' and zone_dist['mid'] < 4:
                score += 1.5
            elif zone == 'high' and zone_dist['high'] < 3:
                score += 1.5
            
            scores[num] = score
        
        # 排序并返回TOP15
        sorted_predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top15 = [num for num, _ in sorted_predictions[:15]]
        
        return top15
    
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
        """验证TOP5策略B（无五行）最近100期的预测成功率"""
        
        print("=" * 80)
        print("TOP5 策略B (无五行) - 最近100期验证")
        print("=" * 80)
        print("\n策略说明：")
        print("  - 去除五行权重评估")
        print("  - 权重重新分配: 频率50% + 避重35% + 区间15%")
        print("  - 使用最近10期数据进行精准预测")
        
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
        hit_ranks = []
        
        print(f"\n{'='*80}")
        print(f"开始验证...")
        print(f"{'='*80}\n")
        
        # 对最近100期进行验证
        for i in range(100):
            current_idx = total_records - 100 + i
            train_data = df.iloc[:current_idx + 1]
            
            if current_idx + 1 < total_records:
                next_actual = int(df.iloc[current_idx + 1]['number'])
                next_date = df.iloc[current_idx + 1]['date']
                period_num = current_idx + 2
            else:
                break
            
            numbers = train_data['number'].values
            elements = train_data['element'].values
            
            # 使用无五行的策略B预测
            top15_predictions = self.predict_strategy_b_no_element(numbers, elements)
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
            
            detail = {
                '期数': period_num,
                '日期': next_date,
                '实际号码': next_actual,
                '命中情况': hit_level,
                'TOP5': '✓' if hit_top5 else '',
                'TOP10': '✓' if hit_top10 else '',
                'TOP15': '✓' if hit_top15 else '',
                '排名': rank if rank else '-',
                'TOP5预测': str(top5_predictions)
            }
            details.append(detail)
            
            if (i + 1) % 20 == 0:
                current_top5_rate = (top5_hits / total_tests) * 100
                print(f"已验证 {i+1}/100 期，当前TOP5成功率: {current_top5_rate:.2f}%")
        
        # 计算成功率
        top5_rate = (top5_hits / total_tests) * 100
        top10_rate = (top10_hits / total_tests) * 100
        top15_rate = (top15_hits / total_tests) * 100
        
        print(f"\n{'='*80}")
        print("验证结果统计（无五行版本）")
        print(f"{'='*80}\n")
        
        print(f"总验证期数: {total_tests}")
        print(f"\n成功率统计:")
        print(f"  ⭐ TOP 5  成功率: {top5_rate:.2f}% ({top5_hits}/{total_tests})")
        print(f"     TOP 10 成功率: {top10_rate:.2f}% ({top10_hits}/{total_tests})")
        print(f"     TOP 15 成功率: {top15_rate:.2f}% ({top15_hits}/{total_tests})")
        
        if hit_ranks:
            avg_rank = sum(hit_ranks) / len(hit_ranks)
            print(f"\n命中质量分析:")
            print(f"  总命中次数: {len(hit_ranks)}")
            print(f"  平均排名: {avg_rank:.2f}")
            print(f"  最佳排名: {min(hit_ranks)}")
            print(f"  最差排名: {max(hit_ranks)}")
        
        # 分段统计
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
        
        # 保存结果
        result_file = 'validate_top5_strategy_b_no_element_results.csv'
        result_df = pd.DataFrame(details)
        result_df.to_csv(result_file, index=False, encoding='utf-8-sig')
        
        print(f"详细结果已保存至: {result_file}")
        
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
    print("\n对比测试：去除五行权重的效果\n")
    
    validator = Top5StrategyBNoElementValidator()
    results = validator.validate_recent_100_periods('data/lucky_numbers.csv')
    
    print(f"\n{'='*80}")
    print("对比分析")
    print(f"{'='*80}\n")
    
    print("原始策略B（含五行）:")
    print("  - TOP5: 12.12% (12/99)")
    print("  - 权重: 频率40% + 避重30% + 五行20% + 区间10%")
    
    print(f"\n新策略B（无五行）:")
    print(f"  - TOP5: {results['top5_rate']:.2f}% ({results['top5_hits']}/{results['total_tests']})")
    print(f"  - 权重: 频率50% + 避重35% + 区间15%")
    
    diff = results['top5_rate'] - 12.12
    if diff > 0:
        print(f"\n✅ 提升: +{diff:.2f}% 📈")
    elif diff < 0:
        print(f"\n❌ 下降: {diff:.2f}% 📉")
    else:
        print(f"\n➡️ 持平: {diff:.2f}%")
    
    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
