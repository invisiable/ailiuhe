"""
Advanced Top 15 预测器 - 100期回测验证
目标：成功率达到60%
"""

from advanced_top15_predictor import AdvancedTop15Predictor
import pandas as pd
import numpy as np


def validate_100_periods():
    """100期回测验证"""
    
    print("=" * 80)
    print("Advanced Top 15 预测器 - 100期回测验证")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    total_periods = len(numbers)
    test_periods = min(100, total_periods - 50)  # 至少保留50期作为训练数据
    
    print(f"\n[数据概况]")
    print(f"   总数据量: {total_periods}期")
    print(f"   回测期数: {test_periods}期")
    print(f"   回测范围: 第{total_periods - test_periods + 1}期 - 第{total_periods}期")
    
    # 创建预测器
    predictor = AdvancedTop15Predictor()
    
    # 统计结果
    results = {
        'top5': 0,
        'top10': 0,
        'top15': 0,
        'details': []
    }
    
    print("\n" + "=" * 80)
    print("逐期回测结果")
    print("=" * 80)
    print(f"\n{'期数':<8}{'实际':<8}{'Top5命中':<12}{'Top10命中':<12}{'Top15命中':<12}{'排名':<8}")
    print("-" * 80)
    
    # 回测每一期
    for i in range(total_periods - test_periods, total_periods):
        period_num = i + 1
        actual = numbers[i]
        history = numbers[:i]
        
        # 跳过数据不足的情况
        if len(history) < 30:
            continue
        
        # 获取预测
        try:
            top15_pred = predictor.predict(history)
            
            # 检查命中
            top5_hit = actual in top15_pred[:5]
            top10_hit = actual in top15_pred[:10]
            top15_hit = actual in top15_pred
            
            # 更新统计
            if top5_hit:
                results['top5'] += 1
            if top10_hit:
                results['top10'] += 1
            if top15_hit:
                results['top15'] += 1
            
            # 获取排名
            rank = top15_pred.index(actual) + 1 if actual in top15_pred else '-'
            
            # 显示结果
            top5_mark = "[V]" if top5_hit else ""
            top10_mark = "[V]" if top10_hit else ""
            top15_mark = "[V]" if top15_hit else ""
            
            print(f"{period_num:<8}{actual:<8}{top5_mark:<12}{top10_mark:<12}{top15_mark:<12}{rank:<8}")
            
            # 记录详情
            results['details'].append({
                'period': period_num,
                'actual': actual,
                'predicted': top15_pred,
                'top5_hit': top5_hit,
                'top10_hit': top10_hit,
                'top15_hit': top15_hit,
                'rank': rank
            })
            
        except Exception as e:
            print(f"{period_num:<8}{actual:<8}{'错误':<12}{'错误':<12}{'错误':<12}{str(e):<8}")
    
    # 计算成功率
    total = len(results['details'])
    if total == 0:
        print("\n[错误] 没有有效的回测数据")
        return
    
    top5_rate = results['top5'] / total * 100
    top10_rate = results['top10'] / total * 100
    top15_rate = results['top15'] / total * 100
    
    print("\n" + "=" * 80)
    print("[回测统计]")
    print("=" * 80)
    
    print(f"\n总回测期数: {total}期")
    print(f"\n命中统计:")
    print(f"  Top 5:  {results['top5']}/{total} = {top5_rate:.1f}%")
    print(f"  Top 10: {results['top10']}/{total} = {top10_rate:.1f}%")
    print(f"  Top 15: {results['top15']}/{total} = {top15_rate:.1f}% {'[达标]' if top15_rate >= 60 else '[待提升]'}")
    
    # 与随机概率对比
    print(f"\n随机概率对比:")
    random_top5 = 5 / 49 * 100
    random_top10 = 10 / 49 * 100
    random_top15 = 15 / 49 * 100
    
    print(f"  Top 5:  {top5_rate:.1f}% vs 随机{random_top5:.1f}% (提升{top5_rate/random_top5:.2f}x)")
    print(f"  Top 10: {top10_rate:.1f}% vs 随机{random_top10:.1f}% (提升{top10_rate/random_top10:.2f}x)")
    print(f"  Top 15: {top15_rate:.1f}% vs 随机{random_top15:.1f}% (提升{top15_rate/random_top15:.2f}x)")
    
    # 目标达成情况
    print(f"\n[目标达成情况]")
    if top15_rate >= 60:
        print(f"  [成功] 已达成目标！Top 15成功率 {top15_rate:.1f}% >= 60%")
    else:
        gap = 60 - top15_rate
        print(f"  [警告] 距离目标还差 {gap:.1f}%")
        print(f"  需要命中 {int(np.ceil(gap * total / 100))} 期以上")
    
    # 最近20期表现
    recent_20_details = results['details'][-20:]
    recent_20_hits = sum(1 for d in recent_20_details if d['top15_hit'])
    recent_20_rate = recent_20_hits / len(recent_20_details) * 100 if recent_20_details else 0
    
    print(f"\n最近20期表现:")
    print(f"  命中: {recent_20_hits}/{len(recent_20_details)} = {recent_20_rate:.1f}%")
    
    # 保存详细结果
    save_details(results, total, top5_rate, top10_rate, top15_rate)
    
    return {
        'total': total,
        'top5_rate': top5_rate,
        'top10_rate': top10_rate,
        'top15_rate': top15_rate,
        'details': results['details']
    }


def save_details(results, total, top5_rate, top10_rate, top15_rate):
    """保存详细结果到CSV"""
    
    details_df = pd.DataFrame(results['details'])
    details_df.to_csv('advanced_top15_validation_100periods_results.csv', 
                      index=False, encoding='utf-8-sig')
    
    # 创建总结报告
    summary = {
        '指标': ['总期数', 'Top5成功率', 'Top10成功率', 'Top15成功率'],
        '数值': [total, f'{top5_rate:.1f}%', f'{top10_rate:.1f}%', f'{top15_rate:.1f}%']
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('advanced_top15_validation_summary.csv', 
                      index=False, encoding='utf-8-sig')
    
    print(f"\n[保存] 详细结果已保存:")
    print(f"   - advanced_top15_validation_100periods_results.csv")
    print(f"   - advanced_top15_validation_summary.csv")


def analyze_failure_patterns(results):
    """分析失败模式"""
    failures = [d for d in results['details'] if not d['top15_hit']]
    
    if not failures:
        print("\n[完美] 所有期数都命中了！")
        return
    
    print(f"\n" + "=" * 80)
    print(f"[未命中分析] ({len(failures)}期)")
    print("=" * 80)
    
    failure_numbers = [d['actual'] for d in failures]
    
    # 区域分析
    zones = {
        '极小(1-10)': [n for n in failure_numbers if 1 <= n <= 10],
        '小(11-20)': [n for n in failure_numbers if 11 <= n <= 20],
        '中(21-30)': [n for n in failure_numbers if 21 <= n <= 30],
        '大(31-40)': [n for n in failure_numbers if 31 <= n <= 40],
        '极大(41-49)': [n for n in failure_numbers if 41 <= n <= 49]
    }
    
    print(f"\n未命中号码区域分布:")
    for zone, nums in zones.items():
        if nums:
            print(f"  {zone}: {len(nums)}个 - {nums}")
    
    # 频率分析
    from collections import Counter
    freq = Counter(failure_numbers)
    if freq:
        print(f"\n未命中号码出现频率:")
        for num, count in freq.most_common(10):
            print(f"  {num}: {count}次")


if __name__ == '__main__':
    # 运行100期回测
    results = validate_100_periods()
    
    # 分析失败模式
    if results:
        analyze_failure_patterns(results)
    
    print("\n" + "=" * 80)
    print("[验证完成]")
    print("=" * 80)
