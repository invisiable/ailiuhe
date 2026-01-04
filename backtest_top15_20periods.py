"""
回测Top 15预测器 - 最近20期数据成功率验证
"""

from top15_predictor import Top15Predictor
import pandas as pd

def backtest_top15(test_periods=20):
    """回测Top 15预测器"""
    
    print("=" * 80)
    print(f"Top 15预测器回测 - 最近{test_periods}期")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    total_periods = len(numbers)
    print(f"\n总数据量: {total_periods}期")
    print(f"回测范围: 第{total_periods - test_periods + 1}期 - 第{total_periods}期")
    
    # 创建预测器
    predictor = Top15Predictor()
    
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
    print(f"\n{'期数':<8}{'实际':<8}{'Top5':<8}{'Top10':<8}{'Top15':<8}{'排名':<8}")
    print("-" * 80)
    
    # 回测每一期
    for i in range(total_periods - test_periods, total_periods):
        period_num = i + 1
        actual = numbers[i]
        history = numbers[:i]
        
        # 获取预测
        analysis = predictor.get_analysis(history)
        top15_pred = analysis['top15']
        
        # 检查命中
        if actual in top15_pred:
            rank = top15_pred.index(actual) + 1
            
            # 统计不同范围的命中
            if rank <= 5:
                results['top5'] += 1
                results['top10'] += 1
                results['top15'] += 1
                marker5 = "✅"
                marker10 = "✅"
                marker15 = "✅"
            elif rank <= 10:
                results['top10'] += 1
                results['top15'] += 1
                marker5 = "❌"
                marker10 = "✅"
                marker15 = "✅"
            else:
                results['top15'] += 1
                marker5 = "❌"
                marker10 = "❌"
                marker15 = "✅"
        else:
            rank = "-"
            marker5 = "❌"
            marker10 = "❌"
            marker15 = "❌"
        
        # 记录详情
        results['details'].append({
            'period': period_num,
            'actual': actual,
            'hit_top5': marker5 == "✅",
            'hit_top10': marker10 == "✅",
            'hit_top15': marker15 == "✅",
            'rank': rank
        })
        
        # 输出结果
        print(f"{period_num:<8}{actual:<8}{marker5:<8}{marker10:<8}{marker15:<8}{rank:<8}")
    
    # 统计汇总
    print("\n" + "=" * 80)
    print("统计汇总")
    print("=" * 80)
    
    total = len(results['details'])
    top5_rate = results['top5'] / total * 100
    top10_rate = results['top10'] / total * 100
    top15_rate = results['top15'] / total * 100
    
    # 随机概率
    random_top5 = 5 / 49 * 100
    random_top10 = 10 / 49 * 100
    random_top15 = 15 / 49 * 100
    
    print(f"\n命中统计 (最近{total}期):")
    print(f"  Top 5:  {results['top5']}/{total} = {top5_rate:.1f}%  (随机{random_top5:.1f}% → 提升{top5_rate/random_top5:.2f}x)")
    print(f"  Top 10: {results['top10']}/{total} = {top10_rate:.1f}% (随机{random_top10:.1f}% → 提升{top10_rate/random_top10:.2f}x)")
    print(f"  Top 15: {results['top15']}/{total} = {top15_rate:.1f}% (随机{random_top15:.1f}% → 提升{top15_rate/random_top15:.2f}x)")
    
    # 评级
    print("\n" + "=" * 80)
    print("性能评估")
    print("=" * 80)
    
    def get_grade(rate, target):
        if rate >= target:
            return "🏆 优秀"
        elif rate >= target * 0.9:
            return "✅ 良好"
        elif rate >= target * 0.8:
            return "🟢 合格"
        else:
            return "⚠️  一般"
    
    print(f"\nTop 5 ({top5_rate:.1f}%):  {get_grade(top5_rate, 30)}")
    print(f"Top 10 ({top10_rate:.1f}%): {get_grade(top10_rate, 40)}")
    print(f"Top 15 ({top15_rate:.1f}%): {get_grade(top15_rate, 60)}")
    
    # 目标达成情况
    print("\n" + "=" * 80)
    print("目标达成情况")
    print("=" * 80)
    
    if top15_rate >= 60:
        status = "✅ 已达标"
        message = f"Top 15成功率{top15_rate:.1f}%，已达到60%目标！"
    elif top15_rate >= 50:
        status = "🟡 接近目标"
        message = f"Top 15成功率{top15_rate:.1f}%，距离60%目标还差{60-top15_rate:.1f}%"
    else:
        status = "⚠️  未达标"
        message = f"Top 15成功率{top15_rate:.1f}%，距离60%目标还差{60-top15_rate:.1f}%"
    
    print(f"\n{status}")
    print(f"{message}")
    
    # 趋势分析
    print("\n" + "=" * 80)
    print("趋势分析")
    print("=" * 80)
    
    # 前10期和后10期对比
    if total >= 20:
        first_half = sum(1 for d in results['details'][:10] if d['hit_top15'])
        second_half = sum(1 for d in results['details'][10:] if d['hit_top15'])
        
        first_rate = first_half / 10 * 100
        second_rate = second_half / 10 * 100
        
        print(f"\n前10期: {first_half}/10 = {first_rate:.1f}%")
        print(f"后10期: {second_half}/10 = {second_rate:.1f}%")
        
        if second_rate > first_rate:
            print(f"趋势: 📈 上升 (+{second_rate-first_rate:.1f}%)")
        elif second_rate < first_rate:
            print(f"趋势: 📉 下降 (-{first_rate-second_rate:.1f}%)")
        else:
            print(f"趋势: ➡️  平稳")
    
    # 连续命中分析
    print("\n" + "=" * 80)
    print("连续命中分析")
    print("=" * 80)
    
    max_streak = 0
    current_streak = 0
    
    for detail in results['details']:
        if detail['hit_top15']:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    print(f"\n最长连续命中: {max_streak}期")
    
    # 命中率分布
    hits_in_windows = []
    window_size = 5
    for i in range(len(results['details']) - window_size + 1):
        window_hits = sum(1 for d in results['details'][i:i+window_size] if d['hit_top15'])
        hits_in_windows.append(window_hits)
    
    if hits_in_windows:
        avg_window_hits = sum(hits_in_windows) / len(hits_in_windows)
        print(f"滑动窗口(5期)平均命中: {avg_window_hits:.1f}期")
    
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    if top15_rate >= 60:
        print(f"\n✅ Top 15预测器性能优秀！")
        print(f"   在最近{total}期的回测中达到{top15_rate:.1f}%的成功率")
        print(f"   相比随机概率{random_top15:.1f}%提升了{top15_rate/random_top15:.2f}倍")
        print(f"   推荐在实际应用中使用")
    elif top15_rate >= 50:
        print(f"\n🟡 Top 15预测器性能良好")
        print(f"   在最近{total}期的回测中达到{top15_rate:.1f}%的成功率")
        print(f"   虽未达到60%目标，但已显著优于随机猜测")
        print(f"   建议结合其他策略使用")
    else:
        print(f"\n⚠️  Top 15预测器需要优化")
        print(f"   当前成功率{top15_rate:.1f}%，建议：")
        print(f"   1. 增加历史数据量")
        print(f"   2. 调整预测权重")
        print(f"   3. 考虑使用Top 20策略")
    
    print("\n" + "=" * 80 + "\n")
    
    return results


if __name__ == '__main__':
    # 回测最近20期
    results = backtest_top15(test_periods=20)
