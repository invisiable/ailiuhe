"""
对比测试：原策略 vs 连续未中切换策略
测试哪种切换方案能获得更高的整体成功率
"""

import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy
from recommended_zodiac_top4_strategy_v2_1 import RecommendedZodiacTop4StrategyV2_1

def backtest_strategy(strategy, df, test_periods, strategy_name):
    """回测策略"""
    start_idx = len(df) - test_periods
    
    predictions = []
    hits = []
    switch_events = []
    
    for i in range(start_idx, len(df)):
        train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
        
        # 预测
        prediction = strategy.predict_top4(train_animals)
        predictions.append(prediction)
        
        # 实际结果
        actual = str(df.iloc[i]['animal']).strip()
        hit = actual in prediction['top4']
        hits.append(hit)
        
        # 更新性能
        strategy.update_performance(hit)
        
        # 检查切换（不限制检查频率，每期都检查）
        switched, msg = strategy.check_and_switch_model()
        if switched:
            switch_events.append({
                'period': i - start_idx + 1,
                'message': msg
            })
    
    # 统计结果
    total_hits = sum(hits)
    hit_rate = total_hits / len(hits) * 100
    
    # 计算收益
    total_investment = 16 * len(hits)
    total_profit = total_hits * 30 - (len(hits) - total_hits) * 16
    roi = (total_profit / total_investment) * 100
    
    # 最大连亏
    max_consecutive_losses = 0
    current_losses = 0
    for hit in hits:
        if hit:
            current_losses = 0
        else:
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    # 最大连中
    max_consecutive_wins = 0
    current_wins = 0
    for hit in hits:
        if hit:
            current_wins += 1
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_wins = 0
    
    return {
        'strategy_name': strategy_name,
        'total_periods': len(hits),
        'total_hits': total_hits,
        'hit_rate': hit_rate,
        'total_investment': total_investment,
        'total_profit': total_profit,
        'roi': roi,
        'max_consecutive_losses': max_consecutive_losses,
        'max_consecutive_wins': max_consecutive_wins,
        'switch_count': len(switch_events),
        'switch_events': switch_events,
        'predictions': predictions,
        'hits': hits
    }


def print_results(result):
    """打印结果"""
    print(f"\n{'='*90}")
    print(f"策略: {result['strategy_name']}")
    print(f"{'='*90}")
    print(f"测试期数: {result['total_periods']}")
    print(f"命中次数: {result['total_hits']}/{result['total_periods']}")
    print(f"命中率: {result['hit_rate']:.2f}%")
    print(f"\n收益分析:")
    print(f"  总投入: {result['total_investment']}元")
    print(f"  总收益: {result['total_profit']:+.0f}元")
    print(f"  投资回报率: {result['roi']:+.2f}%")
    print(f"\n风险指标:")
    print(f"  最大连续未中: {result['max_consecutive_losses']}期")
    print(f"  最大连续命中: {result['max_consecutive_wins']}期")
    print(f"\n模型切换:")
    print(f"  切换次数: {result['switch_count']}次")
    
    if result['switch_events']:
        print(f"\n  切换记录:")
        for event in result['switch_events'][:10]:  # 只显示前10次
            print(f"    第{event['period']}期: {event['message']}")
        if len(result['switch_events']) > 10:
            print(f"    ... 还有 {len(result['switch_events'])-10} 次切换")


def compare_results(result1, result2):
    """对比两个策略的结果"""
    print(f"\n{'='*90}")
    print(f"对比分析")
    print(f"{'='*90}")
    
    print(f"\n{'指标':<20} {'策略1(原版)':<30} {'策略2(连续未中)':<30} {'差异':<15}")
    print(f"{'-'*90}")
    
    # 命中率对比
    diff_hit_rate = result2['hit_rate'] - result1['hit_rate']
    print(f"{'命中率':<20} {result1['hit_rate']:<30.2f}% {result2['hit_rate']:<30.2f}% {diff_hit_rate:>+14.2f}%")
    
    # 收益对比
    diff_profit = result2['total_profit'] - result1['total_profit']
    print(f"{'总收益':<20} {result1['total_profit']:<30.0f}元 {result2['total_profit']:<30.0f}元 {diff_profit:>+14.0f}元")
    
    # ROI对比
    diff_roi = result2['roi'] - result1['roi']
    print(f"{'投资回报率':<20} {result1['roi']:<30.2f}% {result2['roi']:<30.2f}% {diff_roi:>+14.2f}%")
    
    # 最大连亏对比
    diff_max_loss = result2['max_consecutive_losses'] - result1['max_consecutive_losses']
    print(f"{'最大连亏':<20} {result1['max_consecutive_losses']:<30}期 {result2['max_consecutive_losses']:<30}期 {diff_max_loss:>+14}期")
    
    # 切换次数对比
    diff_switch = result2['switch_count'] - result1['switch_count']
    print(f"{'模型切换次数':<20} {result1['switch_count']:<30}次 {result2['switch_count']:<30}次 {diff_switch:>+14}次")
    
    print(f"\n{'='*90}")
    print("结论:")
    print(f"{'='*90}")
    
    if diff_hit_rate > 0:
        print(f"✅ 连续未中策略命中率更高，提升了 {diff_hit_rate:.2f}%")
    elif diff_hit_rate < 0:
        print(f"❌ 连续未中策略命中率更低，下降了 {abs(diff_hit_rate):.2f}%")
    else:
        print(f"⚖️ 两种策略命中率相同")
    
    if diff_profit > 0:
        print(f"✅ 连续未中策略收益更高，多赚了 {diff_profit:.0f}元")
    elif diff_profit < 0:
        print(f"❌ 连续未中策略收益更低，少赚了 {abs(diff_profit):.0f}元")
    else:
        print(f"⚖️ 两种策略收益相同")
    
    if diff_max_loss < 0:
        print(f"✅ 连续未中策略风险更小，最大连亏减少了 {abs(diff_max_loss)}期")
    elif diff_max_loss > 0:
        print(f"❌ 连续未中策略风险更大，最大连亏增加了 {diff_max_loss}期")
    else:
        print(f"⚖️ 两种策略风险相同")
    
    # 综合评分
    score1 = result1['hit_rate'] + result1['roi'] - result1['max_consecutive_losses'] * 2
    score2 = result2['hit_rate'] + result2['roi'] - result2['max_consecutive_losses'] * 2
    
    print(f"\n综合评分（命中率 + ROI - 最大连亏×2）:")
    print(f"  原策略: {score1:.2f}")
    print(f"  连续未中策略: {score2:.2f}")
    
    if score2 > score1:
        print(f"\n🏆 推荐使用：连续未中切换策略（v2.1）")
        print(f"   优势：{score2 - score1:.2f}分")
    elif score2 < score1:
        print(f"\n🏆 推荐使用：原策略（v2.0）")
        print(f"   优势：{score1 - score2:.2f}分")
    else:
        print(f"\n⚖️ 两种策略综合表现相当")


if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"数据加载完成: {len(df)}期")
    
    # 测试期数
    test_periods = 200
    print(f"测试期数: {test_periods}期\n")
    
    # 策略1：原版（命中率<30%切换）
    print("开始测试策略1：原版（最近10期命中率<30%切换）...")
    strategy1 = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
    result1 = backtest_strategy(strategy1, df, test_periods, "原版策略v2.0（命中率<30%切换）")
    print_results(result1)
    
    # 策略2：连续未中切换（连续4期未中切换）
    print("\n\n开始测试策略2：连续未中切换版（连续4期未中切换）...")
    strategy2 = RecommendedZodiacTop4StrategyV2_1(
        use_emergency_backup=True,
        consecutive_miss_threshold=4
    )
    result2 = backtest_strategy(strategy2, df, test_periods, "连续未中策略v2.1（连续4期未中切换）")
    print_results(result2)
    
    # 对比分析
    compare_results(result1, result2)
    
    # 测试不同的连续未中阈值
    print(f"\n\n{'='*90}")
    print("额外测试：不同连续未中阈值的效果")
    print(f"{'='*90}\n")
    
    thresholds = [3, 4, 5, 6]
    threshold_results = []
    
    for threshold in thresholds:
        print(f"测试阈值 = {threshold}...")
        strategy = RecommendedZodiacTop4StrategyV2_1(
            use_emergency_backup=True,
            consecutive_miss_threshold=threshold
        )
        result = backtest_strategy(strategy, df, test_periods, f"连续{threshold}期未中切换")
        threshold_results.append(result)
    
    print(f"\n{'阈值':<10} {'命中率':<12} {'总收益':<12} {'ROI':<12} {'最大连亏':<12} {'切换次数':<12}")
    print(f"{'-'*70}")
    for result in threshold_results:
        threshold = result['strategy_name'].split('连续')[1].split('期')[0]
        print(f"{threshold+'期':<10} {result['hit_rate']:<12.2f}% {result['total_profit']:<12.0f}元 "
              f"{result['roi']:<12.2f}% {result['max_consecutive_losses']:<12}期 {result['switch_count']:<12}次")
    
    # 找出最佳阈值
    best_result = max(threshold_results, key=lambda x: x['hit_rate'])
    print(f"\n🏆 最佳阈值：{best_result['strategy_name']}")
    print(f"   命中率：{best_result['hit_rate']:.2f}%")
    print(f"   总收益：{best_result['total_profit']:.0f}元")
    print(f"   ROI：{best_result['roi']:.2f}%")
