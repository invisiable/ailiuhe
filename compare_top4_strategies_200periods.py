"""
对比生肖TOP4投注 vs 生肖TOP4动态择优
最近200期验证对比
"""

import pandas as pd
import numpy as np
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy
from ensemble_select_best_predictor import EnsembleSelectBestPredictor


def test_strategy_1_recommended():
    """方案1：生肖TOP4投注（推荐策略v2.0）"""
    print("\n" + "="*80)
    print("方案1：生肖TOP4投注（推荐策略v2.0）")
    print("="*80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    # 最近200期
    test_periods = 200
    start_idx = len(df) - test_periods
    
    # 创建推荐策略实例
    strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
    
    # 回测数据
    hit_records = []
    predictor_records = []
    
    print(f"测试期数: {test_periods}期")
    print(f"日期范围: {df.iloc[start_idx]['date']} ~ {df.iloc[-1]['date']}")
    print("\n开始回测...")
    
    for i in range(start_idx, len(df)):
        # 使用i之前的数据进行预测
        train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
        
        # 使用推荐策略进行预测
        prediction = strategy.predict_top4(train_animals)
        top4 = prediction['top4']
        predictor_name = prediction['predictor']
        
        # 实际结果
        actual = str(df.iloc[i]['animal']).strip()
        
        # 判断命中
        hit = actual in top4
        hit_records.append(hit)
        predictor_records.append(predictor_name)
        
        # 更新策略性能监控
        strategy.update_performance(hit)
        
        # 每10期检查是否需要切换模型
        if (i - start_idx + 1) % 10 == 0:
            strategy.check_and_switch_model()
    
    # 计算统计数据
    hits = sum(hit_records)
    hit_rate = hits / len(hit_records) * 100
    
    # 计算财务数据
    BET_PER_ZODIAC = 4
    WIN_REWARD = 47
    cost_per_period = 4 * BET_PER_ZODIAC  # 16元
    
    total_investment = cost_per_period * len(hit_records)
    total_rewards = hits * WIN_REWARD
    total_profit = total_rewards - total_investment
    roi = (total_profit / total_investment) * 100
    
    # 计算连续不中
    max_consecutive_misses = 0
    current_misses = 0
    for hit in hit_records:
        if not hit:
            current_misses += 1
            max_consecutive_misses = max(max_consecutive_misses, current_misses)
        else:
            current_misses = 0
    
    # 统计预测器使用情况
    predictor_usage = {}
    for pred in predictor_records:
        predictor_usage[pred] = predictor_usage.get(pred, 0) + 1
    
    # 输出结果
    print(f"\n✅ 回测完成！")
    print(f"\n【命中统计】")
    print(f"  命中次数: {hits}/{len(hit_records)}")
    print(f"  命中率: {hit_rate:.2f}%")
    print(f"  最大连续不中: {max_consecutive_misses}期")
    
    print(f"\n【财务统计】")
    print(f"  总投入: {total_investment:.0f}元")
    print(f"  总奖励: {total_rewards:.0f}元")
    print(f"  净收益: {total_profit:+.0f}元")
    print(f"  ROI: {roi:+.2f}%")
    
    print(f"\n【预测器使用分布】")
    for pred, count in sorted(predictor_usage.items(), key=lambda x: x[1], reverse=True):
        percentage = count / test_periods * 100
        print(f"  {pred}: {count}次 ({percentage:.1f}%)")
    
    return {
        'name': '生肖TOP4投注（推荐策略v2.0）',
        'hits': hits,
        'total': len(hit_records),
        'hit_rate': hit_rate,
        'max_consecutive_misses': max_consecutive_misses,
        'total_investment': total_investment,
        'total_profit': total_profit,
        'roi': roi,
        'predictor_usage': predictor_usage
    }


def test_strategy_2_ensemble():
    """方案2：生肖TOP4动态择优"""
    print("\n" + "="*80)
    print("方案2：生肖TOP4动态择优")
    print("="*80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    # 最近200期
    test_periods = 200
    start_idx = len(df) - test_periods
    
    # 创建动态择优预测器实例
    predictor = EnsembleSelectBestPredictor(window_size=20)
    
    # 回测数据
    hit_records = []
    predictor_records = []
    
    print(f"测试期数: {test_periods}期")
    print(f"日期范围: {df.iloc[start_idx]['date']} ~ {df.iloc[-1]['date']}")
    print("\n开始回测...")
    
    for i in range(start_idx, len(df)):
        # 使用i之前的数据进行预测
        train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
        
        # 使用动态择优策略进行预测
        result = predictor.predict_top4(train_animals)
        top4 = result['top4']
        predictor_name = result['predictor']
        
        # 实际结果
        actual = str(df.iloc[i]['animal']).strip()
        
        # 判断命中
        hit = actual in top4
        hit_records.append(hit)
        predictor_records.append(predictor_name)
        
        # 更新性能统计
        details = result.get('details', {})
        predictor.update_performance(actual, details)
    
    # 计算统计数据
    hits = sum(hit_records)
    hit_rate = hits / len(hit_records) * 100
    
    # 计算财务数据
    BET_PER_ZODIAC = 4
    WIN_REWARD = 47
    cost_per_period = 4 * BET_PER_ZODIAC  # 16元
    
    total_investment = cost_per_period * len(hit_records)
    total_rewards = hits * WIN_REWARD
    total_profit = total_rewards - total_investment
    roi = (total_profit / total_investment) * 100
    
    # 计算连续不中
    max_consecutive_misses = 0
    current_misses = 0
    for hit in hit_records:
        if not hit:
            current_misses += 1
            max_consecutive_misses = max(max_consecutive_misses, current_misses)
        else:
            current_misses = 0
    
    # 统计预测器使用情况
    predictor_usage = {}
    for pred in predictor_records:
        predictor_usage[pred] = predictor_usage.get(pred, 0) + 1
    
    # 输出结果
    print(f"\n✅ 回测完成！")
    print(f"\n【命中统计】")
    print(f"  命中次数: {hits}/{len(hit_records)}")
    print(f"  命中率: {hit_rate:.2f}%")
    print(f"  最大连续不中: {max_consecutive_misses}期")
    
    print(f"\n【财务统计】")
    print(f"  总投入: {total_investment:.0f}元")
    print(f"  总奖励: {total_rewards:.0f}元")
    print(f"  净收益: {total_profit:+.0f}元")
    print(f"  ROI: {roi:+.2f}%")
    
    print(f"\n【预测器使用分布】")
    for pred, count in sorted(predictor_usage.items(), key=lambda x: x[1], reverse=True):
        percentage = count / test_periods * 100
        print(f"  {pred}: {count}次 ({percentage:.1f}%)")
    
    return {
        'name': '生肖TOP4动态择优',
        'hits': hits,
        'total': len(hit_records),
        'hit_rate': hit_rate,
        'max_consecutive_misses': max_consecutive_misses,
        'total_investment': total_investment,
        'total_profit': total_profit,
        'roi': roi,
        'predictor_usage': predictor_usage
    }


def compare_results(result1, result2):
    """对比两个策略的结果"""
    print("\n" + "="*80)
    print("📊 综合对比分析（最近200期）")
    print("="*80)
    
    # 创建对比表格
    print(f"\n{'指标':<20} {'方案1（推荐策略）':<25} {'方案2（动态择优）':<25} {'差异':<20}")
    print("-" * 100)
    
    # 命中率对比
    hit_rate_diff = result1['hit_rate'] - result2['hit_rate']
    hit_rate_marker = "🏆" if result1['hit_rate'] > result2['hit_rate'] else ""
    hit_rate_marker2 = "🏆" if result2['hit_rate'] > result1['hit_rate'] else ""
    print(f"{'命中率':<20} {result1['hit_rate']:>6.2f}% {hit_rate_marker:<17} {result2['hit_rate']:>6.2f}% {hit_rate_marker2:<17} {hit_rate_diff:>+6.2f}%")
    
    # 命中次数
    print(f"{'命中次数':<20} {result1['hits']:>6}次 / {result1['total']}期{'':<10} {result2['hits']:>6}次 / {result2['total']}期{'':<10} {result1['hits'] - result2['hits']:>+6}次")
    
    # ROI对比
    roi_diff = result1['roi'] - result2['roi']
    roi_marker = "🏆" if result1['roi'] > result2['roi'] else ""
    roi_marker2 = "🏆" if result2['roi'] > result1['roi'] else ""
    print(f"{'ROI':<20} {result1['roi']:>+6.2f}% {roi_marker:<17} {result2['roi']:>+6.2f}% {roi_marker2:<17} {roi_diff:>+6.2f}%")
    
    # 总收益对比
    profit_diff = result1['total_profit'] - result2['total_profit']
    profit_marker = "🏆" if result1['total_profit'] > result2['total_profit'] else ""
    profit_marker2 = "🏆" if result2['total_profit'] > result1['total_profit'] else ""
    print(f"{'总收益':<20} {result1['total_profit']:>+8.0f}元 {profit_marker:<15} {result2['total_profit']:>+8.0f}元 {profit_marker2:<15} {profit_diff:>+7.0f}元")
    
    # 总投入
    print(f"{'总投入':<20} {result1['total_investment']:>8.0f}元{'':<17} {result2['total_investment']:>8.0f}元{'':<17} {'相同':<20}")
    
    # 最大连续不中对比
    miss_diff = result1['max_consecutive_misses'] - result2['max_consecutive_misses']
    miss_marker = "🏆" if result1['max_consecutive_misses'] < result2['max_consecutive_misses'] else ""
    miss_marker2 = "🏆" if result2['max_consecutive_misses'] < result1['max_consecutive_misses'] else ""
    print(f"{'最大连续不中':<20} {result1['max_consecutive_misses']:>6}期 {miss_marker:<17} {result2['max_consecutive_misses']:>6}期 {miss_marker2:<17} {miss_diff:>+6}期")
    
    print("\n" + "="*80)
    print("🏆 综合评分")
    print("="*80)
    
    # 计算综合得分
    score1 = 0
    score2 = 0
    
    # 命中率权重：40%
    if result1['hit_rate'] > result2['hit_rate']:
        score1 += 40
        print(f"✅ 命中率优势: 方案1 ({result1['hit_rate']:.2f}% vs {result2['hit_rate']:.2f}%) +40分")
    else:
        score2 += 40
        print(f"✅ 命中率优势: 方案2 ({result2['hit_rate']:.2f}% vs {result1['hit_rate']:.2f}%) +40分")
    
    # ROI权重：35%
    if result1['roi'] > result2['roi']:
        score1 += 35
        print(f"✅ ROI优势: 方案1 ({result1['roi']:+.2f}% vs {result2['roi']:+.2f}%) +35分")
    else:
        score2 += 35
        print(f"✅ ROI优势: 方案2 ({result2['roi']:+.2f}% vs {result1['roi']:+.2f}%) +35分")
    
    # 风险控制（最大连不中）权重：25%
    if result1['max_consecutive_misses'] < result2['max_consecutive_misses']:
        score1 += 25
        print(f"✅ 风险控制优势: 方案1 (最大连不中{result1['max_consecutive_misses']}期 vs {result2['max_consecutive_misses']}期) +25分")
    else:
        score2 += 25
        print(f"✅ 风险控制优势: 方案2 (最大连不中{result2['max_consecutive_misses']}期 vs {result1['max_consecutive_misses']}期) +25分")
    
    print(f"\n{'='*80}")
    print(f"方案1总分: {score1}分")
    print(f"方案2总分: {score2}分")
    print(f"{'='*80}")
    
    # 确定最优方案
    if score1 > score2:
        winner = result1['name']
        print(f"\n🏆 推荐方案: {winner}")
        print(f"优势: 命中率{result1['hit_rate']:.2f}%, ROI {result1['roi']:+.2f}%, 总收益{result1['total_profit']:+.0f}元")
    elif score2 > score1:
        winner = result2['name']
        print(f"\n🏆 推荐方案: {winner}")
        print(f"优势: 命中率{result2['hit_rate']:.2f}%, ROI {result2['roi']:+.2f}%, 总收益{result2['total_profit']:+.0f}元")
    else:
        print(f"\n⚖️ 两个方案综合评分相同，各有优势")
        print(f"建议: 根据个人风险偏好选择")
    
    print(f"\n{'='*80}")
    print("💡 选择建议")
    print(f"{'='*80}")
    
    if result1['hit_rate'] > result2['hit_rate'] and result1['roi'] > result2['roi']:
        print(f"✅ 明确推荐: {result1['name']}")
        print(f"   理由: 命中率和ROI双优，综合表现更佳")
    elif result2['hit_rate'] > result1['hit_rate'] and result2['roi'] > result1['roi']:
        print(f"✅ 明确推荐: {result2['name']}")
        print(f"   理由: 命中率和ROI双优，综合表现更佳")
    else:
        print(f"根据实际需求选择：")
        if result1['hit_rate'] > result2['hit_rate']:
            print(f"  • 追求高命中率 → {result1['name']}")
        else:
            print(f"  • 追求高命中率 → {result2['name']}")
        
        if result1['max_consecutive_misses'] < result2['max_consecutive_misses']:
            print(f"  • 追求低风险 → {result1['name']}")
        else:
            print(f"  • 追求低风险 → {result2['name']}")
        
        if result1['total_profit'] > result2['total_profit']:
            print(f"  • 追求高收益 → {result1['name']}")
        else:
            print(f"  • 追求高收益 → {result2['name']}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("生肖TOP4投注策略对比测试 - 最近200期验证")
    print("="*80)
    
    # 测试方案1
    result1 = test_strategy_1_recommended()
    
    # 测试方案2
    result2 = test_strategy_2_ensemble()
    
    # 对比结果
    compare_results(result1, result2)
    
    print("\n✅ 测试完成！")
