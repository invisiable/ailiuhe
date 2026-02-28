"""
综合对比验证：精准TOP15投注 vs 生肖TOP4投注（两种方案）
在相同的200期数据上对比三种策略的成功率
"""

import pandas as pd
import numpy as np
from datetime import datetime
from precise_top15_predictor import PreciseTop15Predictor
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy
from ensemble_select_best_predictor import EnsembleSelectBestPredictor

def load_data():
    """加载数据"""
    df = pd.read_csv('data/lucky_numbers.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def test_precise_top15(df, test_periods=200):
    """测试精准TOP15投注策略"""
    print("\n" + "="*60)
    print("策略1: 精准TOP15投注")
    print("="*60)
    
    predictor = PreciseTop15Predictor()
    
    # 使用最近200期进行测试
    total_periods = len(df)
    start_idx = total_periods - test_periods
    
    results = []
    total_bet = 0
    total_reward = 0
    consecutive_misses = 0
    max_consecutive_misses = 0
    
    # Fibonacci倍投序列
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    multiplier_idx = 0
    
    for i in range(start_idx, total_periods):
        train_data = df.iloc[:i]
        test_row = df.iloc[i]
        
        # 使用训练数据进行预测（传入历史号码列表）
        history_numbers = train_data['number'].tolist()
        top15_predictions = predictor.predict(history_numbers)
        
        # 判断是否命中
        actual_number = test_row['number']
        hit = actual_number in top15_predictions
        
        # 计算投注和奖励 (15元 * 倍投 * 15个号码)
        multiplier = fib_sequence[min(multiplier_idx, len(fib_sequence)-1)]
        bet_amount = 15 * multiplier
        reward = 47 * multiplier if hit else 0
        profit = reward - bet_amount
        
        total_bet += bet_amount
        total_reward += reward
        
        # 更新预测器的性能追踪
        predictor.update_performance(top15_predictions, actual_number)
        
        if hit:
            consecutive_misses = 0
            multiplier_idx = 0
        else:
            consecutive_misses += 1
            multiplier_idx += 1
            max_consecutive_misses = max(max_consecutive_misses, consecutive_misses)
        
        results.append({
            'date': test_row['date'],
            'period': i - start_idx + 1,
            'actual': actual_number,
            'predictions': top15_predictions,
            'hit': hit,
            'multiplier': multiplier,
            'bet': bet_amount,
            'reward': reward,
            'profit': profit,
            'cumulative_profit': total_reward - total_bet
        })
    
    # 统计结果
    df_results = pd.DataFrame(results)
    hit_count = df_results['hit'].sum()
    hit_rate = hit_count / len(df_results) * 100
    total_profit = total_reward - total_bet
    roi = (total_profit / total_bet * 100) if total_bet > 0 else 0
    
    print(f"\n测试周期: {results[0]['date'].strftime('%Y/%m/%d')} ~ {results[-1]['date'].strftime('%Y/%m/%d')}")
    print(f"总期数: {len(results)}期")
    print(f"命中次数: {hit_count}次")
    print(f"命中率: {hit_rate:.2f}%")
    print(f"最大连续不中: {max_consecutive_misses}期")
    print(f"总投入: {total_bet}元")
    print(f"总奖励: {total_reward}元")
    print(f"净收益: {total_profit:+d}元")
    print(f"投资回报率(ROI): {roi:+.2f}%")
    
    return {
        'strategy': '精准TOP15投注',
        'hit_count': hit_count,
        'total_periods': len(results),
        'hit_rate': hit_rate,
        'max_consecutive_misses': max_consecutive_misses,
        'total_bet': total_bet,
        'total_reward': total_reward,
        'profit': total_profit,
        'roi': roi
    }

def test_recommended_strategy(df, test_periods=200):
    """测试生肖TOP4推荐策略v2.0"""
    print("\n" + "="*60)
    print("策略2: 生肖TOP4投注 - 推荐策略v2.0")
    print("="*60)
    
    strategy = RecommendedZodiacTop4Strategy()
    
    # 使用最近200期进行测试
    total_periods = len(df)
    start_idx = total_periods - test_periods
    
    results = []
    total_bet = 0
    total_reward = 0
    consecutive_misses = 0
    max_consecutive_misses = 0
    
    predictor_usage = {}
    
    # Fibonacci倍投序列
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    multiplier_idx = 0
    
    for i in range(start_idx, total_periods):
        train_data = df.iloc[:i]
        test_row = df.iloc[i]
        
        # 使用推荐策略获取TOP4
        history_animals = train_data['animal'].tolist()
        result = strategy.predict_top4(history_animals)
        top4_zodiacs = result['top4']
        predictor_name = result['predictor']
        predictor_usage[predictor_name] = predictor_usage.get(predictor_name, 0) + 1
        
        # 判断是否命中
        actual_zodiac = test_row['animal']
        hit = actual_zodiac in top4_zodiacs
        
        # 计算投注和奖励 (4元 * 倍投 * 4个生肖)
        multiplier = fib_sequence[min(multiplier_idx, len(fib_sequence)-1)]
        bet_amount = 16 * multiplier
        reward = 47 * multiplier if hit else 0
        profit = reward - bet_amount
        
        total_bet += bet_amount
        total_reward += reward
        
        # 更新预测器的性能追踪
        strategy.update_performance(hit)
        
        # 每10期检查是否需要切换模型
        if (i - start_idx + 1) % 10 == 0:
            strategy.check_and_switch_model()
        
        if hit:
            consecutive_misses = 0
            multiplier_idx = 0
        else:
            consecutive_misses += 1
            multiplier_idx += 1
            max_consecutive_misses = max(max_consecutive_misses, consecutive_misses)
        
        results.append({
            'date': test_row['date'],
            'period': i - start_idx + 1,
            'actual': actual_zodiac,
            'predictions': top4_zodiacs,
            'hit': hit,
            'predictor': predictor_name,
            'multiplier': multiplier,
            'bet': bet_amount,
            'reward': reward,
            'profit': profit,
            'cumulative_profit': total_reward - total_bet
        })
    
    # 统计结果
    df_results = pd.DataFrame(results)
    hit_count = df_results['hit'].sum()
    hit_rate = hit_count / len(df_results) * 100
    total_profit = total_reward - total_bet
    roi = (total_profit / total_bet * 100) if total_bet > 0 else 0
    
    print(f"\n测试周期: {results[0]['date'].strftime('%Y/%m/%d')} ~ {results[-1]['date'].strftime('%Y/%m/%d')}")
    print(f"总期数: {len(results)}期")
    print(f"命中次数: {hit_count}次")
    print(f"命中率: {hit_rate:.2f}%")
    print(f"最大连续不中: {max_consecutive_misses}期")
    print(f"总投入: {total_bet}元")
    print(f"总奖励: {total_reward}元")
    print(f"净收益: {total_profit:+d}元")
    print(f"投资回报率(ROI): {roi:+.2f}%")
    
    print(f"\n预测器使用分布:")
    for predictor_name, count in sorted(predictor_usage.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(results) * 100
        print(f"  {predictor_name}: {count}次 ({percentage:.1f}%)")
    
    return {
        'strategy': '生肖TOP4推荐策略',
        'hit_count': hit_count,
        'total_periods': len(results),
        'hit_rate': hit_rate,
        'max_consecutive_misses': max_consecutive_misses,
        'total_bet': total_bet,
        'total_reward': total_reward,
        'profit': total_profit,
        'roi': roi,
        'predictor_usage': predictor_usage
    }

def test_ensemble_strategy(df, test_periods=200):
    """测试生肖TOP4动态择优策略"""
    print("\n" + "="*60)
    print("策略3: 生肖TOP4投注 - 动态择优")
    print("="*60)
    
    predictor = EnsembleSelectBestPredictor()
    
    # 使用最近200期进行测试
    total_periods = len(df)
    start_idx = total_periods - test_periods
    
    results = []
    total_bet = 0
    total_reward = 0
    consecutive_misses = 0
    max_consecutive_misses = 0
    
    predictor_usage = {}
    switch_count = 0
    last_predictor = None
    
    # Fibonacci倍投序列
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    multiplier_idx = 0
    
    for i in range(start_idx, total_periods):
        train_data = df.iloc[:i]
        test_row = df.iloc[i]
        
        # 使用集成择优策略获取TOP4
        history_animals = train_data['animal'].tolist()
        result = predictor.predict_top4(history_animals)
        top4_zodiacs = result['top4']
        best_predictor_name = result['predictor']
        prediction_details = result.get('details', {})
        
        # 统计预测器切换
        if last_predictor is not None and best_predictor_name != last_predictor:
            switch_count += 1
        last_predictor = best_predictor_name
        
        predictor_usage[best_predictor_name] = predictor_usage.get(best_predictor_name, 0) + 1
        
        # 判断是否命中
        actual_zodiac = test_row['animal']
        hit = actual_zodiac in top4_zodiacs
        
        # 计算投注和奖励 (4元 * 倍投 * 4个生肖)
        multiplier = fib_sequence[min(multiplier_idx, len(fib_sequence)-1)]
        bet_amount = 16 * multiplier
        reward = 47 * multiplier if hit else 0
        profit = reward - bet_amount
        
        total_bet += bet_amount
        total_reward += reward
        
        # 更新预测器的性能追踪
        predictor.update_performance(actual_zodiac, prediction_details)
        
        if hit:
            consecutive_misses = 0
            multiplier_idx = 0
        else:
            consecutive_misses += 1
            multiplier_idx += 1
            max_consecutive_misses = max(max_consecutive_misses, consecutive_misses)
        
        results.append({
            'date': test_row['date'],
            'period': i - start_idx + 1,
            'actual': actual_zodiac,
            'predictions': top4_zodiacs,
            'hit': hit,
            'predictor': best_predictor_name,
            'multiplier': multiplier,
            'bet': bet_amount,
            'reward': reward,
            'profit': profit,
            'cumulative_profit': total_reward - total_bet
        })
    
    # 统计结果
    df_results = pd.DataFrame(results)
    hit_count = df_results['hit'].sum()
    hit_rate = hit_count / len(df_results) * 100
    total_profit = total_reward - total_bet
    roi = (total_profit / total_bet * 100) if total_bet > 0 else 0
    
    print(f"\n测试周期: {results[0]['date'].strftime('%Y/%m/%d')} ~ {results[-1]['date'].strftime('%Y/%m/%d')}")
    print(f"总期数: {len(results)}期")
    print(f"命中次数: {hit_count}次")
    print(f"命中率: {hit_rate:.2f}%")
    print(f"最大连续不中: {max_consecutive_misses}期")
    print(f"总投入: {total_bet}元")
    print(f"总奖励: {total_reward}元")
    print(f"净收益: {total_profit:+d}元")
    print(f"投资回报率(ROI): {roi:+.2f}%")
    print(f"策略切换次数: {switch_count}次")
    
    print(f"\n预测器使用分布:")
    for predictor_name, count in sorted(predictor_usage.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(results) * 100
        print(f"  {predictor_name}: {count}次 ({percentage:.1f}%)")
    
    return {
        'strategy': '生肖TOP4动态择优',
        'hit_count': hit_count,
        'total_periods': len(results),
        'hit_rate': hit_rate,
        'max_consecutive_misses': max_consecutive_misses,
        'total_bet': total_bet,
        'total_reward': total_reward,
        'profit': total_profit,
        'roi': roi,
        'switch_count': switch_count,
        'predictor_usage': predictor_usage
    }

def compare_all_strategies(results):
    """综合对比所有策略"""
    print("\n" + "="*80)
    print("三大策略综合对比分析")
    print("="*80)
    
    # 创建对比表格
    print("\n📊 核心指标对比:")
    print("-" * 80)
    print(f"{'策略名称':<20} {'命中率':<12} {'ROI':<12} {'净收益':<12} {'最大连不中':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['strategy']:<20} "
              f"{result['hit_rate']:<11.2f}% "
              f"{result['roi']:<11.2f}% "
              f"{result['profit']:<11,d}元 "
              f"{result['max_consecutive_misses']:<10}期")
    
    print("-" * 80)
    
    # 找出最优策略
    print("\n🏆 分项最优:")
    
    best_hit_rate = max(results, key=lambda x: x['hit_rate'])
    print(f"✓ 最高命中率: {best_hit_rate['strategy']} ({best_hit_rate['hit_rate']:.2f}%)")
    
    best_roi = max(results, key=lambda x: x['roi'])
    print(f"✓ 最高ROI: {best_roi['strategy']} ({best_roi['roi']:+.2f}%)")
    
    best_profit = max(results, key=lambda x: x['profit'])
    print(f"✓ 最高收益: {best_profit['strategy']} ({best_profit['profit']:+,d}元)")
    
    best_risk = min(results, key=lambda x: x['max_consecutive_misses'])
    print(f"✓ 最低风险: {best_risk['strategy']} (最大连不中{best_risk['max_consecutive_misses']}期)")
    
    # 综合评分 (命中率40% + ROI35% + 收益25%)
    print("\n📈 综合评分体系 (命中率40% + ROI35% + 收益25%):")
    print("-" * 60)
    
    max_hit_rate = max(r['hit_rate'] for r in results)
    max_roi = max(r['roi'] for r in results)
    max_profit = max(r['profit'] for r in results)
    
    scored_results = []
    for result in results:
        hit_score = (result['hit_rate'] / max_hit_rate) * 40 if max_hit_rate > 0 else 0
        roi_score = (result['roi'] / max_roi) * 35 if max_roi > 0 else 0
        profit_score = (result['profit'] / max_profit) * 25 if max_profit > 0 else 0
        total_score = hit_score + roi_score + profit_score
        
        scored_results.append({
            'strategy': result['strategy'],
            'total_score': total_score,
            'hit_score': hit_score,
            'roi_score': roi_score,
            'profit_score': profit_score
        })
        
        print(f"{result['strategy']:<20} 总分: {total_score:.1f} "
              f"(命中率{hit_score:.1f} + ROI{roi_score:.1f} + 收益{profit_score:.1f})")
    
    # 最终推荐
    winner = max(scored_results, key=lambda x: x['total_score'])
    print("\n" + "="*80)
    print(f"🎯 最终推荐: {winner['strategy']}")
    print("="*80)
    
    # 详细分析
    winner_detail = next(r for r in results if r['strategy'] == winner['strategy'])
    print(f"\n推荐理由:")
    print(f"  • 命中率: {winner_detail['hit_rate']:.2f}%")
    print(f"  • 投资回报率: {winner_detail['roi']:+.2f}%")
    print(f"  • 净收益: {winner_detail['profit']:+,d}元")
    print(f"  • 风险控制: 最大连续不中{winner_detail['max_consecutive_misses']}期")
    print(f"  • 综合评分: {winner['total_score']:.1f}分")
    
    # 策略对比建议
    print("\n💡 投资建议:")
    if winner['strategy'] == '精准TOP15投注':
        print("  • 适合追求稳定回报的投资者")
        print("  • 投入金额较大，需要充足资金")
        print("  • 15个号码覆盖面广，命中机会多")
    elif winner['strategy'] == '生肖TOP4推荐策略':
        print("  • 适合追求稳健策略的投资者")
        print("  • 4个生肖投入适中，风险可控")
        print("  • 多预测器协同，应急机制完善")
    else:  # 动态择优
        print("  • 适合追求智能调优的投资者")
        print("  • 动态切换预测器，快速适应变化")
        print("  • 频繁调整策略，持续优化收益")

def main():
    """主函数"""
    print("="*80)
    print("综合策略对比验证: 精准TOP15投注 vs 生肖TOP4投注(两种方案)")
    print("="*80)
    print("验证期数: 最近200期")
    print("倍投策略: Fibonacci数列 (1,1,2,3,5,8,13,21,34,55,89,144)")
    print("="*80)
    
    # 加载数据
    df = load_data()
    print(f"\n数据加载完成: 共{len(df)}期数据")
    
    # 测试三种策略
    results = []
    
    # 1. 测试精准TOP15
    result1 = test_precise_top15(df, test_periods=200)
    results.append(result1)
    
    # 2. 测试推荐策略v2.0
    result2 = test_recommended_strategy(df, test_periods=200)
    results.append(result2)
    
    # 3. 测试动态择优
    result3 = test_ensemble_strategy(df, test_periods=200)
    results.append(result3)
    
    # 综合对比
    compare_all_strategies(results)
    
    print("\n" + "="*80)
    print("验证完成!")
    print("="*80)

if __name__ == "__main__":
    main()
