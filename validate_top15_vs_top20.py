"""
验证精准TOP15 vs TOP20 的命中率和收益率对比
分析增加5个号码后的性能变化
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

def load_data():
    """加载数据"""
    df = pd.read_csv('data/lucky_numbers.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def test_topN_strategy(df, top_n=15, test_periods=200):
    """
    测试TOPN策略
    
    Args:
        df: 数据DataFrame
        top_n: 预测号码数量（15或20）
        test_periods: 测试期数
    """
    print("\n" + "="*80)
    print(f"策略: 精准TOP{top_n}投注")
    print("="*80)
    
    predictor = PreciseTop15Predictor()
    
    # 使用最近N期进行测试
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
        
        # 使用训练数据进行预测
        history_numbers = train_data['number'].tolist()
        
        # 获取预测结果
        if top_n == 15:
            predictions = predictor.predict(history_numbers)  # 返回15个
        else:  # top_n == 20
            # 需要获取20个预测
            pattern = predictor.analyze_pattern(history_numbers)
            if len(history_numbers) >= 50:
                pattern['recent_50'] = history_numbers[-50:]
            else:
                pattern['recent_50'] = history_numbers
            
            # 运行多个方法，获取更多候选
            base_k = 25  # 增加候选数量
            methods = [
                (predictor.method_precision_frequency(pattern, base_k), 0.40),
                (predictor.method_zone_dynamic(pattern, base_k), 0.25),
                (predictor.method_gap_analysis(pattern, base_k), 0.20),
                (predictor.method_avoid_recent_misses(pattern, base_k), 0.15)
            ]
            
            # 综合评分
            scores = {}
            for candidates, weight in methods:
                for rank, num in enumerate(candidates):
                    score = weight * (1.0 - rank / len(candidates))
                    scores[num] = scores.get(num, 0) + score
            
            # 返回TOP20
            final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            predictions = [num for num, _ in final[:20]]
        
        # 判断是否命中
        actual_number = test_row['number']
        hit = actual_number in predictions
        
        # 计算投注和奖励 (每个号码1元 * 倍投 * N个号码)
        multiplier = fib_sequence[min(multiplier_idx, len(fib_sequence)-1)]
        bet_amount = top_n * multiplier
        reward = 47 * multiplier if hit else 0
        profit = reward - bet_amount
        
        total_bet += bet_amount
        total_reward += reward
        
        # 更新预测器的性能追踪
        predictor.update_performance(predictions, actual_number)
        
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
            'predictions': predictions,
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
    print(f"总投入: {total_bet:,}元")
    print(f"总奖励: {total_reward:,}元")
    print(f"净收益: {total_profit:+,}元")
    print(f"投资回报率(ROI): {roi:+.2f}%")
    
    # 计算平均单期投入和回报
    avg_bet = total_bet / len(results)
    avg_reward = total_reward / len(results)
    avg_profit = total_profit / len(results)
    
    print(f"\n平均每期:")
    print(f"  投入: {avg_bet:.2f}元")
    print(f"  奖励: {avg_reward:.2f}元")
    print(f"  收益: {avg_profit:+.2f}元")
    
    return {
        'strategy': f'TOP{top_n}',
        'hit_count': hit_count,
        'total_periods': len(results),
        'hit_rate': hit_rate,
        'max_consecutive_misses': max_consecutive_misses,
        'total_bet': total_bet,
        'total_reward': total_reward,
        'profit': total_profit,
        'roi': roi,
        'avg_bet': avg_bet,
        'avg_reward': avg_reward,
        'avg_profit': avg_profit,
        'results': results
    }

def compare_strategies(result15, result20):
    """对比TOP15和TOP20策略"""
    print("\n" + "="*80)
    print("TOP15 vs TOP20 综合对比分析")
    print("="*80)
    
    # 核心指标对比表格
    print("\n📊 核心指标对比:")
    print("-" * 80)
    print(f"{'指标':<20} {'TOP15':<20} {'TOP20':<20} {'差异':<20}")
    print("-" * 80)
    
    # 命中率对比
    hit_rate_diff = result20['hit_rate'] - result15['hit_rate']
    hit_rate_diff_str = f"{hit_rate_diff:+.2f}%"
    print(f"{'命中率':<20} {result15['hit_rate']:<19.2f}% {result20['hit_rate']:<19.2f}% {hit_rate_diff_str:<20}")
    
    # 命中次数对比
    hit_count_diff = result20['hit_count'] - result15['hit_count']
    hit_count_diff_str = f"{hit_count_diff:+d}次"
    print(f"{'命中次数':<20} {result15['hit_count']:<19d}次 {result20['hit_count']:<19d}次 {hit_count_diff_str:<20}")
    
    # ROI对比
    roi_diff = result20['roi'] - result15['roi']
    roi_diff_str = f"{roi_diff:+.2f}%"
    print(f"{'ROI':<20} {result15['roi']:<19.2f}% {result20['roi']:<19.2f}% {roi_diff_str:<20}")
    
    # 净收益对比
    profit_diff = result20['profit'] - result15['profit']
    profit_diff_str = f"{profit_diff:+,}元"
    print(f"{'净收益':<20} {result15['profit']:<19,}元 {result20['profit']:<19,}元 {profit_diff_str:<20}")
    
    # 总投入对比
    bet_diff = result20['total_bet'] - result15['total_bet']
    bet_diff_str = f"{bet_diff:+,}元"
    print(f"{'总投入':<20} {result15['total_bet']:<19,}元 {result20['total_bet']:<19,}元 {bet_diff_str:<20}")
    
    # 最大连不中对比
    miss_diff = result20['max_consecutive_misses'] - result15['max_consecutive_misses']
    miss_diff_str = f"{miss_diff:+d}期"
    print(f"{'最大连不中':<20} {result15['max_consecutive_misses']:<19d}期 {result20['max_consecutive_misses']:<19d}期 {miss_diff_str:<20}")
    
    print("-" * 80)
    
    # 详细分析
    print("\n" + "="*80)
    print("详细分析")
    print("="*80)
    
    # 命中率提升分析
    print(f"\n✅ 命中率分析:")
    if hit_rate_diff > 0:
        improvement = (result20['hit_rate'] / result15['hit_rate'] - 1) * 100
        print(f"  TOP20命中率提升 {hit_rate_diff:.2f}个百分点")
        print(f"  相对提升幅度: {improvement:.2f}%")
        print(f"  新增命中次数: {hit_count_diff}次")
    else:
        print(f"  ⚠️ TOP20命中率未提升（{hit_rate_diff:.2f}%）")
    
    # ROI分析
    print(f"\n💰 投资回报率分析:")
    if roi_diff > 0:
        print(f"  ✅ TOP20 ROI更优，提升 {roi_diff:.2f}个百分点")
        print(f"  净收益增加: {profit_diff:+,}元")
    elif roi_diff < 0:
        print(f"  ⚠️ TOP20 ROI下降 {roi_diff:.2f}个百分点")
        print(f"  净收益减少: {profit_diff:,}元")
    else:
        print(f"  ROI持平")
    
    # 成本效益分析
    print(f"\n📊 成本效益分析:")
    print(f"  TOP15 单期平均投入: {result15['avg_bet']:.2f}元")
    print(f"  TOP20 单期平均投入: {result20['avg_bet']:.2f}元")
    print(f"  投入增加: {result20['avg_bet'] - result15['avg_bet']:+.2f}元 ({(result20['avg_bet']/result15['avg_bet']-1)*100:+.2f}%)")
    print()
    print(f"  TOP15 单期平均收益: {result15['avg_profit']:+.2f}元")
    print(f"  TOP20 单期平均收益: {result20['avg_profit']:+.2f}元")
    print(f"  收益增加: {result20['avg_profit'] - result15['avg_profit']:+.2f}元")
    
    # 风险控制分析
    print(f"\n⚠️ 风险控制分析:")
    if miss_diff < 0:
        print(f"  ✅ TOP20风险更低，最大连不中减少{abs(miss_diff)}期")
    elif miss_diff > 0:
        print(f"  ⚠️ TOP20风险更高，最大连不中增加{miss_diff}期")
    else:
        print(f"  风险相同，最大连不中均为{result15['max_consecutive_misses']}期")
    
    # 综合评分
    print("\n" + "="*80)
    print("综合评分 (命中率40% + ROI35% + 收益25%)")
    print("="*80)
    
    # 计算评分
    max_hit_rate = max(result15['hit_rate'], result20['hit_rate'])
    max_roi = max(result15['roi'], result20['roi'])
    max_profit = max(result15['profit'], result20['profit'])
    
    def calculate_score(result):
        hit_score = (result['hit_rate'] / max_hit_rate) * 40 if max_hit_rate > 0 else 0
        roi_score = (result['roi'] / max_roi) * 35 if max_roi > 0 else 0
        profit_score = (result['profit'] / max_profit) * 25 if max_profit > 0 else 0
        return hit_score + roi_score + profit_score
    
    score15 = calculate_score(result15)
    score20 = calculate_score(result20)
    
    print(f"\nTOP15 综合评分: {score15:.1f}分")
    print(f"TOP20 综合评分: {score20:.1f}分")
    print(f"评分差异: {score20 - score15:+.1f}分")
    
    # 最终推荐
    print("\n" + "="*80)
    print("💡 最终推荐")
    print("="*80)
    
    if score20 > score15:
        winner = "TOP20"
        winner_result = result20
        print(f"\n🏆 推荐: {winner}")
        print(f"\n推荐理由:")
        if hit_rate_diff > 0:
            print(f"  ✅ 命中率提升 {hit_rate_diff:.2f}个百分点")
        if roi_diff > 0:
            print(f"  ✅ ROI提升 {roi_diff:.2f}个百分点")
        if profit_diff > 0:
            print(f"  ✅ 净收益增加 {profit_diff:+,}元")
    else:
        winner = "TOP15"
        winner_result = result15
        print(f"\n🏆 推荐: {winner}")
        print(f"\n推荐理由:")
        if hit_rate_diff < 0:
            print(f"  ⚠️ TOP20命中率未显著提升（仅{hit_rate_diff:+.2f}%）")
        if roi_diff < 0:
            print(f"  ⚠️ TOP20 ROI下降 {abs(roi_diff):.2f}个百分点")
        if profit_diff < 0:
            print(f"  ⚠️ TOP20净收益减少 {abs(profit_diff):,}元")
        print(f"  ✅ TOP15性价比更优，投入更少，回报更高")
    
    print(f"\n核心指标:")
    print(f"  命中率: {winner_result['hit_rate']:.2f}%")
    print(f"  ROI: {winner_result['roi']:+.2f}%")
    print(f"  净收益: {winner_result['profit']:+,}元")
    print(f"  综合评分: {score20 if winner == 'TOP20' else score15:.1f}分")
    
    # 使用建议
    print(f"\n💡 使用建议:")
    if winner == "TOP20":
        print(f"  • 适合追求高命中率的投资者")
        print(f"  • 覆盖面更广，降低遗漏风险")
        print(f"  • 需要接受单期投入增加约{(result20['avg_bet']/result15['avg_bet']-1)*100:.1f}%")
    else:
        print(f"  • 适合追求投资效率的投资者")
        print(f"  • 精准选择，降低无效投入")
        print(f"  • ROI最优，资金利用率更高")
    
    # 特殊场景建议
    print(f"\n📌 特殊场景:")
    print(f"  • 资金充足 → 可选择{winner}")
    print(f"  • 追求极致ROI → 选择TOP15")
    print(f"  • 追求高命中率 → 选择TOP20")
    print(f"  • 风险厌恶型 → 选择{winner}")

def main():
    """主函数"""
    print("\n" + "="*80)
    print("精准TOP15 vs TOP20 命中率和收益率对比验证")
    print("="*80)
    print("验证期数: 最近200期")
    print("倍投策略: Fibonacci数列 (1,1,2,3,5,8,13,21,34,55,89,144)")
    print("投注规则: TOP15每期15元×倍投, TOP20每期20元×倍投")
    print("奖励规则: 中奖获得47元×倍投")
    print("="*80)
    
    # 加载数据
    df = load_data()
    print(f"\n数据加载完成: 共{len(df)}期数据")
    
    # 测试TOP15
    result15 = test_topN_strategy(df, top_n=15, test_periods=200)
    
    # 测试TOP20
    result20 = test_topN_strategy(df, top_n=20, test_periods=200)
    
    # 综合对比
    compare_strategies(result15, result20)
    
    print("\n" + "="*80)
    print("验证完成!")
    print("="*80)

if __name__ == "__main__":
    main()
