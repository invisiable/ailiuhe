"""
测试生肖投注策略：按排名顺序使用不同倍数 vs 统一倍数
"""

import pandas as pd
from ensemble_zodiac_predictor import EnsembleZodiacPredictor

def test_weighted_betting_strategies(data_file='data/lucky_numbers.csv', test_periods=100):
    """测试多种倍数分配策略"""
    
    print("="*80)
    print("生肖投注倍数分配策略对比测试")
    print("="*80)
    print(f"测试期数: 最近{test_periods}期\n")
    
    # 读取数据
    df = pd.read_csv(data_file, encoding='utf-8-sig')
    print(f"数据加载: {len(df)}期历史数据")
    print(f"最新期: {df.iloc[-1]['date']} - {df.iloc[-1]['animal']}\n")
    
    # 创建预测器
    predictor = EnsembleZodiacPredictor()
    
    # 测试数据范围
    start_idx = len(df) - test_periods
    
    # 生成预测数据
    predictions = []
    actuals = []
    hit_positions = []  # 记录命中的是第几个预测
    
    print("生成历史预测数据...\n")
    for i in range(start_idx, len(df)):
        train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
        result = predictor.predict_from_history(train_animals, top_n=5, debug=False)
        top4 = result['top4']
        predictions.append(top4)
        
        actual = str(df.iloc[i]['animal']).strip()
        actuals.append(actual)
        
        # 记录命中位置（0-3表示TOP1-TOP4，-1表示未命中）
        if actual in top4:
            hit_positions.append(top4.index(actual))
        else:
            hit_positions.append(-1)
        
        if (i - start_idx + 1) % 20 == 0:
            print(f"  已处理 {i - start_idx + 1}/{test_periods} 期...")
    
    print(f"\n✅ 预测数据生成完成！\n")
    
    # 定义多种倍数分配策略
    strategies = {
        '统一倍数': {
            'multipliers': [1, 1, 1, 1],  # TOP1-TOP4都是1倍
            'description': '每个生肖4元，总投入16元'
        },
        '递减倍数': {
            'multipliers': [2.5, 2, 1.5, 1],  # 总和=7，归一化后保持总投入16元
            'description': 'TOP1重点投入，逐级递减'
        },
        '集中TOP1': {
            'multipliers': [4, 1, 1, 1],  # 总和=7
            'description': '重点集中在TOP1'
        },
        '前两重点': {
            'multipliers': [3, 3, 1, 1],  # 总和=8
            'description': 'TOP1和TOP2各占50%资金'
        },
        '平方递减': {
            'multipliers': [4, 3, 2, 1],  # 总和=10
            'description': '按平方级递减'
        },
        '极端集中': {
            'multipliers': [6, 2, 1, 1],  # 总和=10
            'description': 'TOP1占60%资金'
        },
        '金字塔型': {
            'multipliers': [5, 3, 2, 1],  # 总和=11
            'description': '金字塔式分配'
        }
    }
    
    base_total_bet = 16  # 固定总投入16元
    win_amount = 45  # 命中奖励45元
    
    results = {}
    
    print("="*80)
    print("测试各种倍数分配策略")
    print("="*80)
    print()
    
    for strategy_name, strategy_config in strategies.items():
        multipliers = strategy_config['multipliers']
        description = strategy_config['description']
        
        # 归一化倍数，使总投入始终为16元
        total_multiplier = sum(multipliers)
        normalized_multipliers = [m * base_total_bet / total_multiplier for m in multipliers]
        
        # 计算每期收益
        total_profit = 0
        total_investment = 0
        wins = 0
        
        period_details = []
        
        for i, hit_pos in enumerate(hit_positions):
            # 当期投注金额
            period_bet = sum(normalized_multipliers)
            total_investment += period_bet
            
            if hit_pos >= 0:  # 命中了
                # 获得奖励（命中的那个生肖的投注金额 * 奖励倍数）
                hit_bet = normalized_multipliers[hit_pos]
                period_profit = win_amount - period_bet  # 奖励减去总投入
                total_profit += period_profit
                wins += 1
                period_details.append({
                    'profit': period_profit,
                    'hit_pos': hit_pos,
                    'hit_bet': hit_bet
                })
            else:  # 未命中
                period_profit = -period_bet
                total_profit += period_profit
                period_details.append({
                    'profit': period_profit,
                    'hit_pos': -1,
                    'hit_bet': 0
                })
        
        roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
        win_rate = (wins / len(hit_positions) * 100) if len(hit_positions) > 0 else 0
        
        results[strategy_name] = {
            'multipliers': normalized_multipliers,
            'total_profit': total_profit,
            'total_investment': total_investment,
            'roi': roi,
            'wins': wins,
            'win_rate': win_rate,
            'description': description,
            'period_details': period_details
        }
        
        print(f"【{strategy_name}】{description}")
        print(f"  倍数分配: TOP1={normalized_multipliers[0]:.2f}元, TOP2={normalized_multipliers[1]:.2f}元, "
              f"TOP3={normalized_multipliers[2]:.2f}元, TOP4={normalized_multipliers[3]:.2f}元")
        print(f"  总投入: {total_investment:.2f}元")
        print(f"  总收益: {total_profit:+.2f}元")
        print(f"  ROI: {roi:+.2f}%")
        print(f"  命中率: {win_rate:.2f}% ({wins}/{test_periods})")
        print()
    
    # 排序并显示对比
    print("="*80)
    print("策略对比排名（按ROI排序）")
    print("="*80)
    print()
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['roi'], reverse=True)
    
    print(f"{'排名':<4} {'策略':<12} {'ROI':<12} {'总收益':<12} {'命中率':<10} {'说明':<30}")
    print("-"*80)
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        marker = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        print(f"{marker:<4} {name:<12} {data['roi']:>+10.2f}% {data['total_profit']:>+10.2f}元 "
              f"{data['win_rate']:>8.2f}% {data['description']:<30}")
    
    print()
    
    # 统计命中位置分布
    print("="*80)
    print("命中位置分析")
    print("="*80)
    print()
    
    hit_count = [0, 0, 0, 0]  # TOP1-TOP4命中次数
    miss_count = 0
    
    for pos in hit_positions:
        if pos >= 0:
            hit_count[pos] += 1
        else:
            miss_count += 1
    
    total_hits = sum(hit_count)
    
    print(f"总命中次数: {total_hits}/{test_periods} = {total_hits/test_periods*100:.2f}%")
    print(f"TOP1 命中: {hit_count[0]}次 ({hit_count[0]/test_periods*100:.2f}%) - {'占总命中' if total_hits > 0 else ''} {hit_count[0]/total_hits*100:.1f}%")
    print(f"TOP2 命中: {hit_count[1]}次 ({hit_count[1]/test_periods*100:.2f}%) - {'占总命中' if total_hits > 0 else ''} {hit_count[1]/total_hits*100:.1f}%")
    print(f"TOP3 命中: {hit_count[2]}次 ({hit_count[2]/test_periods*100:.2f}%) - {'占总命中' if total_hits > 0 else ''} {hit_count[2]/total_hits*100:.1f}%")
    print(f"TOP4 命中: {hit_count[3]}次 ({hit_count[3]/test_periods*100:.2f}%) - {'占总命中' if total_hits > 0 else ''} {hit_count[3]/total_hits*100:.1f}%")
    print(f"未命中: {miss_count}次 ({miss_count/test_periods*100:.2f}%)")
    print()
    
    # 最优策略建议
    print("="*80)
    print("最优策略建议")
    print("="*80)
    print()
    
    best_strategy = sorted_results[0]
    best_name = best_strategy[0]
    best_data = best_strategy[1]
    
    baseline = results['统一倍数']
    
    print(f"🏆 最优策略: {best_name}")
    print(f"   ROI: {best_data['roi']:+.2f}%")
    print(f"   总收益: {best_data['total_profit']:+.2f}元")
    print(f"   相比统一倍数策略:")
    print(f"   - ROI差异: {best_data['roi'] - baseline['roi']:+.2f}%")
    print(f"   - 收益差异: {best_data['total_profit'] - baseline['total_profit']:+.2f}元")
    print()
    print(f"   倍数配置:")
    for i, mult in enumerate(best_data['multipliers'], 1):
        print(f"   - TOP{i}: {mult:.2f}元")
    print()
    
    # 详细收益曲线对比（最近20期）
    print("="*80)
    print("最近20期详细对比（最优策略 vs 统一倍数）")
    print("="*80)
    print()
    
    print(f"{'期数':<8} {'日期':<12} {'实际':<6} {'命中位置':<10} "
          f"{'最优收益':<12} {'统一收益':<12} {'差异':<10}")
    print("-"*80)
    
    for i in range(max(0, test_periods-20), test_periods):
        idx = start_idx + i
        date_str = df.iloc[idx]['date']
        actual = actuals[i]
        hit_pos = hit_positions[i]
        
        if hit_pos >= 0:
            pos_str = f"TOP{hit_pos+1}"
        else:
            pos_str = "未中"
        
        best_profit = best_data['period_details'][i]['profit']
        baseline_profit = baseline['period_details'][i]['profit']
        diff = best_profit - baseline_profit
        
        print(f"第{idx+1:<5}期 {date_str:<12} {actual:<6} {pos_str:<10} "
              f"{best_profit:>+10.2f}元 {baseline_profit:>+10.2f}元 {diff:>+8.2f}元")
    
    print("-"*80)
    print()
    
    return results

if __name__ == '__main__':
    results = test_weighted_betting_strategies(test_periods=100)
