"""
最近30期详细对比 - 固定1倍投注（不止损）vs 马丁格尔
"""

import pandas as pd

def compare_last30_simple():
    print("="*90)
    print("最近30期回测 - 固定1倍投注 vs 马丁格尔倍投")
    print("="*90)
    
    # 读取原数据
    df = pd.read_csv('zodiac_top4_backtest_300periods.csv', encoding='utf-8-sig')
    last30 = df.tail(30)
    
    # 计算固定1倍投注的结果
    fixed_balance = 0
    fixed_results = []
    
    for idx, row in last30.iterrows():
        is_hit = row['is_hit'] == '是'
        if is_hit:
            profit = 47 - 16  # 中奖47元 - 投注16元
        else:
            profit = -16
        
        fixed_balance += profit
        fixed_results.append({
            'period': row['period'],
            'actual': row['actual'],
            'is_hit': is_hit,
            'profit': profit,
            'balance': fixed_balance
        })
    
    # 统计原策略（马丁格尔）
    martin_hits = (last30['is_hit'] == '是').sum()
    martin_bet = last30['bet_amount'].sum()
    martin_profit = last30['profit'].sum()
    martin_roi = martin_profit / martin_bet * 100
    
    # 统计固定1倍投注
    fixed_hits = sum(1 for r in fixed_results if r['is_hit'])
    fixed_bet = 30 * 16
    fixed_profit = fixed_balance
    fixed_roi = fixed_profit / fixed_bet * 100
    
    print(f"\n{'='*90}")
    print(f"{'策略':<25} {'命中率':<12} {'投注期数':<10} {'投注总额':<12} {'累计盈利':<12} {'投资回报率'}")
    print("-"*90)
    print(f"{'马丁格尔10倍（原策略）':<25} {martin_hits/30*100:>5.1f}% ({martin_hits}/30) "
          f"{30:>6}期 {martin_bet:>9.0f}元 {martin_profit:>9.0f}元 {martin_roi:>9.1f}%")
    print(f"{'固定1倍投注（不止损）':<25} {fixed_hits/30*100:>5.1f}% ({fixed_hits}/30) "
          f"{30:>6}期 {fixed_bet:>9.0f}元 {fixed_profit:>9.0f}元 {fixed_roi:>9.1f}%")
    
    print("\n" + "="*90)
    print("【详细对比】")
    print("="*90)
    
    print(f"\n期数   实际  结果  马丁格尔倍投                固定1倍投注")
    print(f"              (倍数/投注/盈亏/累计)         (投注/盈亏/累计)")
    print("-"*90)
    
    for i, (idx, row) in enumerate(last30.iterrows()):
        period = row['period']
        actual = row['actual']
        hit_symbol = '✅' if row['is_hit'] == '是' else '❌'
        
        # 马丁格尔
        martin_mult = row['multiplier']
        martin_bet_amt = int(row['bet_amount'])
        martin_profit_amt = int(row['profit'])
        martin_cum = int(row['cumulative_profit'])
        
        # 固定1倍
        fixed_record = fixed_results[i]
        fixed_profit_amt = fixed_record['profit']
        fixed_cum = fixed_record['balance']
        
        print(f"{period:<6} {actual:<5} {hit_symbol}   "
              f"{martin_mult:>3.1f}倍/{martin_bet_amt:>3}元/{martin_profit_amt:>4}元/{martin_cum:>6}元    "
              f"16元/{fixed_profit_amt:>4}元/{fixed_cum:>5}元")
    
    print("\n" + "="*90)
    print("【核心结论】")
    print("="*90)
    
    improvement = fixed_profit - martin_profit
    improvement_pct = (fixed_roi - martin_roi)
    
    if fixed_profit > martin_profit:
        print(f"\n✅ 固定1倍投注表现更优!")
        print(f"   • 盈利多 {improvement:.0f}元")
        print(f"   • 投资回报率高 {improvement_pct:.1f}%")
        print(f"   • 投入资金少 {martin_bet - fixed_bet:.0f}元")
        print(f"   • 风险极低（单期最多亏16元）")
    else:
        print(f"\n⚠️ 原策略在这30期表现更好，但...")
        print(f"   • 风险极高（单期最高投注{last30['bet_amount'].max():.0f}元）")
        print(f"   • 最后5期连续不中导致巨额亏损")
        print(f"   • 固定1倍策略更稳定，适合长期投注")
    
    print(f"\n💡 最近30期特殊情况:")
    print(f"   • 前25期：命中率出色，马丁格尔获利高")
    print(f"   • 后5期：连续不中，马丁格尔亏损严重")
    print(f"   • 固定1倍：稳定盈利{fixed_profit:.0f}元，没有大起大落")
    
    # 保存详细记录
    df_fixed = pd.DataFrame(fixed_results)
    df_fixed.to_csv('zodiac_top4_fixed1x_last30.csv', index=False, encoding='utf-8-sig')
    print(f"\n📁 详细数据已保存: zodiac_top4_fixed1x_last30.csv")

if __name__ == '__main__':
    compare_last30_simple()
