"""
正确验证最近30期 - 基于号码匹配
"""

import pandas as pd

def verify_last30_correct():
    df = pd.read_csv('zodiac_top4_backtest_300periods.csv', encoding='utf-8-sig')
    last30 = df.tail(30)
    
    # 同时加载源数据获取生肖信息
    df_source = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    number_to_zodiac = {}
    for _, row in df_source.iterrows():
        number_to_zodiac[int(row['number'])] = str(row['animal']).strip()
    
    print("="*100)
    print("最近30期命中情况正确验证（号码匹配）")
    print("="*100)
    
    manual_hits = 0
    csv_hits = 0
    mismatches = []
    
    print(f"\n{'期数':<6} {'实际号码':<8} {'生肖':<6} {'投注号码(部分)':<25} CSV  手动  结果")
    print("-"*100)
    
    for idx, row in last30.iterrows():
        period = int(row['period'])
        actual_number = int(row['actual'])
        actual_zodiac = number_to_zodiac.get(actual_number, '?')
        
        bet_numbers_str = str(row['bet_numbers'])
        bet_numbers = [int(n.strip()) for n in bet_numbers_str.split(',')]
        
        csv_hit = row['is_hit'] == '是'
        manual_hit = actual_number in bet_numbers
        
        if csv_hit:
            csv_hits += 1
        if manual_hit:
            manual_hits += 1
        
        csv_symbol = '✅' if csv_hit else '❌'
        manual_symbol = '✅' if manual_hit else '❌'
        
        if csv_hit != manual_hit:
            mismatches.append(period)
            result = '⚠️不符'
        else:
            result = '一致'
        
        bet_short = ', '.join(map(str, bet_numbers[:6])) + '...'
        print(f"{period:<6} {actual_number:<8} {actual_zodiac:<6} {bet_short:<25} {csv_symbol:<4} {manual_symbol:<5} {result}")
    
    print("\n" + "="*100)
    print("【验证结果】")
    print("="*100)
    print(f"CSV记录命中数: {csv_hits}/30 ({csv_hits/30*100:.1f}%)")
    print(f"手动验证命中数: {manual_hits}/30 ({manual_hits/30*100:.1f}%)")
    
    if mismatches:
        print(f"\n⚠️ 发现{len(mismatches)}处不一致，数据有误！")
        for p in mismatches:
            print(f"   期数 {p}")
    else:
        print(f"\n✅ CSV数据与手动验证完全一致，数据正确！")
    
    # 统计不同生肖的命中情况
    print(f"\n【生肖命中统计】")
    zodiac_hit_count = {}
    zodiac_total_count = {}
    
    for idx, row in last30.iterrows():
        actual_number = int(row['actual'])
        actual_zodiac = number_to_zodiac.get(actual_number, '?')
        
        bet_numbers_str = str(row['bet_numbers'])
        bet_numbers = [int(n.strip()) for n in bet_numbers_str.split(',')]
        
        zodiac_total_count[actual_zodiac] = zodiac_total_count.get(actual_zodiac, 0) + 1
        
        if actual_number in bet_numbers:
            zodiac_hit_count[actual_zodiac] = zodiac_hit_count.get(actual_zodiac, 0) + 1
    
    for zodiac in sorted(zodiac_total_count.keys()):
        hits = zodiac_hit_count.get(zodiac, 0)
        total = zodiac_total_count[zodiac]
        rate = hits/total*100 if total > 0 else 0
        print(f"  {zodiac}: {hits}/{total} ({rate:.0f}%)")
    
    return csv_hits, manual_hits

if __name__ == '__main__':
    verify_last30_correct()
