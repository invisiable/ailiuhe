"""
重新验证最近30期数据
检查CSV中的is_hit字段是否准确
"""

import pandas as pd

def verify_last30():
    df = pd.read_csv('zodiac_top4_backtest_300periods.csv', encoding='utf-8-sig')
    last30 = df.tail(30)
    
    print("="*90)
    print("最近30期命中情况详细验证")
    print("="*90)
    
    manual_hits = 0
    csv_hits = 0
    mismatches = []
    
    print(f"\n{'期数':<6} {'实际':<6} {'预测TOP4':<35} CSV记录  手动验证  结果")
    print("-"*90)
    
    for idx, row in last30.iterrows():
        period = int(row['period'])
        actual = str(row['actual']).strip()
        top4_str = str(row['top4_zodiacs'])
        top4_list = [z.strip() for z in top4_str.split(',')]
        csv_hit = row['is_hit'] == '是'
        
        # 手动验证
        manual_hit = actual in top4_list
        
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
        
        top4_short = ', '.join(top4_list[:2]) + '...'
        print(f"{period:<6} {actual:<6} {top4_short:<35} {csv_symbol:<8} {manual_symbol:<10} {result}")
    
    print("\n" + "="*90)
    print("【验证结果】")
    print("="*90)
    print(f"CSV记录命中数: {csv_hits}/30 ({csv_hits/30*100:.1f}%)")
    print(f"手动验证命中数: {manual_hits}/30 ({manual_hits/30*100:.1f}%)")
    
    if mismatches:
        print(f"\n⚠️ 发现{len(mismatches)}处不一致:")
        for p in mismatches:
            print(f"   期数 {p}")
    else:
        print(f"\n✅ CSV数据与手动验证完全一致")
    
    # 列出所有命中的期数
    print(f"\n【命中期数列表】")
    hit_periods = []
    for idx, row in last30.iterrows():
        actual = str(row['actual']).strip()
        top4_list = [z.strip() for z in str(row['top4_zodiacs']).split(',')]
        if actual in top4_list:
            hit_periods.append(int(row['period']))
    
    print(f"命中期数: {', '.join(map(str, hit_periods))}")
    print(f"总计: {len(hit_periods)}期")

if __name__ == '__main__':
    verify_last30()
