"""
深度分析最近10期未命中的真正原因
"""

import pandas as pd
from collections import Counter

def deep_analysis_recent_10():
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    print('='*100)
    print('最近10期深度分析')
    print('='*100)
    print()
    
    # 分析最近10期的实际开奖生肖
    recent_10_actual = [str(df['animal'].values[i]).strip() for i in range(total-10, total)]
    print(f"最近10期实际开奖: {', '.join(recent_10_actual)}")
    print()
    
    # 统计频率
    counter = Counter(recent_10_actual)
    print("最近10期生肖频率:")
    for zodiac, count in counter.most_common():
        print(f"  {zodiac}: {count}次 ({count/10*100:.0f}%)")
    print()
    
    # 分析每个未命中生肖在历史上的表现
    print('='*100)
    print('未命中生肖的历史特征分析')
    print('='*100)
    print()
    
    unmatch_zodiacs = {'虎': 2, '蛇': 3, '鼠': 1, '狗': 1, '马': 1}
    
    for zodiac, unmatch_count in unmatch_zodiacs.items():
        print(f"\n{zodiac} (最近10期未命中{unmatch_count}次):")
        print('-'*100)
        
        # 最近50期统计
        recent_50 = [str(df['animal'].values[i]).strip() for i in range(total-50, total)]
        count_50 = recent_50.count(zodiac)
        print(f"  最近50期出现: {count_50}次 (理论4.17次, {'+超出' if count_50 > 4.17 else '-低于'}{abs(count_50-4.17):.2f}次)")
        
        # 最近20期统计
        recent_20 = [str(df['animal'].values[i]).strip() for i in range(total-20, total)]
        count_20 = recent_20.count(zodiac)
        print(f"  最近20期出现: {count_20}次 (理论1.67次, {'+超出' if count_20 > 1.67 else '-低于'}{abs(count_20-1.67):.2f}次)")
        
        # 最近10期统计
        count_10 = recent_10_actual.count(zodiac)
        print(f"  最近10期出现: {count_10}次 (理论0.83次, {'+超出' if count_10 > 0.83 else '-低于'}{abs(count_10-0.83):.2f}次)")
        
        # 判断冷热
        if count_10 >= 2:
            status = "热门 (近期频繁出现)"
        elif count_20 >= 3:
            status = "偏热 (中期较多出现)"
        elif count_50 <= 2:
            status = "冷门 (长期很少出现)"
        else:
            status = "正常"
        
        print(f"  状态判断: {status}")
        
        # 为什么冷门策略会排除它
        if count_50 > 4.17:
            print(f"  [!] 冷门策略会严重扣分（最近50期出现{count_50}次，超出理论值）")
        if count_10 >= 2:
            print(f"  [!] 反热门策略会扣分（最近10期出现{count_10}次）")
    
    print()
    print('='*100)
    print('关键发现')
    print('='*100)
    print()
    
    # 统计"热门"生肖在最近10期的表现
    hot_zodiacs = [z for z, c in counter.items() if c >= 2]
    print(f"最近10期的'热门'生肖: {', '.join([f'{z}({counter[z]}次)' for z in hot_zodiacs])}")
    print(f"这些热门生肖占最近10期开奖的: {sum(counter[z] for z in hot_zodiacs)}/10 = "
          f"{sum(counter[z] for z in hot_zodiacs)/10*100:.0f}%")
    print()
    
    # 分析是否存在"连续模式"
    print("连续性分析:")
    consecutive_same = 0
    for i in range(1, len(recent_10_actual)):
        if recent_10_actual[i] == recent_10_actual[i-1]:
            consecutive_same += 1
            print(f"  第{total-10+i}期和第{total-10+i+1}期: 连续出现 {recent_10_actual[i]}")
    
    if consecutive_same == 0:
        print("  [!] 没有任何连续出现的情况")
    print()
    
    # 分析间隔
    print("间隔模式分析:")
    for zodiac in set(recent_10_actual):
        indices = [i for i, z in enumerate(recent_10_actual) if z == zodiac]
        if len(indices) >= 2:
            gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            print(f"  {zodiac}: 出现在第{[total-10+i+1 for i in indices]}期, 间隔={gaps}")
    print()
    
    print('='*100)
    print('结论')
    print('='*100)
    print()
    print("1. 最近10期虎(2次)、蛇(3次)、狗(2次)确实是'热门'，占比70%")
    print("2. 但这些热门生肖在最近50期也是相对活跃的，不是突然变热")
    print("3. 冷门策略认为近期活跃的生肖不会再出现，但实际它们持续出现")
    print("4. 问题不是'周期切换'，而是'持续性'没有被正确建模")
    print()
    print("建议:")
    print("  - 不是检测'热门周期'然后切换权重")
    print("  - 而是增加'动量/惯性'因子：最近频繁出现的，可能继续出现")
    print("  - 保持冷门策略，但添加'热门惯性'作为补充策略")

if __name__ == '__main__':
    deep_analysis_recent_10()
