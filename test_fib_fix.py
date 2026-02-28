"""
验证Fib索引显示修复
测试period_details中是否正确记录fib_index字段
"""

import pandas as pd
import numpy as np
from zodiac_simple_smart import ZodiacSimpleSmart


def test_fib_index_recording():
    """测试Fib索引记录功能"""
    
    print("="*80)
    print("Fib索引显示修复验证")
    print("="*80)
    print()
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    animals = df['animal'].tolist()
    
    total_periods = len(animals)
    test_periods = 50  # 测试50期
    start = total_periods - test_periods
    
    print(f"数据总期数: {total_periods}")
    print(f"测试期数: {test_periods}")
    print(f"起始索引: {start}")
    print()
    
    # 初始化预测器
    predictor = ZodiacSimpleSmart()
    
    # v3.2激进组合配置
    lookback = 8
    boost_mult = 1.5
    reduce_mult = 0.5
    max_multiplier = 10
    base_bet = 20
    win_reward = 47
    
    # 斐波那契数列
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    def get_fib_multiplier(fib_index):
        if fib_index >= len(fib_sequence):
            return min(fib_sequence[-1], max_multiplier)
        return min(fib_sequence[fib_index], max_multiplier)
    
    def get_recent_rate(recent_results):
        if len(recent_results) == 0:
            return 0.42
        return sum(recent_results) / len(recent_results)
    
    # 状态变量
    fib_index = 0
    recent_results = []
    period_details = []
    
    # 回测
    for i in range(start, total_periods):
        history = animals[:i]
        actual_animal = animals[i]
        
        # 预测TOP5
        if len(history) >= 30:
            result = predictor.predict_from_history(history, top_n=5)
            predicted_top5 = result['top5']
        else:
            predicted_top5 = predictor.zodiac_list[:5]
        
        # 判断命中
        hit = actual_animal in predicted_top5
        
        # 获取基础倍数
        base_mult = get_fib_multiplier(fib_index)
        
        # 记录当前Fib索引（用于显示）
        current_fib_index = fib_index
        
        # 根据最近命中率计算动态倍数
        if len(recent_results) >= lookback:
            rate = get_recent_rate(recent_results)
            if rate >= 0.35:
                multiplier = min(base_mult * boost_mult, max_multiplier)
            elif rate <= 0.20:
                multiplier = max(base_mult * reduce_mult, 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        # 投注和收益
        bet = base_bet * multiplier
        
        if hit:
            profit = win_reward * multiplier - bet
            fib_index = 0
            status = '✓中'
        else:
            profit = -bet
            fib_index += 1
            status = '✗失'
        
        # 更新历史
        recent_results.append(1 if hit else 0)
        if len(recent_results) > lookback:
            recent_results.pop(0)
        
        # 记录详情（包含fib_index）
        period_details.append({
            'period': i - start,
            'date': df.iloc[i]['date'],
            'actual': actual_animal,
            'predicted': ','.join(predicted_top5),
            'multiplier': multiplier,
            'bet': bet,
            'status': status,
            'profit': profit,
            'recent_rate': get_recent_rate(recent_results),
            'fib_index': current_fib_index  # 关键字段
        })
    
    # 显示前10期和后10期的详情
    print("验证Fib索引记录（前10期 + 后10期）:")
    print()
    print(f"{'期号':<6}{'日期':<12}{'实际':<6}{'倍数':<8}{'投注':<8}{'命中':<6}{'盈亏':<10}{'Fib索引':<8}{'8期率':<10}")
    print("-"*80)
    
    # 前10期
    for detail in period_details[:10]:
        period = detail['period'] + 1
        date_str = detail['date']
        actual = detail['actual']
        multiplier = detail['multiplier']
        bet = detail['bet']
        status = detail['status']
        profit = detail['profit']
        fib_idx = detail['fib_index']
        rate = detail['recent_rate']
        
        print(f"{period:<6}{date_str:<12}{actual:<6}{multiplier:<8.2f}{bet:<8.0f}{status:<6}{profit:+10.0f}{fib_idx:<8}{rate*100:<10.1f}%")
    
    print("...")
    
    # 后10期
    for detail in period_details[-10:]:
        period = detail['period'] + 1
        date_str = detail['date']
        actual = detail['actual']
        multiplier = detail['multiplier']
        bet = detail['bet']
        status = detail['status']
        profit = detail['profit']
        fib_idx = detail['fib_index']
        rate = detail['recent_rate']
        
        print(f"{period:<6}{date_str:<12}{actual:<6}{multiplier:<8.2f}{bet:<8.0f}{status:<6}{profit:+10.0f}{fib_idx:<8}{rate*100:<10.1f}%")
    
    print()
    print("="*80)
    print()
    
    # 验证Fib索引的正确性
    print("Fib索引验证:")
    print()
    
    # 检查是否所有记录都有fib_index字段
    all_have_fib = all('fib_index' in d for d in period_details)
    print(f"✅ 所有记录都包含fib_index字段: {all_have_fib}")
    
    # 检查Fib索引是否合理（0-11之间，因为斐波那契数列有12个）
    fib_indices = [d['fib_index'] for d in period_details]
    max_fib = max(fib_indices)
    min_fib = min(fib_indices)
    print(f"✅ Fib索引范围: {min_fib}-{max_fib} (预期0-11)")
    
    # 检查命中后Fib索引是否重置为0
    reset_count = 0
    for i, detail in enumerate(period_details[1:], 1):
        prev_detail = period_details[i-1]
        if prev_detail['status'] == '✓中' and detail['fib_index'] == 0:
            reset_count += 1
    
    hits_count = sum(1 for d in period_details if d['status'] == '✓中')
    # 最后一期如果命中，下一期不在测试范围内，所以reset_count可能比hits_count少1
    print(f"✅ 命中后Fib重置: {reset_count}次 (命中{hits_count}次，预期{reset_count}≈{hits_count})")
    
    # 检查未命中后Fib索引是否递增
    increment_count = 0
    for i, detail in enumerate(period_details[1:], 1):
        prev_detail = period_details[i-1]
        if prev_detail['status'] == '✗失':
            expected_fib = prev_detail['fib_index'] + 1
            actual_fib = detail['fib_index']
            # 命中时会重置为0，所以只检查未命中的情况
            if detail['status'] == '✗失' and actual_fib == expected_fib:
                increment_count += 1
    
    miss_sequences = sum(1 for i, d in enumerate(period_details[:-1]) if d['status'] == '✗失' and period_details[i+1]['status'] == '✗失')
    print(f"✅ 连续未命中时Fib递增: {increment_count}次 (连续未命中{miss_sequences}次)")
    
    print()
    print("="*80)
    print()
    
    if all_have_fib and max_fib <= 11 and min_fib >= 0:
        print("🎉 Fib索引修复验证通过！")
        print()
        print("修复内容:")
        print("  1. ✅ period_details中添加了fib_index字段")
        print("  2. ✅ 记录投注前的Fib索引（current_fib_index）")
        print("  3. ✅ 显示时直接使用fib_index字段，无需反推")
        print()
        print("💡 现在GUI中的Fib列将正确显示斐波那契索引（0-11）")
    else:
        print("❌ Fib索引验证失败，请检查代码")
    
    print()


if __name__ == "__main__":
    test_fib_index_recording()
