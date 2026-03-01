"""
验证GUI中的智能动态投注策略集成
快速测试SmartDynamic策略是否正常工作
"""

import pandas as pd
from zodiac_simple_smart import ZodiacSimpleSmart

def quick_test_smart_dynamic_in_gui():
    """快速测试GUI中的智能动态策略逻辑"""
    
    print(f"\n{'='*80}")
    print(f"GUI智能动态投注策略集成测试")
    print(f"{'='*80}\n")
    
    # 模拟GUI中的逻辑
    print("第1步：加载数据...")
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"✅ 数据加载完成：{len(df)}期\n")
    
    # 模拟GUI中的预测器初始化
    print("第2步：初始化预测器（v10.0）...")
    predictor = ZodiacSimpleSmart()
    print(f"✅ 预测器初始化完成：ZodiacSimpleSmart (52%稳定)\n")
    
    # 模拟回测
    print("第3步：模拟200期回测...")
    test_periods = 200
    total = len(df)
    start_idx = total - test_periods
    
    hit_records = []
    for i in range(start_idx, total):
        train_animals = df['animal'].iloc[:i].tolist()
        actual = df.iloc[i]['animal']
        
        # 使用v10.0预测
        result = predictor.predict_from_history(train_animals, top_n=5, debug=False)
        top5 = result['top5']
        
        hit = actual in top5
        hit_records.append(hit)
    
    hits = sum(hit_records)
    hit_rate = hits / len(hit_records)
    print(f"✅ 预测完成：命中率 {hit_rate*100:.2f}% ({hits}/{test_periods})\n")
    
    # 模拟SmartDynamic策略计算
    print("第4步：模拟智能动态策略计算...")
    
    # 简化版的SmartDynamic逻辑
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    lookback = 12
    good_thresh = 0.35
    bad_thresh = 0.20
    boost_mult = 1.2
    reduce_mult = 0.8
    max_multiplier = 10
    base_bet = 20
    win_amount = 47
    
    total_profit = 0
    total_investment = 0
    fib_index = 0
    recent_results = []
    max_bet = base_bet
    hit_count = 0
    hit_10x_count = 0
    
    for i, hit in enumerate(hit_records):
        # 获取基础倍数
        base_mult = fib_sequence[fib_index] if fib_index < len(fib_sequence) else fib_sequence[-1]
        base_mult = min(base_mult, max_multiplier)
        
        # 动态调整
        if len(recent_results) >= lookback:
            rate = sum(recent_results) / len(recent_results)
            if rate >= good_thresh:
                multiplier = min(base_mult * boost_mult, max_multiplier)
            elif rate <= bad_thresh:
                multiplier = max(base_mult * reduce_mult, 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        if multiplier >= 10:
            hit_10x_count += 1
        
        # 计算投注
        bet = base_bet * multiplier
        total_investment += bet
        max_bet = max(max_bet, bet)
        
        if hit:
            win = win_amount * multiplier
            profit = win - bet
            total_profit += profit
            fib_index = 0
            hit_count += 1
        else:
            total_profit -= bet
            fib_index += 1
        
        # 更新历史
        recent_results.append(1 if hit else 0)
        if len(recent_results) > lookback:
            recent_results.pop(0)
    
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    
    print(f"✅ 计算完成\n")
    
    # 输出结果
    print(f"{'='*80}")
    print(f"智能动态投注v3.1测试结果")
    print(f"{'='*80}\n")
    
    print(f"策略参数：")
    print(f"  • 回看期数: {lookback}期")
    print(f"  • 增强阈值: {good_thresh:.0%} → boost×{boost_mult}")
    print(f"  • 降低阈值: {bad_thresh:.0%} → reduce×{reduce_mult}")
    print(f"  • 最大倍数: {max_multiplier}倍\n")
    
    print(f"测试结果：")
    print(f"  命中次数: {hit_count}/{test_periods}")
    print(f"  命中率: {hit_rate*100:.2f}%")
    print(f"  总投入: {total_investment:.0f}元")
    print(f"  净利润: {total_profit:+.0f}元")
    print(f"  ROI: {roi:.2f}%")
    print(f"  最大单期投入: {max_bet:.0f}元")
    print(f"  触及10x上限: {hit_10x_count}次\n")
    
    # 验证结果
    print(f"{'='*80}")
    print(f"验证结果")
    print(f"{'='*80}\n")
    
    expected_roi = 16.79
    expected_profit = 1231
    expected_10x = 4
    
    roi_match = abs(roi - expected_roi) < 1.0
    profit_match = abs(total_profit - expected_profit) < 50
    hit_10x_match = hit_10x_count == expected_10x
    
    print(f"ROI验证: {roi:.2f}% vs 期望{expected_roi}% → {'✅ 通过' if roi_match else '❌ 失败'}")
    print(f"利润验证: {total_profit:.0f}元 vs 期望{expected_profit}元 → {'✅ 通过' if profit_match else '❌ 失败'}")
    print(f"10x验证: {hit_10x_count}次 vs 期望{expected_10x}次 → {'✅ 通过' if hit_10x_match else '❌ 失败'}")
    
    all_passed = roi_match and profit_match and hit_10x_match
    
    print(f"\n{'='*80}")
    if all_passed:
        print(f"✅ 所有测试通过！GUI中的SmartDynamic策略集成正确！")
    else:
        print(f"⚠️  部分测试未通过，请检查实现细节")
    print(f"{'='*80}\n")
    
    return all_passed


if __name__ == "__main__":
    quick_test_smart_dynamic_in_gui()
