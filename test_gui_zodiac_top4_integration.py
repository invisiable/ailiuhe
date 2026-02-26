"""
测试GUI中生肖TOP4投注策略集成
验证新的RecommendedZodiacTop4Strategy是否正确集成到GUI中
"""

import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

def test_gui_integration():
    """模拟GUI中的流程"""
    
    print("="*80)
    print("测试GUI生肖TOP4投注策略集成")
    print("="*80)
    
    # 1. 加载数据（模拟GUI的data loading）
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"\n✅ 数据加载完成: {len(df)}期")
    
    # 2. 初始化策略（模拟GUI的strategy initialization）
    strategy = RecommendedZodiacTop4Strategy()
    print("✅ 策略初始化完成")
    
    # 3. 验证最近100期（模拟GUI的validation logic）
    test_periods = min(100, len(df))
    start_idx = len(df) - test_periods
    
    print(f"\n开始验证最近{test_periods}期...")
    
    hit_records = []
    predictions_top4 = []
    
    for i in range(test_periods):
        idx = start_idx + i
        history_animals = df['animal'].iloc[:idx].tolist()
        actual_animal = df.iloc[idx]['animal']
        
        # 预测
        prediction_result = strategy.predict_top4(history_animals)
        predicted_top4 = prediction_result['top4']
        predictions_top4.append(predicted_top4)
        
        # 检查命中
        hit = actual_animal in predicted_top4
        hit_records.append(hit)
        
        # 更新策略性能
        strategy.update_performance(hit)
    
    # 4. 计算统计数据（模拟GUI的statistics calculation）
    hits = sum(hit_records)
    hit_rate = hits / len(hit_records)
    total_investment = 16 * len(hit_records)
    total_profit = hits * 30 - (len(hit_records) - hits) * 16
    roi = (total_profit / total_investment) * 100
    
    # 计算最大连亏
    max_consecutive_losses = 0
    current_consecutive_losses = 0
    for hit in hit_records:
        if hit:
            current_consecutive_losses = 0
        else:
            current_consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
    
    print(f"\n{'='*80}")
    print("验证结果统计")
    print(f"{'='*80}")
    print(f"测试期数: {test_periods}期")
    print(f"命中次数: {hits}次")
    print(f"命中率: {hit_rate*100:.2f}%")
    print(f"总投入: {total_investment}元")
    print(f"净收益: {total_profit:+.0f}元")
    print(f"ROI: {roi:+.2f}%")
    print(f"最大连亏: {max_consecutive_losses}期")
    print(f"\n模型切换记录: {len(strategy.switch_history)}次")
    
    # 5. 下期预测（模拟GUI的next period prediction）
    all_animals = df['animal'].tolist()
    prediction_result = strategy.predict_top4(all_animals)
    next_top4 = prediction_result['top4']
    current_model = strategy.get_current_model_name()
    
    print(f"\n{'='*80}")
    print("下期投注建议")
    print(f"{'='*80}")
    print(f"预测TOP4: {', '.join(next_top4)}")
    print(f"当前模型: {current_model}")
    print(f"推荐投注: 16元 (每个生肖4元)")
    print(f"如果命中: +30元")
    print(f"如果未中: -16元")
    
    # 6. 验证结果
    print(f"\n{'='*80}")
    print("集成测试结果")
    print(f"{'='*80}")
    
    if hit_rate >= 0.45:  # 至少45%命中率
        print("✅ 命中率达标（>=45%）")
        result = "通过"
    else:
        print(f"⚠️  命中率偏低（{hit_rate*100:.2f}% < 45%）")
        result = "需关注"
    
    if roi > 0:
        print(f"✅ ROI为正（{roi:+.2f}%）")
        if result == "通过":
            result = "优秀"
    else:
        print(f"❌ ROI为负（{roi:+.2f}%）")
        result = "失败"
    
    if max_consecutive_losses <= 8:
        print(f"✅ 最大连亏可控（{max_consecutive_losses}期 <= 8期）")
    else:
        print(f"⚠️  最大连亏较高（{max_consecutive_losses}期 > 8期）")
        if result == "优秀":
            result = "通过"
    
    print(f"\n{'='*80}")
    print(f"最终评估: {result}")
    print(f"{'='*80}\n")
    
    return result == "优秀" or result == "通过"

if __name__ == "__main__":
    success = test_gui_integration()
    if success:
        print("🎉 GUI集成测试成功！新策略工作正常。")
    else:
        print("⚠️  GUI集成测试完成，但性能不及预期。")
