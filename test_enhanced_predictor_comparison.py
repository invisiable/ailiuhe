"""
对比测试：原版 vs 增强版 TOP15预测器
重点观察：连续不中期数的改善
"""

import pandas as pd
from top15_predictor import Top15Predictor
from enhanced_top15_predictor import EnhancedTop15Predictor
from betting_strategy import BettingStrategy

print("="*120)
print("TOP15预测器对比测试 - 原版 vs 增强版")
print("="*120)
print()

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f"✓ 总数据量: {len(df)}期")

# 测试期数
test_periods = min(300, len(df))
start_idx = len(df) - test_periods
print(f"✓ 测试期数: {test_periods}期\n")

# ========== 测试原版预测器 ==========
print("="*120)
print("【原版预测器】- 固定15个号码")
print("="*120)

predictor_original = Top15Predictor()
predictions_original = []
actuals = []

print("正在生成预测...")
for i in range(start_idx, len(df)):
    train_data = df.iloc[:i]['number'].values
    analysis = predictor_original.get_analysis(train_data)
    top15 = analysis['top15']
    predictions_original.append(top15)
    
    actual = df.iloc[i]['number']
    actuals.append(actual)

# 计算连续不中统计
hits_original = [1 if actuals[i] in predictions_original[i] else 0 for i in range(len(actuals))]
hit_rate_original = sum(hits_original) / len(hits_original)

# 统计连续不中
max_consecutive_misses_original = 0
current_misses = 0
miss_periods_original = []

for hit in hits_original:
    if hit == 0:
        current_misses += 1
        max_consecutive_misses_original = max(max_consecutive_misses_original, current_misses)
    else:
        if current_misses > 0:
            miss_periods_original.append(current_misses)
        current_misses = 0

print(f"\n原版结果:")
print(f"  命中率: {hit_rate_original*100:.2f}% ({sum(hits_original)}/{len(hits_original)})")
print(f"  最大连续不中: {max_consecutive_misses_original}期")
print(f"  平均连续不中: {sum(miss_periods_original)/len(miss_periods_original):.2f}期" if miss_periods_original else "  平均连续不中: 0期")
print(f"  连续不中≥5期的次数: {sum(1 for m in miss_periods_original if m >= 5)}次")
print(f"  连续不中≥10期的次数: {sum(1 for m in miss_periods_original if m >= 10)}次")

# 投注策略回测
betting_original = BettingStrategy(base_bet=15, win_reward=47, loss_penalty=15)
result_original = betting_original.simulate_strategy(predictions_original, actuals, 'fibonacci', hit_rate=hit_rate_original)
print(f"  总投注: {result_original['total_cost']:.2f}元")
print(f"  总收益: {result_original['total_profit']:+.2f}元")
print(f"  ROI: {result_original['roi']:+.2f}%")

# ========== 测试增强版预测器 ==========
print("\n" + "="*120)
print("【增强版预测器】- 自适应15-20个号码")
print("="*120)

predictor_enhanced = EnhancedTop15Predictor()
predictions_enhanced = []

print("正在生成预测（带自适应）...")
for i in range(start_idx, len(df)):
    train_data = df.iloc[:i]['number'].values
    
    # 预测
    prediction = predictor_enhanced.predict(train_data, adaptive=True)
    predictions_enhanced.append(prediction)
    
    # 更新性能（用于自适应）
    actual = actuals[i - start_idx]
    hit = actual in prediction
    predictor_enhanced.update_performance(hit)

# 计算连续不中统计
hits_enhanced = [1 if actuals[i] in predictions_enhanced[i] else 0 for i in range(len(actuals))]
hit_rate_enhanced = sum(hits_enhanced) / len(hits_enhanced)

# 统计连续不中
max_consecutive_misses_enhanced = 0
current_misses = 0
miss_periods_enhanced = []

for hit in hits_enhanced:
    if hit == 0:
        current_misses += 1
        max_consecutive_misses_enhanced = max(max_consecutive_misses_enhanced, current_misses)
    else:
        if current_misses > 0:
            miss_periods_enhanced.append(current_misses)
        current_misses = 0

print(f"\n增强版结果:")
print(f"  命中率: {hit_rate_enhanced*100:.2f}% ({sum(hits_enhanced)}/{len(hits_enhanced)})")
print(f"  最大连续不中: {max_consecutive_misses_enhanced}期")
print(f"  平均连续不中: {sum(miss_periods_enhanced)/len(miss_periods_enhanced):.2f}期" if miss_periods_enhanced else "  平均连续不中: 0期")
print(f"  连续不中≥5期的次数: {sum(1 for m in miss_periods_enhanced if m >= 5)}次")
print(f"  连续不中≥10期的次数: {sum(1 for m in miss_periods_enhanced if m >= 10)}次")

# 计算平均预测号码数量
avg_pred_count = sum(len(p) for p in predictions_enhanced) / len(predictions_enhanced)
print(f"  平均预测号码数: {avg_pred_count:.1f}个")

# 投注策略回测（动态成本）
total_cost_enhanced = 0
total_profit_enhanced = 0
consecutive_losses = 0

from betting_strategy import BettingStrategy
betting_temp = BettingStrategy(base_bet=15, win_reward=47, loss_penalty=15)

for i, (prediction, actual) in enumerate(zip(predictions_enhanced, actuals)):
    # 动态成本（预测几个号码就花几元）
    pred_count = len(prediction)
    
    # 计算斐波那契倍数
    multiplier, _ = betting_temp.calculate_fibonacci_bet(consecutive_losses)
    
    cost = pred_count * multiplier
    total_cost_enhanced += cost
    
    if actual in prediction:
        # 命中
        reward = 47 * multiplier
        profit = reward - cost
        total_profit_enhanced += profit
        consecutive_losses = 0
    else:
        # 未中
        total_profit_enhanced -= cost
        consecutive_losses += 1

roi_enhanced = (total_profit_enhanced / total_cost_enhanced * 100) if total_cost_enhanced > 0 else 0

print(f"  总投注: {total_cost_enhanced:.2f}元")
print(f"  总收益: {total_profit_enhanced:+.2f}元")
print(f"  ROI: {roi_enhanced:+.2f}%")

# ========== 对比总结 ==========
print("\n" + "="*120)
print("【对比总结】")
print("="*120)

print(f"\n{'指标':<20} {'原版':<25} {'增强版':<25} {'改善':<20}")
print("-" * 120)

hit_improvement = (hit_rate_enhanced - hit_rate_original) * 100
print(f"{'命中率':<20} {hit_rate_original*100:>6.2f}% ({sum(hits_original):>3}/{len(hits_original):<3}) "
      f"{hit_rate_enhanced*100:>6.2f}% ({sum(hits_enhanced):>3}/{len(hits_enhanced):<3}) "
      f"{hit_improvement:>+6.2f}%")

miss_improvement = max_consecutive_misses_original - max_consecutive_misses_enhanced
print(f"{'最大连续不中':<20} {max_consecutive_misses_original:>6}期{' '*15} "
      f"{max_consecutive_misses_enhanced:>6}期{' '*15} "
      f"{miss_improvement:>+6}期")

if miss_periods_original and miss_periods_enhanced:
    avg_miss_orig = sum(miss_periods_original) / len(miss_periods_original)
    avg_miss_enh = sum(miss_periods_enhanced) / len(miss_periods_enhanced)
    avg_improvement = avg_miss_orig - avg_miss_enh
    print(f"{'平均连续不中':<20} {avg_miss_orig:>6.2f}期{' '*15} "
          f"{avg_miss_enh:>6.2f}期{' '*15} "
          f"{avg_improvement:>+6.2f}期")

miss_5_orig = sum(1 for m in miss_periods_original if m >= 5)
miss_5_enh = sum(1 for m in miss_periods_enhanced if m >= 5)
print(f"{'连续不中≥5期次数':<20} {miss_5_orig:>6}次{' '*15} "
      f"{miss_5_enh:>6}次{' '*15} "
      f"{miss_5_orig - miss_5_enh:>+6}次")

miss_10_orig = sum(1 for m in miss_periods_original if m >= 10)
miss_10_enh = sum(1 for m in miss_periods_enhanced if m >= 10)
print(f"{'连续不中≥10期次数':<20} {miss_10_orig:>6}次{' '*15} "
      f"{miss_10_enh:>6}次{' '*15} "
      f"{miss_10_orig - miss_10_enh:>+6}次")

print(f"\n{'投资回报':<20} {'原版':<25} {'增强版':<25} {'差异':<20}")
print("-" * 120)

profit_improvement = total_profit_enhanced - result_original['total_profit']
print(f"{'总收益':<20} {result_original['total_profit']:>+10.2f}元{' '*10} "
      f"{total_profit_enhanced:>+10.2f}元{' '*10} "
      f"{profit_improvement:>+10.2f}元")

roi_improvement = roi_enhanced - result_original['roi']
print(f"{'ROI':<20} {result_original['roi']:>+10.2f}%{' '*10} "
      f"{roi_enhanced:>+10.2f}%{' '*10} "
      f"{roi_improvement:>+10.2f}%")

cost_diff = total_cost_enhanced - result_original['total_cost']
print(f"{'总投注':<20} {result_original['total_cost']:>10.2f}元{' '*10} "
      f"{total_cost_enhanced:>10.2f}元{' '*10} "
      f"{cost_diff:>+10.2f}元")

print("\n" + "="*120)
print("✅ 对比测试完成")
print("="*120)

# 评估结论
print("\n📊 优化效果评估:")
if max_consecutive_misses_enhanced < max_consecutive_misses_original:
    print(f"  ✅ 最大连续不中降低了 {miss_improvement} 期（从{max_consecutive_misses_original}期降至{max_consecutive_misses_enhanced}期）")
else:
    print(f"  ⚠️ 最大连续不中未改善")

if hit_rate_enhanced > hit_rate_original:
    print(f"  ✅ 命中率提升了 {hit_improvement:.2f}%")
else:
    print(f"  ⚠️ 命中率未提升")

if miss_5_enh < miss_5_orig:
    print(f"  ✅ 连续不中≥5期的次数减少了 {miss_5_orig - miss_5_enh} 次")

if miss_10_enh < miss_10_orig:
    print(f"  ✅ 连续不中≥10期的次数减少了 {miss_10_orig - miss_10_enh} 次")

if roi_enhanced > result_original['roi']:
    print(f"  ✅ ROI提升了 {roi_improvement:.2f}%")

print("\n💡 建议:")
if max_consecutive_misses_enhanced < max_consecutive_misses_original and hit_rate_enhanced > hit_rate_original:
    print("  强烈推荐使用增强版预测器！")
    print("  • 连续不中风险显著降低")
    print("  • 命中率有所提升")
    print("  • 整体投资回报更优")
elif max_consecutive_misses_enhanced < max_consecutive_misses_original:
    print("  推荐使用增强版预测器")
    print("  • 连续不中风险降低")
    print("  • 风险控制更好")
else:
    print("  增强版在某些指标上有改善，可根据实际情况选择")
