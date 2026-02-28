"""
对比测试：原版 vs 精准版 TOP15预测器
目标：在保持15个号码的情况下，提高命中率，降低连续不中
"""

import pandas as pd
from top15_predictor import Top15Predictor
from precise_top15_predictor import PreciseTop15Predictor
from betting_strategy import BettingStrategy

print("="*120)
print("TOP15预测器对比测试 - 原版 vs 精准版（均为15个号码）")
print("="*120)
print()

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f"✓ 总数据量: {len(df)}期")

# 测试期数
test_periods = min(300, len(df))
start_idx = len(df) - test_periods
print(f"✓ 测试期数: {test_periods}期\n")

actuals = df.iloc[start_idx:]['number'].values

# ========== 测试原版预测器 ==========
print("="*120)
print("【原版预测器】")
print("="*120)

predictor_original = Top15Predictor()
predictions_original = []

print("正在生成预测...")
for i in range(start_idx, len(df)):
    train_data = df.iloc[:i]['number'].values
    analysis = predictor_original.get_analysis(train_data)
    predictions_original.append(analysis['top15'])

hits_original = [1 if actuals[i] in predictions_original[i] else 0 for i in range(len(actuals))]
hit_rate_original = sum(hits_original) / len(hits_original)

# 统计连续不中
max_miss_orig = 0
current_miss = 0
miss_periods_orig = []
for hit in hits_original:
    if hit == 0:
        current_miss += 1
        max_miss_orig = max(max_miss_orig, current_miss)
    else:
        if current_miss > 0:
            miss_periods_orig.append(current_miss)
        current_miss = 0

print(f"\n原版结果:")
print(f"  命中率: {hit_rate_original*100:.2f}% ({sum(hits_original)}/{len(hits_original)})")
print(f"  最大连续不中: {max_miss_orig}期")
if miss_periods_orig:
    print(f"  平均连续不中: {sum(miss_periods_orig)/len(miss_periods_orig):.2f}期")
    print(f"  连续不中≥5期: {sum(1 for m in miss_periods_orig if m >= 5)}次")
    print(f"  连续不中≥10期: {sum(1 for m in miss_periods_orig if m >= 10)}次")

# 投注回测
betting = BettingStrategy(base_bet=15, win_reward=47, loss_penalty=15)
result_orig = betting.simulate_strategy(predictions_original, list(actuals), 'fibonacci', hit_rate=hit_rate_original)
print(f"  总投注: {result_orig['total_cost']:.2f}元")
print(f"  总收益: {result_orig['total_profit']:+.2f}元")
print(f"  ROI: {result_orig['roi']:+.2f}%")

# ========== 测试精准版预测器 ==========
print("\n" + "="*120)
print("【精准版预测器】")
print("="*120)

predictor_precise = PreciseTop15Predictor()
predictions_precise = []

print("正在生成预测（带性能追踪）...")
for i in range(start_idx, len(df)):
    train_data = df.iloc[:i]['number'].values
    prediction = predictor_precise.predict(train_data)
    predictions_precise.append(prediction)
    
    # 更新性能追踪
    if i > start_idx:
        actual_prev = actuals[i - start_idx - 1]
        prev_pred = predictions_precise[-2] if len(predictions_precise) >= 2 else []
        if prev_pred:
            predictor_precise.update_performance(prev_pred, actual_prev)

hits_precise = [1 if actuals[i] in predictions_precise[i] else 0 for i in range(len(actuals))]
hit_rate_precise = sum(hits_precise) / len(hits_precise)

# 统计连续不中
max_miss_precise = 0
current_miss = 0
miss_periods_precise = []
for hit in hits_precise:
    if hit == 0:
        current_miss += 1
        max_miss_precise = max(max_miss_precise, current_miss)
    else:
        if current_miss > 0:
            miss_periods_precise.append(current_miss)
        current_miss = 0

print(f"\n精准版结果:")
print(f"  命中率: {hit_rate_precise*100:.2f}% ({sum(hits_precise)}/{len(hits_precise)})")
print(f"  最大连续不中: {max_miss_precise}期")
if miss_periods_precise:
    print(f"  平均连续不中: {sum(miss_periods_precise)/len(miss_periods_precise):.2f}期")
    print(f"  连续不中≥5期: {sum(1 for m in miss_periods_precise if m >= 5)}次")
    print(f"  连续不中≥10期: {sum(1 for m in miss_periods_precise if m >= 10)}次")

# 投注回测
result_precise = betting.simulate_strategy(predictions_precise, list(actuals), 'fibonacci', hit_rate=hit_rate_precise)
print(f"  总投注: {result_precise['total_cost']:.2f}元")
print(f"  总收益: {result_precise['total_profit']:+.2f}元")
print(f"  ROI: {result_precise['roi']:+.2f}%")

# ========== 对比总结 ==========
print("\n" + "="*120)
print("【对比总结】")
print("="*120)

print(f"\n{'指标':<20} {'原版':<25} {'精准版':<25} {'改善':<20}")
print("-" * 120)

hit_diff = (hit_rate_precise - hit_rate_original) * 100
print(f"{'命中率':<20} {hit_rate_original*100:>6.2f}% ({sum(hits_original):>3}/{len(hits_original):<3}) "
      f"{hit_rate_precise*100:>6.2f}% ({sum(hits_precise):>3}/{len(hits_precise):<3}) "
      f"{hit_diff:>+6.2f}%")

miss_diff = max_miss_orig - max_miss_precise
print(f"{'最大连续不中':<20} {max_miss_orig:>6}期{' '*15} "
      f"{max_miss_precise:>6}期{' '*15} "
      f"{miss_diff:>+6}期")

if miss_periods_orig and miss_periods_precise:
    avg_miss_orig = sum(miss_periods_orig) / len(miss_periods_orig)
    avg_miss_precise = sum(miss_periods_precise) / len(miss_periods_precise)
    avg_diff = avg_miss_orig - avg_miss_precise
    print(f"{'平均连续不中':<20} {avg_miss_orig:>6.2f}期{' '*15} "
          f"{avg_miss_precise:>6.2f}期{' '*15} "
          f"{avg_diff:>+6.2f}期")

miss5_orig = sum(1 for m in miss_periods_orig if m >= 5)
miss5_precise = sum(1 for m in miss_periods_precise if m >= 5)
print(f"{'连续不中≥5期次数':<20} {miss5_orig:>6}次{' '*15} "
      f"{miss5_precise:>6}次{' '*15} "
      f"{miss5_orig - miss5_precise:>+6}次")

miss10_orig = sum(1 for m in miss_periods_orig if m >= 10)
miss10_precise = sum(1 for m in miss_periods_precise if m >= 10)
print(f"{'连续不中≥10期次数':<20} {miss10_orig:>6}次{' '*15} "
      f"{miss10_precise:>6}次{' '*15} "
      f"{miss10_orig - miss10_precise:>+6}次")

print(f"\n{'投资回报':<20} {'原版':<25} {'精准版':<25} {'差异':<20}")
print("-" * 120)

profit_diff = result_precise['total_profit'] - result_orig['total_profit']
print(f"{'总收益':<20} {result_orig['total_profit']:>+10.2f}元{' '*10} "
      f"{result_precise['total_profit']:>+10.2f}元{' '*10} "
      f"{profit_diff:>+10.2f}元")

roi_diff = result_precise['roi'] - result_orig['roi']
print(f"{'ROI':<20} {result_orig['roi']:>+10.2f}%{' '*10} "
      f"{result_precise['roi']:>+10.2f}%{' '*10} "
      f"{roi_diff:>+10.2f}%")

print("\n" + "="*120)
print("✅ 对比测试完成")
print("="*120)

# 评估
print("\n📊 优化效果评估:")
improvements = []
if max_miss_precise < max_miss_orig:
    improvements.append(f"✅ 最大连续不中降低 {miss_diff} 期（{max_miss_orig}期→{max_miss_precise}期）")
if hit_rate_precise > hit_rate_original:
    improvements.append(f"✅ 命中率提升 {hit_diff:.2f}%")
if miss5_precise < miss5_orig:
    improvements.append(f"✅ 连续不中≥5期次数减少 {miss5_orig - miss5_precise} 次")
if miss10_precise < miss10_orig:
    improvements.append(f"✅ 连续不中≥10期次数减少 {miss10_orig - miss10_precise} 次")
if result_precise['roi'] > result_orig['roi']:
    improvements.append(f"✅ ROI提升 {roi_diff:.2f}%")

if improvements:
    for imp in improvements:
        print(f"  {imp}")
else:
    print("  ⚠️ 精准版未显示显著改善")

print("\n💡 推荐:")
if max_miss_precise < max_miss_orig and hit_rate_precise >= hit_rate_original:
    print("  🎯 强烈推荐使用精准版预测器")
    print("  • 连续不中风险更低")
    print("  • 命中率保持或提升")
    print("  • 成本不变（均为15个号码）")
elif hit_rate_precise > hit_rate_original:
    print("  🎯 推荐使用精准版预测器")
    print("  • 命中率更高")
else:
    print("  📌 继续使用原版预测器或根据实际情况选择")
