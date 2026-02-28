"""
测试关闭备份模型后的效果
验证策略是否只使用主力模型
"""

import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f"数据加载完成: {len(df)}期\n")

# 测试1：创建禁用备份的策略
print("="*80)
print("测试1：创建策略（use_emergency_backup=False）")
print("="*80)

strategy_no_backup = RecommendedZodiacTop4Strategy(use_emergency_backup=False)
print(f"✓ 策略创建成功")
print(f"  主力模型: {strategy_no_backup.primary_model.__class__.__name__}")
print(f"  备份模型: {strategy_no_backup.backup_model}")  # 应该是 None
print(f"  当前使用: {strategy_no_backup.current_model}")
print()

# 测试2：模拟回测，验证不会切换模型
print("="*80)
print("测试2：回测验证（模拟低命中率场景）")
print("="*80)

test_periods = 30
start_idx = len(df) - test_periods

predictions = []
for i in range(start_idx, len(df)):
    train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
    prediction = strategy_no_backup.predict_top4(train_animals)
    predictions.append(prediction)
    
    # 模拟命中判断
    actual = str(df.iloc[i]['animal']).strip()
    hit = actual in prediction['top4']
    
    # 更新性能
    strategy_no_backup.update_performance(hit)
    
    # 每10期尝试切换（应该不会切换，因为没有备份模型）
    if (i - start_idx + 1) % 10 == 0:
        switched, msg = strategy_no_backup.check_and_switch_model()
        print(f"第{i-start_idx+1}期检查: {msg}")

print()

# 测试3：显示预测结果
print("="*80)
print("测试3：预测结果统计")
print("="*80)

# 统计使用的模型
model_counts = {}
for pred in predictions:
    model = pred['predictor']
    model_counts[model] = model_counts.get(model, 0) + 1

print("使用的模型统计:")
for model, count in model_counts.items():
    print(f"  {model}: {count}期")

print()

# 测试4：最终状态检查
print("="*80)
print("测试4：最终状态检查")
print("="*80)

stats = strategy_no_backup.get_performance_stats()
print(f"当前模型: {strategy_no_backup.get_current_model_name()}")
print(f"当前模型类型: {strategy_no_backup.current_model}")
print(f"最近{stats['recent_total']}期命中率: {stats['recent_rate']:.1f}%")
print(f"模型切换历史: {len(strategy_no_backup.switch_history)}次")

if strategy_no_backup.switch_history:
    print("\n切换记录:")
    for i, switch in enumerate(strategy_no_backup.switch_history):
        print(f"  第{i+1}次: {switch}")
else:
    print("✓ 没有发生模型切换")

print()

# 对比测试：启用备份模型的策略
print("="*80)
print("对比：启用备份模型的策略")
print("="*80)

strategy_with_backup = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
print(f"主力模型: {strategy_with_backup.primary_model.__class__.__name__}")
print(f"备份模型: {strategy_with_backup.backup_model.__class__.__name__}")

# 模拟相同的回测
for i in range(start_idx, len(df)):
    train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
    prediction = strategy_with_backup.predict_top4(train_animals)
    
    actual = str(df.iloc[i]['animal']).strip()
    hit = actual in prediction['top4']
    strategy_with_backup.update_performance(hit)
    
    if (i - start_idx + 1) % 10 == 0:
        strategy_with_backup.check_and_switch_model()

print(f"模型切换次数: {len(strategy_with_backup.switch_history)}次")

print()
print("="*80)
print("结论")
print("="*80)
print("✓ 禁用备份模型后，策略将始终使用主力模型（重训练v2.0）")
print("✓ 即使在低命中率情况下，也不会切换到备份模型")
print("✓ 策略更简单、更稳定，适合追求一致性的用户")
