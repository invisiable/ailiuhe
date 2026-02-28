"""
验证备份模型已重新开启
"""

import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

print("="*80)
print("验证备份模型已重新开启")
print("="*80)
print()

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f"✓ 数据加载完成: {len(df)}期\n")

# 测试1：创建策略
print("测试1：创建策略（use_emergency_backup=True）")
print("-"*80)

strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
print(f"✓ 策略创建成功")
print(f"  主力模型: {strategy.primary_model.__class__.__name__}")
print(f"  备份模型: {strategy.backup_model.__class__.__name__ if strategy.backup_model else 'None'}")
print(f"  当前使用: {strategy.current_model}")
print()

# 测试2：模拟低命中率触发切换
print("测试2：模拟低命中率场景（应该会触发切换）")
print("-"*80)

test_periods = 30
start_idx = len(df) - test_periods

predictions = []
switches = []

for i in range(start_idx, len(df)):
    train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
    prediction = strategy.predict_top4(train_animals)
    predictions.append(prediction)
    
    actual = str(df.iloc[i]['animal']).strip()
    hit = actual in prediction['top4']
    
    strategy.update_performance(hit)
    
    if (i - start_idx + 1) % 10 == 0:
        switched, msg = strategy.check_and_switch_model()
        period_num = i - start_idx + 1
        print(f"第{period_num}期检查: {msg}")
        if switched:
            switches.append((period_num, msg))

print()

# 测试3：统计使用的模型
print("测试3：使用的模型统计")
print("-"*80)

model_counts = {}
for pred in predictions:
    model = pred['predictor']
    model_counts[model] = model_counts.get(model, 0) + 1

for model, count in model_counts.items():
    print(f"  {model}: {count}期")

print()

# 测试4：最终状态
print("测试4：最终状态检查")
print("-"*80)

stats = strategy.get_performance_stats()
print(f"当前模型: {strategy.get_current_model_name()}")
print(f"当前模型类型: {strategy.current_model}")
print(f"最近{stats['recent_total']}期命中率: {stats['recent_rate']:.1f}%")
print(f"模型切换次数: {len(strategy.switch_history)}次")

if strategy.switch_history:
    print("\n切换记录:")
    for i, switch in enumerate(strategy.switch_history):
        print(f"  第{i+1}次切换: {switch['from']} → {switch['to']}")
        print(f"    原因: {switch['reason']}")
        print(f"    当时命中率: {switch['recent_rate']*100:.1f}%")

print()
print("="*80)
print("结论")
print("="*80)

if strategy.backup_model:
    print("✅ 备份模型已成功开启")
    print(f"✅ 主力模型: {strategy.primary_model.__class__.__name__}")
    print(f"✅ 备份模型: {strategy.backup_model.__class__.__name__}")
    
    if len(strategy.switch_history) > 0:
        print(f"✅ 应急机制正常工作（发生了{len(strategy.switch_history)}次切换）")
    else:
        print("⚠️ 测试期内未触发切换（可能命中率良好或数据不足）")
else:
    print("❌ 备份模型未开启")

print()
print("备份模型优势:")
print("  • 在主力模型表现不佳时自动切换到备份模型")
print("  • 提供多层保护机制，降低风险")
print("  • 自适应市场变化，灵活应对异常情况")
