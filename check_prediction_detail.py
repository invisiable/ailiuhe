"""
详细检查：最近100期中每一期的预测是否正确
对比回测记录的预测 vs 实际每期应该的预测
"""

import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f"总数据: {len(df)}期\n")

# 模拟GUI回测
test_periods = min(200, len(df))
start_idx = len(df) - test_periods

# 方法1：GUI中的回测方式（一个strategy实例贯穿整个回测）
print("【方法1：GUI方式 - 单个strategy实例贯穿整个回测】")
print(f"{'='*100}\n")

strategy_gui = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
predictions_gui = []

for i in range(start_idx, len(df)):
    train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
    prediction = strategy_gui.predict_top4(train_animals)
    predictions_gui.append(prediction['top4'])
    
    # 更新性能
    actual = str(df.iloc[i]['animal']).strip()
    hit = actual in prediction['top4']
    strategy_gui.update_performance(hit)
    if (i - start_idx + 1) % 10 == 0:
        switched, msg = strategy_gui.check_and_switch_model()
        if switched:
            print(f"第{i-start_idx+1}期: {msg}")

print(f"\n完成！当前模型: {strategy_gui.get_current_model_name()}\n")

# 方法2：每期独立预测（每期创建新的strategy实例）
print("【方法2：独立方式 - 每期创建新的strategy实例】")
print(f"{'='*100}\n")

predictions_independent = []
for i in range(start_idx, len(df)):
    # 每期创建新实例
    strategy_fresh = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
    train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
    prediction = strategy_fresh.predict_top4(train_animals)
    predictions_independent.append(prediction['top4'])

print("完成！\n")

# 对比差异
print("【对比两种方法的差异】")
print(f"{'='*100}\n")

differences = []
for idx in range(len(predictions_gui)):
    if predictions_gui[idx] != predictions_independent[idx]:
        differences.append(idx)

if differences:
    print(f"⚠️ 发现 {len(differences)} 期预测不一致！\n")
    print(f"{'期号':<8} {'日期':<12} {'GUI方式预测':<35} {'独立方式预测':<35} {'实际':<8}")
    print(f"{'-'*110}")
    
    # 只显示前10个差异避免输出太长
    for diff_idx in differences[:10]:
        period_idx = start_idx + diff_idx
        date_str = df.iloc[period_idx]['date']
        actual = df.iloc[period_idx]['animal']
        gui_pred = ', '.join(predictions_gui[diff_idx])
        ind_pred = ', '.join(predictions_independent[diff_idx])
        
        print(f"{diff_idx+1:<8} {date_str:<12} {gui_pred:<35} {ind_pred:<35} {actual:<8}")
    
    if len(differences) > 10:
        print(f"... 还有 {len(differences)-10} 期差异未显示\n")
else:
    print("✅ 两种方法的预测完全一致！\n")

# 特别检查最近100期
print(f"\n{'='*100}")
print("【特别检查：最近100期】")
print(f"{'='*100}\n")

recent_100_start = len(predictions_gui) - 100
print(f"检查最近100期（第{recent_100_start+1}期 ~ 第{len(predictions_gui)}期）\n")

recent_differences = [d for d in differences if d >= recent_100_start]

if recent_differences:
    print(f"⚠️ 最近100期中有 {len(recent_differences)} 期预测不一致！\n")
    
    # 显示详细信息
    print("详细列表（最多显示20期）：")
    print(f"{'期号':<8} {'日期':<12} {'GUI方式':<35} {'独立方式':<35} {'实际':<8} {'GUI命中':<8} {'独立命中':<8}")
    print(f"{'-'*120}")
    
    for diff_idx in recent_differences[:20]:
        period_idx = start_idx + diff_idx
        date_str = df.iloc[period_idx]['date']
        actual = df.iloc[period_idx]['animal']
        gui_pred = ', '.join(predictions_gui[diff_idx])
        ind_pred = ', '.join(predictions_independent[diff_idx])
        
        gui_hit = '✓' if actual in predictions_gui[diff_idx] else '✗'
        ind_hit = '✓' if actual in predictions_independent[diff_idx] else '✗'
        
        # 在最近100期中的编号
        recent_no = diff_idx - recent_100_start + 1
        
        print(f"{recent_no:<8} {date_str:<12} {gui_pred:<35} {ind_pred:<35} {actual:<8} {gui_hit:<8} {ind_hit:<8}")
else:
    print("✅ 最近100期的预测完全一致！\n")

# 计算命中率
print(f"\n{'='*100}")
print("【命中率对比】")
print(f"{'='*100}\n")

# 全部期数
gui_hits_all = sum(1 for i, pred in enumerate(predictions_gui) 
                   if df.iloc[start_idx + i]['animal'] in pred)
ind_hits_all = sum(1 for i, pred in enumerate(predictions_independent) 
                   if df.iloc[start_idx + i]['animal'] in pred)

print(f"全部{test_periods}期：")
print(f"  GUI方式命中率: {gui_hits_all}/{test_periods} = {gui_hits_all/test_periods*100:.2f}%")
print(f"  独立方式命中率: {ind_hits_all}/{test_periods} = {ind_hits_all/test_periods*100:.2f}%")
print(f"  差异: {(gui_hits_all - ind_hits_all)/test_periods*100:+.2f}%\n")

# 最近100期
recent_100_gui = predictions_gui[-100:]
recent_100_ind = predictions_independent[-100:]

gui_hits_100 = sum(1 for i, pred in enumerate(recent_100_gui) 
                   if df.iloc[len(df) - 100 + i]['animal'] in pred)
ind_hits_100 = sum(1 for i, pred in enumerate(recent_100_ind) 
                   if df.iloc[len(df) - 100 + i]['animal'] in pred)

print(f"最近100期：")
print(f"  GUI方式命中率: {gui_hits_100}/100 = {gui_hits_100:.0f}%")
print(f"  独立方式命中率: {ind_hits_100}/100 = {ind_hits_100:.0f}%")
print(f"  差异: {gui_hits_100 - ind_hits_100:+.0f}\n")

print(f"{'='*100}\n")
print("【结论】")
if differences:
    print("❌ GUI方式和独立方式的预测存在差异！")
    print("   原因：GUI方式使用单个strategy实例，其内部状态会随着回测改变")
    print("   影响：模型切换（primary ↔ backup）会导致后续预测不同")
    print("\n建议：")
    print("   1. 如果要显示'回测时的实际预测'，应该使用GUI方式（当前实现）")
    print("   2. 如果要显示'用当时数据重新预测的理想结果'，应该每期独立预测")
else:
    print("✅ 两种方式完全一致，说明模型没有状态依赖问题")
