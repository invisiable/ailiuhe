"""
分析不同预测号码数量对收益率的影响
测试范围: TOP15 到 TOP24
目标: 找出最优的预测号码数量配置
"""

import pandas as pd
import numpy as np
from top15_predictor import Top15Predictor
from betting_strategy import BettingStrategy
import time

print("="*120)
print("TOP-N 预测号码数量优化分析")
print("测试范围: TOP15 ~ TOP24")
print("="*120)
print()

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f"✓ 总数据量: {len(df)}期")

# 测试期数
test_periods = min(200, len(df))
start_idx = len(df) - test_periods
print(f"✓ 测试期数: {test_periods}期")
print()

# 准备数据
actuals = df.iloc[start_idx:]['number'].values
dates = df.iloc[start_idx:]['date'].values

print("正在生成不同数量的TOP-N预测...")
print("-" * 120)

# 存储所有结果
results = []

# 测试从15到24个号码
for n in range(15, 25):
    print(f"\n处理 TOP{n} 预测...")
    
    predictor = Top15Predictor()
    predictions = []
    
    # 生成预测
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        
        # 直接调用predict获取综合评分后的候选号码
        # 我们需要修改predict方法来返回更多候选
        pattern = predictor.analyze_pattern(train_data)
        
        # 运行所有方法
        methods = [
            (predictor.method_frequency_advanced(pattern, 25), 0.25),
            (predictor.method_zone_dynamic(pattern, 25), 0.25),
            (predictor.method_cyclic_pattern(pattern, 25), 0.25),
            (predictor.method_gap_prediction(pattern, 25), 0.25)
        ]
        
        # 综合评分
        scores = {}
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                scores[num] = scores.get(num, 0) + score
        
        # 排序并返回top_n个
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_n = [num for num, _ in final[:n]]
        predictions.append(top_n)
    
    # 计算命中率
    hits = sum(1 for i in range(len(actuals)) if actuals[i] in predictions[i])
    hit_rate = hits / len(actuals)
    
    print(f"  命中次数: {hits}/{len(actuals)}")
    print(f"  命中率: {hit_rate*100:.2f}%")
    
    # 使用斐波那契策略进行回测
    # 重要：成本等于预测号码数量（买几个号码就花几元）
    betting = BettingStrategy(base_bet=n, win_reward=47, loss_penalty=n)
    result = betting.simulate_strategy(predictions, actuals, 'fibonacci', hit_rate=hit_rate)
    
    # 保存结果
    results.append({
        'top_n': n,
        'hit_count': hits,
        'hit_rate': hit_rate * 100,
        'total_cost': result['total_cost'],
        'total_profit': result['total_profit'],
        'roi': result['roi'],
        'max_consecutive_losses': result['max_consecutive_losses'],
        'wins': result['wins'],
        'losses': result['losses']
    })
    
    print(f"  总投注: {result['total_cost']:.2f}元")
    print(f"  总收益: {result['total_profit']:+.2f}元")
    print(f"  ROI: {result['roi']:+.2f}%")
    print(f"  最大连亏: {result['max_consecutive_losses']}期")

# 转换为DataFrame
df_results = pd.DataFrame(results)

# 显示对比表格
print("\n" + "="*120)
print("TOP-N 数量优化对比分析")
print("="*120)
print()
print(f"{'TOP-N':<8} {'命中次数':<12} {'命中率':<10} {'总投注(元)':<12} {'总收益(元)':<12} {'ROI(%)':<10} {'最大连亏':<10}")
print("-" * 120)

for _, row in df_results.iterrows():
    print(
        f"TOP{row['top_n']:<4} "
        f"{row['hit_count']:<4}/{len(actuals):<5} "
        f"{row['hit_rate']:>7.2f}%  "
        f"{row['total_cost']:>10.2f}  "
        f"{row['total_profit']:>+11.2f}  "
        f"{row['roi']:>+9.2f}%  "
        f"{row['max_consecutive_losses']:>6}期"
    )

print()

# 找出最优配置
print("="*120)
print("关键指标排名")
print("="*120)
print()

# 按命中率排序
print("【按命中率排序 - TOP5】")
df_by_hitrate = df_results.sort_values('hit_rate', ascending=False).head()
for i, (_, row) in enumerate(df_by_hitrate.iterrows(), 1):
    print(f"  {i}. TOP{row['top_n']}: {row['hit_rate']:.2f}% (收益: {row['total_profit']:+.2f}元, ROI: {row['roi']:+.2f}%)")

print()

# 按ROI排序
print("【按ROI排序 - TOP5】")
df_by_roi = df_results.sort_values('roi', ascending=False).head()
for i, (_, row) in enumerate(df_by_roi.iterrows(), 1):
    print(f"  {i}. TOP{row['top_n']}: ROI {row['roi']:+.2f}% (收益: {row['total_profit']:+.2f}元, 命中率: {row['hit_rate']:.2f}%)")

print()

# 按总收益排序
print("【按总收益排序 - TOP5】")
df_by_profit = df_results.sort_values('total_profit', ascending=False).head()
for i, (_, row) in enumerate(df_by_profit.iterrows(), 1):
    print(f"  {i}. TOP{row['top_n']}: {row['total_profit']:+.2f}元 (ROI: {row['roi']:+.2f}%, 命中率: {row['hit_rate']:.2f}%)")

print()

# 综合评分（命中率 + ROI + 总收益的标准化得分）
df_results['hit_rate_score'] = (df_results['hit_rate'] - df_results['hit_rate'].min()) / (df_results['hit_rate'].max() - df_results['hit_rate'].min())
df_results['roi_score'] = (df_results['roi'] - df_results['roi'].min()) / (df_results['roi'].max() - df_results['roi'].min())
df_results['profit_score'] = (df_results['total_profit'] - df_results['total_profit'].min()) / (df_results['total_profit'].max() - df_results['total_profit'].min())
df_results['composite_score'] = df_results['hit_rate_score'] * 0.4 + df_results['roi_score'] * 0.3 + df_results['profit_score'] * 0.3

print("【综合评分排序 - TOP5】")
print("  (综合评分 = 命中率40% + ROI30% + 总收益30%)")
df_by_composite = df_results.sort_values('composite_score', ascending=False).head()
for i, (_, row) in enumerate(df_by_composite.iterrows(), 1):
    print(f"  {i}. TOP{row['top_n']}: 综合得分 {row['composite_score']:.3f}")
    print(f"     └─ 命中率: {row['hit_rate']:.2f}%, ROI: {row['roi']:+.2f}%, 收益: {row['total_profit']:+.2f}元")

print()

# 最优推荐
best_composite = df_results.loc[df_results['composite_score'].idxmax()]
best_hitrate = df_results.loc[df_results['hit_rate'].idxmax()]
best_roi = df_results.loc[df_results['roi'].idxmax()]
best_profit = df_results.loc[df_results['total_profit'].idxmax()]

print("="*120)
print("最优配置推荐")
print("="*120)
print()
print(f"🏆 综合最优: TOP{int(best_composite['top_n'])}")
print(f"   └─ 命中率: {best_composite['hit_rate']:.2f}%, ROI: {best_composite['roi']:+.2f}%, 收益: {best_composite['total_profit']:+.2f}元")
print()
print(f"🎯 命中率最高: TOP{int(best_hitrate['top_n'])} ({best_hitrate['hit_rate']:.2f}%)")
print(f"💰 ROI最高: TOP{int(best_roi['top_n'])} ({best_roi['roi']:+.2f}%)")
print(f"💵 收益最高: TOP{int(best_profit['top_n'])} ({best_profit['total_profit']:+.2f}元)")
print()

# 趋势分析
print("="*120)
print("趋势分析")
print("="*120)
print()

# 计算相关性
corr_topn_hitrate = np.corrcoef(df_results['top_n'], df_results['hit_rate'])[0,1]
corr_topn_roi = np.corrcoef(df_results['top_n'], df_results['roi'])[0,1]
corr_topn_profit = np.corrcoef(df_results['top_n'], df_results['total_profit'])[0,1]

print(f"TOP-N数量 vs 命中率 相关性: {corr_topn_hitrate:+.3f}")
print(f"TOP-N数量 vs ROI 相关性: {corr_topn_roi:+.3f}")
print(f"TOP-N数量 vs 总收益 相关性: {corr_topn_profit:+.3f}")
print()

if corr_topn_hitrate > 0.7:
    print("📈 命中率随TOP-N数量增加呈强正相关 → 建议选择较大的N值")
elif corr_topn_hitrate < -0.7:
    print("📉 命中率随TOP-N数量增加呈强负相关 → 建议选择较小的N值")
else:
    print("📊 命中率与TOP-N数量关联不强 → 需综合考虑其他指标")

print()

if corr_topn_roi > 0.5:
    print("📈 ROI随TOP-N数量增加而提升 → 更多号码能提高回报率")
elif corr_topn_roi < -0.5:
    print("📉 ROI随TOP-N数量增加而下降 → 精选少量号码更高效")
else:
    print("📊 ROI与TOP-N数量关联较弱 → 可能存在最优范围")

print()

# 保存结果到CSV
output_file = 'top_n_optimization_analysis.csv'
df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✓ 详细数据已保存至: {output_file}")

print()
print("="*120)
print("✅ 分析完成！")
print("="*120)
print()
print("建议:")
if best_composite['top_n'] == 15:
    print(f"  • 当前TOP15配置已经是最优选择，无需调整")
elif best_composite['top_n'] < 15:
    print(f"  • 建议缩小预测范围至 TOP{int(best_composite['top_n'])}，可提升整体表现")
else:
    print(f"  • 建议扩大预测范围至 TOP{int(best_composite['top_n'])}，可提升整体表现")
print(f"  • 预期命中率: {best_composite['hit_rate']:.2f}%")
print(f"  • 预期ROI: {best_composite['roi']:+.2f}%")
print(f"  • 预期{test_periods}期收益: {best_composite['total_profit']:+.2f}元")
