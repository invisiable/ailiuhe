"""
可视化对比原模型和改进模型的表现
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取数据
original_df = pd.read_csv('odd_even_validation_50periods.csv', encoding='utf-8-sig')
improved_df = pd.read_csv('improved_odd_even_validation_50periods.csv', encoding='utf-8-sig')

# 原模型列名: 期数,日期,实际数字,实际奇偶,预测奇偶,置信度,是否正确
# 改进模型列名: period,date,predicted,actual,actual_number,correct,confidence

# 转换原模型数据
original_correct = original_df.iloc[:, 6].apply(lambda x: 1 if '✅' in str(x) else 0)
original_predicted = original_df.iloc[:, 4]
original_actual = original_df.iloc[:, 3]

# 改进模型已经是正确的格式
improved_correct = improved_df['correct'].astype(int)
improved_predicted = improved_df['predicted']
improved_actual = improved_df['actual']

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 总体准确率对比
ax1 = axes[0, 0]
models = ['原模型\n(GB, 28特征)', '改进模型\n(Ensemble, 72特征)']
accuracies = [original_correct.mean() * 100, improved_correct.mean() * 100]
colors = ['#3498db', '#e74c3c']
bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.axhline(y=50, color='green', linestyle='--', label='随机猜测线(50%)', linewidth=2)
ax1.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
ax1.set_title('总体准确率对比 (50期)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 2. 累计准确率走势
ax2 = axes[0, 1]
original_cumsum = original_correct.cumsum() / pd.Series(range(1, len(original_correct) + 1)) * 100
improved_cumsum = improved_correct.cumsum() / pd.Series(range(1, len(improved_correct) + 1)) * 100
ax2.plot(range(1, 51), original_cumsum, 'o-', label='原模型', color='#3498db', linewidth=2, markersize=4)
ax2.plot(range(1, 51), improved_cumsum, 's-', label='改进模型', color='#e74c3c', linewidth=2, markersize=4)
ax2.axhline(y=50, color='green', linestyle='--', label='随机猜测线', linewidth=2, alpha=0.7)
ax2.set_xlabel('预测期数', fontsize=12, fontweight='bold')
ax2.set_ylabel('累计准确率 (%)', fontsize=12, fontweight='bold')
ax2.set_title('累计准确率走势对比', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

# 3. 分段准确率对比
ax3 = axes[1, 0]
segments = ['前10期\n(266-275)', '中10期\n(276-285)', '后10期\n(286-295)', 
            '再10期\n(296-305)', '最近10期\n(306-315)']
original_segments = [
    original_correct.iloc[0:10].mean() * 100,
    original_correct.iloc[10:20].mean() * 100,
    original_correct.iloc[20:30].mean() * 100,
    original_correct.iloc[30:40].mean() * 100,
    original_correct.iloc[40:50].mean() * 100,
]
improved_segments = [
    improved_correct.iloc[0:10].mean() * 100,
    improved_correct.iloc[10:20].mean() * 100,
    improved_correct.iloc[20:30].mean() * 100,
    improved_correct.iloc[30:40].mean() * 100,
    improved_correct.iloc[40:50].mean() * 100,
]
x = range(len(segments))
width = 0.35
bars1 = ax3.bar([i - width/2 for i in x], original_segments, width, label='原模型', 
                color='#3498db', alpha=0.7, edgecolor='black')
bars2 = ax3.bar([i + width/2 for i in x], improved_segments, width, label='改进模型', 
                color='#e74c3c', alpha=0.7, edgecolor='black')
ax3.axhline(y=50, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
ax3.set_xlabel('时间段', fontsize=12, fontweight='bold')
ax3.set_title('分段准确率对比 (每10期)', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(segments, fontsize=9)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 100)

# 4. 奇数vs偶数预测准确率
ax4 = axes[1, 1]
categories = ['奇数预测', '偶数预测']
original_odd_acc = original_correct[original_predicted == '奇数'].mean() * 100
original_even_acc = original_correct[original_predicted == '偶数'].mean() * 100
improved_odd_acc = improved_correct[improved_predicted == '奇数'].mean() * 100
improved_even_acc = improved_correct[improved_predicted == '偶数'].mean() * 100

original_accs = [original_odd_acc, original_even_acc]
improved_accs = [improved_odd_acc, improved_even_acc]
x = range(len(categories))
bars1 = ax4.bar([i - width/2 for i in x], original_accs, width, label='原模型', 
                color='#3498db', alpha=0.7, edgecolor='black')
bars2 = ax4.bar([i + width/2 for i in x], improved_accs, width, label='改进模型', 
                color='#e74c3c', alpha=0.7, edgecolor='black')
ax4.axhline(y=50, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax4.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
ax4.set_title('奇数vs偶数预测准确率', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(categories, fontsize=11)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim(0, 100)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('奇偶预测模型对比图.png', dpi=300, bbox_inches='tight')
print("图表已保存: 奇偶预测模型对比图.png")

# 打印详细统计
print("\n" + "="*80)
print("详细统计对比")
print("="*80)
print(f"\n原模型 (Gradient Boosting, 28特征):")
print(f"  总体准确率: {original_correct.sum()}/{len(original_correct)} = {original_correct.mean()*100:.2f}%")
print(f"  奇数预测: {original_correct[original_predicted=='奇数'].sum()}/{(original_predicted=='奇数').sum()} = {original_odd_acc:.2f}%")
print(f"  偶数预测: {original_correct[original_predicted=='偶数'].sum()}/{(original_predicted=='偶数').sum()} = {original_even_acc:.2f}%")

print(f"\n改进模型 (Ensemble Voting, 72特征):")
print(f"  总体准确率: {improved_correct.sum()}/{len(improved_correct)} = {improved_correct.mean()*100:.2f}%")
print(f"  奇数预测: {improved_correct[improved_predicted=='奇数'].sum()}/{(improved_predicted=='奇数').sum()} = {improved_odd_acc:.2f}%")
print(f"  偶数预测: {improved_correct[improved_predicted=='偶数'].sum()}/{(improved_predicted=='偶数').sum()} = {improved_even_acc:.2f}%")

print(f"\n结论:")
print(f"  整体准确率变化: {improved_correct.mean()*100 - original_correct.mean()*100:+.2f}个百分点")
print(f"  特征数量增加: {72-28} (+{(72-28)/28*100:.1f}%)")
print(f"  模型复杂度: 从单一模型升级到5模型集成")
if abs(improved_correct.mean()*100 - original_correct.mean()*100) < 1:
    print(f"  状态: ⚠️ 整体性能持平，但在最近10期都达到60%")
elif improved_correct.mean()*100 > original_correct.mean()*100:
    print(f"  状态: ✅ 改进模型表现更好")
else:
    print(f"  状态: ❌ 改进模型表现不如原模型")
print("="*80)
