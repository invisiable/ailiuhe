"""
生肖TOP4投注策略 - 可视化对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

def visualize_comparison():
    """可视化策略对比"""
    
    # 读取数据
    df_old = pd.read_csv('zodiac_top4_backtest_300periods.csv', encoding='utf-8-sig').tail(50)
    df_conservative = pd.read_csv('zodiac_top4_optimized_conservative_50periods.csv', encoding='utf-8-sig')
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('生肖TOP4投注策略对比分析', fontsize=16, fontweight='bold')
    
    # 1. 累计盈利曲线
    ax1 = axes[0, 0]
    old_balance = df_old['cumulative_profit'].values
    conservative_balance = df_conservative['balance'].values
    
    ax1.plot(range(1, len(old_balance)+1), old_balance, 'r-', label='原策略（马丁格尔10倍）', linewidth=2)
    ax1.plot(range(1, len(conservative_balance)+1), conservative_balance, 'g-', label='保守策略（固定1倍）', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('期数')
    ax1.set_ylabel('累计盈利（元）')
    ax1.set_title('累计盈利曲线对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 投注额分布
    ax2 = axes[0, 1]
    old_bets = df_old['bet_amount'].value_counts().sort_index()
    conservative_bets = df_conservative[df_conservative['bet_amount'] > 0]['bet_amount'].value_counts().sort_index()
    
    x = range(len(old_bets))
    width = 0.35
    ax2.bar([i-width/2 for i in x], old_bets.values, width, label='原策略', color='red', alpha=0.7)
    ax2.bar([i+width/2 for i in x], [conservative_bets.get(k, 0) for k in old_bets.index], 
            width, label='保守策略', color='green', alpha=0.7)
    ax2.set_xlabel('投注额（元）')
    ax2.set_ylabel('次数')
    ax2.set_title('投注额分布对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(old_bets.index)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 关键指标对比
    ax3 = axes[1, 0]
    
    # 计算指标
    old_stats = {
        '命中率': (df_old['is_hit'] == '是').sum() / len(df_old) * 100,
        '投资回报': df_old['profit'].sum() / df_old['bet_amount'].sum() * 100,
        '单期最高投注': df_old['bet_amount'].max()
    }
    
    conservative_bet_records = df_conservative[df_conservative['bet_amount'] > 0]
    conservative_stats = {
        '命中率': (conservative_bet_records['is_hit'] == '✅').sum() / len(conservative_bet_records) * 100,
        '投资回报': conservative_bet_records['profit'].sum() / conservative_bet_records['bet_amount'].sum() * 100,
        '单期最高投注': conservative_bet_records['bet_amount'].max()
    }
    
    metrics = list(old_stats.keys())
    x = range(len(metrics))
    width = 0.35
    
    old_values = [old_stats[m] for m in metrics]
    conservative_values = [conservative_stats[m] for m in metrics]
    
    ax3.bar([i-width/2 for i in x], old_values, width, label='原策略', color='red', alpha=0.7)
    ax3.bar([i+width/2 for i in x], conservative_values, width, label='保守策略', color='green', alpha=0.7)
    ax3.set_ylabel('数值')
    ax3.set_title('关键指标对比')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (old_val, cons_val) in enumerate(zip(old_values, conservative_values)):
        ax3.text(i-width/2, old_val, f'{old_val:.1f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i+width/2, cons_val, f'{cons_val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 4. 盈利分布
    ax4 = axes[1, 1]
    
    old_profit_dist = df_old['profit'].value_counts().sort_index()
    conservative_profit_dist = df_conservative[df_conservative['bet_amount'] > 0]['profit'].value_counts().sort_index()
    
    ax4.hist([df_old['profit'].values, 
              conservative_bet_records['profit'].values],
             bins=20, label=['原策略', '保守策略'], 
             color=['red', 'green'], alpha=0.6)
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_xlabel('单期盈亏（元）')
    ax4.set_ylabel('次数')
    ax4.set_title('单期盈亏分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('zodiac_top4_策略对比图表.png', dpi=300, bbox_inches='tight')
    print("✅ 图表已保存：zodiac_top4_策略对比图表.png")
    plt.close()
    
    # 打印统计摘要
    print("\n" + "="*80)
    print("策略对比统计摘要")
    print("="*80)
    
    print(f"\n【原策略 - 马丁格尔10倍】")
    print(f"  投注期数: {len(df_old)}期")
    print(f"  命中次数: {(df_old['is_hit'] == '是').sum()}次")
    print(f"  命中率: {old_stats['命中率']:.1f}%")
    print(f"  投注总额: {df_old['bet_amount'].sum():.0f}元")
    print(f"  累计盈利: {df_old['profit'].sum():.0f}元")
    print(f"  投资回报率: {old_stats['投资回报']:.1f}%")
    print(f"  单期最高投注: {old_stats['单期最高投注']:.0f}元")
    print(f"  最长连续不中: {df_old['consecutive_losses'].max()}期")
    
    print(f"\n【保守策略 - 固定1倍+止损】")
    print(f"  投注期数: {len(conservative_bet_records)}期（跳过{len(df_conservative)-len(conservative_bet_records)}期）")
    print(f"  命中次数: {(conservative_bet_records['is_hit'] == '✅').sum()}次")
    print(f"  命中率: {conservative_stats['命中率']:.1f}%")
    print(f"  投注总额: {conservative_bet_records['bet_amount'].sum():.0f}元")
    print(f"  累计盈利: {conservative_bet_records['profit'].sum():.0f}元")
    print(f"  投资回报率: {conservative_stats['投资回报']:.1f}%")
    print(f"  单期最高投注: {conservative_stats['单期最高投注']:.0f}元")
    
    print(f"\n【改进效果】")
    print(f"  ✅ 投资回报率提升: {conservative_stats['投资回报'] - old_stats['投资回报']:.1f}%")
    print(f"  ✅ 单期最高投注降低: {old_stats['单期最高投注'] - conservative_stats['单期最高投注']:.0f}元 (-{(1-conservative_stats['单期最高投注']/old_stats['单期最高投注'])*100:.0f}%)")
    print(f"  ✅ 风险等级: 从 🔴极高 降至 🟢低")
    print(f"  ⚠️ 命中率变化: {conservative_stats['命中率'] - old_stats['命中率']:.1f}% (因止损机制跳过部分期数)")
    
if __name__ == '__main__':
    visualize_comparison()
