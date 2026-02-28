"""
对比TOP15投注策略 vs 精准TOP15投注策略
回测最近200期，使用斐波那契倍投方式
"""

import pandas as pd
import numpy as np
from top15_predictor import Top15Predictor
from precise_top15_predictor import PreciseTop15Predictor
from betting_strategy import BettingStrategy


def compare_top15_strategies(test_periods=200):
    """
    对比两种TOP15策略的收益率
    
    投注规则（两个策略相同）：
    - 每期购买：TOP15全部15个数字
    - 单注成本：15元（15个×1元）
    - 命中奖励：47元
    - 未中亏损：15元
    
    倍投方式：斐波那契（平衡策略）
    """
    
    print("="*100)
    print("TOP15投注策略 vs 精准TOP15投注策略 - 200期对比分析")
    print("="*100)
    print()
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_periods = len(df)
    
    if test_periods > total_periods - 20:
        test_periods = total_periods - 20
        print(f"调整测试期数为: {test_periods}期\n")
    
    start_idx = total_periods - test_periods
    
    print(f"数据加载完成: {total_periods}期")
    print(f"回测期数: {test_periods}期 (第{start_idx+1}期 到 第{total_periods}期)")
    print(f"最新日期: {df.iloc[-1]['date']}\n")
    
    print("="*100)
    print("投注规则说明（两个策略相同）")
    print("="*100)
    print("• 每期购买：TOP15全部15个数字")
    print("• 单注成本：15元（每个数字1元 × 15个）")
    print("• 命中奖励：47元（中1个数字）")
    print("• 未中亏损：15元")
    print("• 倍投策略：斐波那契数列（1,1,2,3,5,8,13...）")
    print("• 倍投规则：命中后重置为1倍，未命中增加倍数\n")
    
    # ==================== 策略1：TOP15投注策略 ====================
    print("="*100)
    print("策略1：TOP15投注策略（综合预测Top15）")
    print("="*100)
    print("预测器：Top15Predictor (60%成功率固化版本)")
    print("方法：综合5种统计方法\n")
    
    predictor1 = Top15Predictor()
    predictions1 = []
    actuals1 = []
    dates1 = []
    
    print("正在生成预测...")
    for i in range(start_idx, total_periods):
        # 使用i之前的数据
        train_data = df.iloc[:i]['number'].values
        
        # 获取TOP15预测
        analysis = predictor1.get_analysis(train_data)
        top15 = analysis['top15']
        predictions1.append(top15)
        
        # 实际结果
        actual = df.iloc[i]['number']
        actuals1.append(actual)
        dates1.append(df.iloc[i]['date'])
        
        if (i - start_idx + 1) % 50 == 0:
            print(f"  已处理 {i - start_idx + 1}/{test_periods} 期...")
    
    print(f"✅ 策略1预测完成: {len(predictions1)}期\n")
    
    # 计算命中率
    hit_rate1 = sum(1 for i in range(len(actuals1)) if actuals1[i] in predictions1[i]) / len(actuals1)
    print(f"实际命中率: {hit_rate1*100:.2f}%\n")
    
    # 斐波那契倍投回测
    betting1 = BettingStrategy(base_bet=15, win_reward=47, loss_penalty=15)
    result1 = betting1.simulate_strategy(predictions1, actuals1, 'fibonacci', hit_rate=hit_rate1)
    
    # ==================== 策略2：精准TOP15投注策略 ====================
    print("="*100)
    print("策略2：精准TOP15投注策略（优化版）")
    print("="*100)
    print("预测器：PreciseTop15Predictor（风险控制优化版）")
    print("特点：多窗口频率融合 + 历史错误规避 + 间隔优化\n")
    
    predictor2 = PreciseTop15Predictor()
    predictions2 = []
    actuals2 = []
    dates2 = []
    
    print("正在生成预测...")
    for i in range(start_idx, total_periods):
        # 使用i之前的数据
        train_data = df.iloc[:i]['number'].values
        
        # 获取TOP15预测
        top15 = predictor2.predict(train_data)
        predictions2.append(top15)
        
        # 实际结果
        actual = df.iloc[i]['number']
        actuals2.append(actual)
        dates2.append(df.iloc[i]['date'])
        
        # 更新性能跟踪
        hit = actual in top15
        predictor2.update_performance(top15, actual)
        
        if (i - start_idx + 1) % 50 == 0:
            print(f"  已处理 {i - start_idx + 1}/{test_periods} 期...")
    
    print(f"✅ 策略2预测完成: {len(predictions2)}期\n")
    
    # 计算命中率
    hit_rate2 = sum(1 for i in range(len(actuals2)) if actuals2[i] in predictions2[i]) / len(actuals2)
    print(f"实际命中率: {hit_rate2*100:.2f}%\n")
    
    # 斐波那契倍投回测
    betting2 = BettingStrategy(base_bet=15, win_reward=47, loss_penalty=15)
    result2 = betting2.simulate_strategy(predictions2, actuals2, 'fibonacci', hit_rate=hit_rate2)
    
    # ==================== 对比分析 ====================
    print("\n" + "="*100)
    print("📊 对比分析结果（斐波那契倍投，200期）")
    print("="*100)
    print()
    
    # 创建对比表格
    comparison_data = [
        ["指标", "策略1 (综合Top15)", "策略2 (精准Top15)", "差异"],
        ["-"*20, "-"*25, "-"*25, "-"*20],
    ]
    
    # 命中率对比
    hit_diff = (hit_rate2 - hit_rate1) * 100
    comparison_data.append([
        "命中率",
        f"{hit_rate1*100:.2f}%",
        f"{hit_rate2*100:.2f}%",
        f"{hit_diff:+.2f}%"
    ])
    
    # 命中次数
    comparison_data.append([
        "命中次数",
        f"{result1['wins']}/{test_periods}",
        f"{result2['wins']}/{test_periods}",
        f"{result2['wins'] - result1['wins']:+d}"
    ])
    
    # ROI对比
    roi_diff = result2['roi'] - result1['roi']
    comparison_data.append([
        "投资回报率(ROI)",
        f"{result1['roi']:+.2f}%",
        f"{result2['roi']:+.2f}%",
        f"{roi_diff:+.2f}%"
    ])
    
    # 总收益对比
    profit_diff = result2['total_profit'] - result1['total_profit']
    comparison_data.append([
        "总收益",
        f"{result1['total_profit']:+.2f}元",
        f"{result2['total_profit']:+.2f}元",
        f"{profit_diff:+.2f}元"
    ])
    
    # 总投注
    cost_diff = result2['total_cost'] - result1['total_cost']
    comparison_data.append([
        "总投注",
        f"{result1['total_cost']:.2f}元",
        f"{result2['total_cost']:.2f}元",
        f"{cost_diff:+.2f}元"
    ])
    
    # 平均每期收益
    avg_profit_diff = result2['avg_profit_per_period'] - result1['avg_profit_per_period']
    comparison_data.append([
        "平均每期收益",
        f"{result1['avg_profit_per_period']:+.2f}元",
        f"{result2['avg_profit_per_period']:+.2f}元",
        f"{avg_profit_diff:+.2f}元"
    ])
    
    # 最大连续亏损
    loss_diff = result2['max_consecutive_losses'] - result1['max_consecutive_losses']
    comparison_data.append([
        "最大连续不中",
        f"{result1['max_consecutive_losses']}期",
        f"{result2['max_consecutive_losses']}期",
        f"{loss_diff:+d}期"
    ])
    
    # 最大回撤
    drawdown_diff = result2['max_drawdown'] - result1['max_drawdown']
    comparison_data.append([
        "最大回撤",
        f"{result1['max_drawdown']:.2f}元",
        f"{result2['max_drawdown']:.2f}元",
        f"{drawdown_diff:+.2f}元"
    ])
    
    # 最终余额
    balance_diff = result2['final_balance'] - result1['final_balance']
    comparison_data.append([
        "最终余额",
        f"{result1['final_balance']:+.2f}元",
        f"{result2['final_balance']:+.2f}元",
        f"{balance_diff:+.2f}元"
    ])
    
    # 打印对比表格
    for row in comparison_data:
        print(f"{row[0]:25s} {row[1]:30s} {row[2]:30s} {row[3]:25s}")
    
    print()
    print("="*100)
    print("💡 结论")
    print("="*100)
    
    # 判断哪个策略更好
    if result2['roi'] > result1['roi']:
        better_strategy = "精准TOP15投注策略（策略2）"
        roi_improvement = result2['roi'] - result1['roi']
        profit_improvement = result2['total_profit'] - result1['total_profit']
        print(f"✅ 策略2（精准TOP15）表现更优！")
        print(f"   • ROI优势: {roi_improvement:+.2f}%")
        print(f"   • 收益优势: {profit_improvement:+.2f}元")
    elif result1['roi'] > result2['roi']:
        better_strategy = "综合TOP15投注策略（策略1）"
        roi_improvement = result1['roi'] - result2['roi']
        profit_improvement = result1['total_profit'] - result2['total_profit']
        print(f"✅ 策略1（综合TOP15）表现更优！")
        print(f"   • ROI优势: {roi_improvement:+.2f}%")
        print(f"   • 收益优势: {profit_improvement:+.2f}元")
    else:
        better_strategy = "两个策略表现相当"
        print(f"⚖️ 两个策略表现相当")
    
    # 命中率分析
    print()
    if hit_rate2 > hit_rate1:
        print(f"📈 命中率：策略2更高 ({hit_rate2*100:.2f}% vs {hit_rate1*100:.2f}%)")
    elif hit_rate1 > hit_rate2:
        print(f"📈 命中率：策略1更高 ({hit_rate1*100:.2f}% vs {hit_rate2*100:.2f}%)")
    else:
        print(f"📊 命中率：两者相同 ({hit_rate1*100:.2f}%)")
    
    # 风险控制分析
    print()
    if result2['max_consecutive_losses'] < result1['max_consecutive_losses']:
        print(f"🛡️ 风险控制：策略2更优（最大连不中 {result2['max_consecutive_losses']}期 vs {result1['max_consecutive_losses']}期）")
    elif result1['max_consecutive_losses'] < result2['max_consecutive_losses']:
        print(f"🛡️ 风险控制：策略1更优（最大连不中 {result1['max_consecutive_losses']}期 vs {result2['max_consecutive_losses']}期）")
    else:
        print(f"🛡️ 风险控制：两者相同（最大连不中 {result1['max_consecutive_losses']}期）")
    
    print()
    print("="*100)
    print("📝 说明")
    print("="*100)
    print("• TOP15投注：每期购买15个数字，单注成本15元，命中奖励47元")
    print("• 斐波那契倍投：命中后恢复1倍，未命中按斐波那契数列递增倍数")
    print("• 回测期数：最近200期")
    print("• 数据来源：data/lucky_numbers.csv")
    print()
    
    return {
        'strategy1': {
            'name': '综合TOP15投注策略',
            'hit_rate': hit_rate1,
            'result': result1
        },
        'strategy2': {
            'name': '精准TOP15投注策略',
            'hit_rate': hit_rate2,
            'result': result2
        },
        'better_strategy': better_strategy
    }


if __name__ == '__main__':
    result = compare_top15_strategies(test_periods=200)
    
    print("\n" + "="*100)
    print("🏆 最终推荐")
    print("="*100)
    print(f"推荐策略: {result['better_strategy']}")
    print()
    
    # 保存详细对比结果
    print("💾 详细结果已计算完成，可通过返回值获取")
