#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
概率预测投注策略 - 300期详细验证
"""

import pandas as pd
from probability_betting_strategy import validate_probability_strategy
from precise_top15_predictor import PreciseTop15Predictor
from datetime import datetime


def main():
    # 设置输出编码为UTF-8
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 80)
    print("🔮 概率预测动态倍投策略 - 300期详细验证")
    print("=" * 80)
    print()
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"验证时间: {current_time}\n")
    
    # 加载数据
    file_path = 'data/lucky_numbers.csv'
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    print(f"✅ 数据加载完成: {len(df)}期")
    
    # 验证最近300期
    test_periods = min(300, len(df))
    print(f"验证期数: 最近{test_periods}期\n")
    
    numbers = df['number'].values
    animals = df['animal'].values
    elements = df['element'].values
    
    # 创建预测器
    predictor = PreciseTop15Predictor()
    
    print("=" * 80)
    print("第一步：执行300期回测验证")
    print("=" * 80)
    print()
    
    print("正在生成预测和执行策略分析...\n")
    
    # 执行验证
    result = validate_probability_strategy(
        predictor,
        numbers,
        animals,
        elements,
        test_periods=test_periods
    )
    
    print("✅ 验证完成！\n")
    
    # ========== 第二步：汇总统计 ==========
    print("=" * 80)
    print("第二步：汇总统计信息")
    print("=" * 80)
    print()
    
    print(f"【基础统计】")
    print(f"  验证期数: {result['total_periods']}期")
    print(f"  命中次数: {result['wins']}次")
    print(f"  未中次数: {result['losses']}次")
    print(f"  命中率: {result['hit_rate']*100:.2f}%")
    print()
    
    print(f"【财务统计】")
    print(f"  总投注: {result['total_bet']:.0f}元")
    print(f"  总收益: {result['total_win']:.0f}元")
    print(f"  净利润: {result['total_profit']:+.0f}元")
    print(f"  平均每期投注: {result['total_bet']/result['total_periods']:.2f}元")
    print(f"  平均每期收益: {result['total_profit']/result['total_periods']:+.2f}元")
    print(f"  投资回报率(ROI): {result['roi']:+.2f}%")
    print(f"  最大回撤: {result['max_drawdown']:.0f}元")
    print()
    
    # 预测准确性
    if result['prediction_accuracy']:
        acc = result['prediction_accuracy']
        print(f"【概率预测准确性】")
        print(f"  平均绝对误差(MAE): {acc['mae']:.4f}")
        print(f"  均方根误差(RMSE): {acc['rmse']:.4f}")
        print(f"  总预测次数: {acc['total_predictions']}")
        print()
        
        if 'calibration' in acc and acc['calibration']:
            print(f"【概率校准度分析】")
            print(f"  概率范围     | 预测次数 | 平均预测概率 | 实际命中率 |   偏差")
            print(f"  {'-'*70}")
            for cal in acc['calibration']:
                print(
                    f"  {cal['range']:>12} | {cal['count']:>8} | "
                    f"{cal['avg_predicted']:>12.1%} | {cal['avg_actual']:>10.1%} | "
                    f"{cal['bias']:>+7.1%}"
                )
            print()
    
    # 倍投分析
    history = result['history']
    
    # 统计不同倍数的使用情况
    mult_stats = {}
    for h in history:
        mult = round(h['multiplier'], 1)
        if mult not in mult_stats:
            mult_stats[mult] = {'count': 0, 'hits': 0, 'bet': 0, 'profit': 0}
        mult_stats[mult]['count'] += 1
        if h['hit']:
            mult_stats[mult]['hits'] += 1
        mult_stats[mult]['bet'] += h['bet']
        mult_stats[mult]['profit'] += h['profit']
    
    print(f"【倍投分布统计】")
    print(f"  倍数  | 使用次数 | 命中次数 | 命中率  |  总投注  |   净利润")
    print(f"  {'-'*70}")
    for mult in sorted(mult_stats.keys()):
        stat = mult_stats[mult]
        hit_rate = stat['hits'] / stat['count'] if stat['count'] > 0 else 0
        print(
            f"  {mult:>4.1f}x | {stat['count']:>8} | {stat['hits']:>8} | "
            f"{hit_rate:>6.1%} | {stat['bet']:>8.0f}元 | {stat['profit']:>+9.0f}元"
        )
    print()
    
    # 概率区间统计
    prob_ranges = {
        '0-20%': (0, 0.20),
        '20-30%': (0.20, 0.30),
        '30-40%': (0.30, 0.40),
        '40-50%': (0.40, 0.50),
        '50%+': (0.50, 1.0)
    }
    
    prob_stats = {k: {'count': 0, 'hits': 0, 'bet': 0, 'profit': 0} for k in prob_ranges.keys()}
    
    for h in history:
        prob = h['predicted_prob']
        for range_name, (low, high) in prob_ranges.items():
            if low <= prob < high:
                prob_stats[range_name]['count'] += 1
                if h['hit']:
                    prob_stats[range_name]['hits'] += 1
                prob_stats[range_name]['bet'] += h['bet']
                prob_stats[range_name]['profit'] += h['profit']
                break
    
    print(f"【预测概率区间分析】")
    print(f"  概率区间  | 预测次数 | 命中次数 | 命中率  |  总投注  |   净利润")
    print(f"  {'-'*70}")
    for range_name in prob_ranges.keys():
        stat = prob_stats[range_name]
        if stat['count'] > 0:
            hit_rate = stat['hits'] / stat['count']
            print(
                f"  {range_name:>9} | {stat['count']:>8} | {stat['hits']:>8} | "
                f"{hit_rate:>6.1%} | {stat['bet']:>8.0f}元 | {stat['profit']:>+9.0f}元"
            )
    print()
    
    # 连胜连亏统计
    max_win_streak = 0
    max_loss_streak = 0
    current_win = 0
    current_loss = 0
    
    for h in history:
        if h['hit']:
            current_win += 1
            current_loss = 0
            max_win_streak = max(max_win_streak, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss_streak = max(max_loss_streak, current_loss)
    
    print(f"【连续性统计】")
    print(f"  最长连胜: {max_win_streak}期")
    print(f"  最长连亏: {max_loss_streak}期")
    print()
    
    # 累计收益曲线关键点
    min_profit = min(h['profit'] for h in history)
    max_profit = max(h['profit'] for h in history)
    min_period = next(i for i, h in enumerate(history, 1) if h['profit'] == min_profit)
    max_period = next(i for i, h in enumerate(history, 1) if h['profit'] == max_profit)
    
    # 计算累计收益
    cumulative = 0
    min_cumulative = 0
    max_cumulative = 0
    min_cum_period = 0
    max_cum_period = 0
    
    for i, h in enumerate(history, 1):
        cumulative += h['profit']
        if cumulative < min_cumulative:
            min_cumulative = cumulative
            min_cum_period = i
        if cumulative > max_cumulative:
            max_cumulative = cumulative
            max_cum_period = i
    
    print(f"【收益曲线关键点】")
    print(f"  单期最大亏损: {min_profit:.0f}元（第{min_period}期）")
    print(f"  单期最大盈利: {max_profit:.0f}元（第{max_period}期）")
    print(f"  累计最低点: {min_cumulative:+.0f}元（第{min_cum_period}期）")
    print(f"  累计最高点: {max_cumulative:+.0f}元（第{max_cum_period}期）")
    print(f"  最终累计: {result['total_profit']:+.0f}元（第{test_periods}期）")
    print()
    
    # ========== 第三步：详细投注历史 ==========
    print("=" * 80)
    print("第三步：300期详细投注历史")
    print("=" * 80)
    print()
    
    print(f"{'期数':<6} {'日期':<12} {'实际':<6} {'预测概率':<10} {'倍数':<8} "
          f"{'投注':<8} {'结果':<6} {'单期盈亏':<10} {'累计盈亏':<10}")
    print("-" * 90)
    
    cumulative_profit = 0
    for i, h in enumerate(history, 1):
        cumulative_profit += h['profit']
        
        # 获取日期（如果有）
        period_idx = len(df) - test_periods + i - 1
        date = df.iloc[period_idx]['date'] if period_idx < len(df) else 'N/A'
        
        actual = h['actual']
        predicted_prob = h['predicted_prob']
        multiplier = h['multiplier']
        bet = h['bet']
        hit_mark = '✓命中' if h['hit'] else '✗未中'
        profit = h['profit']
        
        # 每20期加一个分隔线
        if i % 20 == 0 and i < len(history):
            print("-" * 90)
        
        print(
            f"{i:<6} {str(date):<12} {actual:<6} {predicted_prob:>8.1%}  {multiplier:>6.2f}x  "
            f"{bet:>6.0f}元  {hit_mark:<6} {profit:>+9.0f}元  {cumulative_profit:>+9.0f}元"
        )
    
    print("-" * 90)
    print()
    
    # ========== 第四步：月度分析 ==========
    print("=" * 80)
    print("第四步：月度统计分析")
    print("=" * 80)
    print()
    
    # 按月份统计
    from collections import defaultdict
    import re
    
    monthly_stats = defaultdict(lambda: {
        'periods': 0, 'hits': 0, 'total_bet': 0, 'total_profit': 0,
        'dates': []
    })
    
    for i, h in enumerate(history, 1):
        period_idx = len(df) - test_periods + i - 1
        if period_idx < len(df):
            date_str = str(df.iloc[period_idx]['date'])
            # 解析日期 YYYY/M/D
            date_match = re.match(r'(\d{4})/(\d{1,2})/\d{1,2}', date_str)
            if date_match:
                year = int(date_match.group(1))
                month = int(date_match.group(2))
                year_month = f"{year}年{month}月"
                
                monthly_stats[year_month]['periods'] += 1
                if h['hit']:
                    monthly_stats[year_month]['hits'] += 1
                monthly_stats[year_month]['total_bet'] += h['bet']
                monthly_stats[year_month]['total_profit'] += h['profit']
                monthly_stats[year_month]['year'] = year
                monthly_stats[year_month]['month'] = month
    
    if monthly_stats:
        print(f"{'月份':<12} | {'期数':<6} | {'命中':<6} | {'命中率':<8} | {'总投注':<10} | {'净利润':<10} | {'ROI':<8}")
        print("-" * 90)
        
        # 按时间排序
        sorted_months = sorted(monthly_stats.items(), 
                              key=lambda x: (x[1].get('year', 0), x[1].get('month', 0)))
        
        for month, stat in sorted_months:
            hit_rate = stat['hits'] / stat['periods'] if stat['periods'] > 0 else 0
            roi = (stat['total_profit'] / stat['total_bet'] * 100) if stat['total_bet'] > 0 else 0
            
            print(
                f"{month:<12} | {stat['periods']:<6} | {stat['hits']:<6} | "
                f"{hit_rate:>6.1%}  | {stat['total_bet']:>9.0f}元 | "
                f"{stat['total_profit']:>+9.0f}元 | {roi:>+6.1f}%"
            )
        print()
    
    # ========== 第五步：策略建议 ==========
    print("=" * 80)
    print("第五步：策略评估与建议")
    print("=" * 80)
    print()
    
    print(f"【整体表现评估】")
    
    # ROI评级
    roi = result['roi']
    if roi >= 20:
        roi_rating = "⭐⭐⭐⭐⭐ 优秀"
    elif roi >= 10:
        roi_rating = "⭐⭐⭐⭐ 良好"
    elif roi >= 5:
        roi_rating = "⭐⭐⭐ 一般"
    elif roi >= 0:
        roi_rating = "⭐⭐ 偏低"
    else:
        roi_rating = "⭐ 需改进"
    
    print(f"  ROI表现: {roi:+.2f}% - {roi_rating}")
    
    # 回撤评级
    drawdown = result['max_drawdown']
    if drawdown < 200:
        dd_rating = "⭐⭐⭐⭐⭐ 优秀"
    elif drawdown < 400:
        dd_rating = "⭐⭐⭐⭐ 良好"
    elif drawdown < 600:
        dd_rating = "⭐⭐⭐ 一般"
    elif drawdown < 800:
        dd_rating = "⭐⭐ 偏高"
    else:
        dd_rating = "⭐ 需改进"
    
    print(f"  风险控制: 回撤{drawdown:.0f}元 - {dd_rating}")
    
    # 预测准确性评级
    if result['prediction_accuracy']:
        mae = result['prediction_accuracy']['mae']
        if mae < 0.4:
            pred_rating = "⭐⭐⭐⭐⭐ 优秀"
        elif mae < 0.45:
            pred_rating = "⭐⭐⭐⭐ 良好"
        elif mae < 0.5:
            pred_rating = "⭐⭐⭐ 一般"
        else:
            pred_rating = "⭐⭐ 需改进"
        
        print(f"  预测准确性: MAE {mae:.4f} - {pred_rating}")
    
    print()
    
    print(f"【使用建议】")
    print(f"  建议资金: {drawdown * 3:.0f}元以上（3倍最大回撤）")
    print(f"  单期上限: 不超过总资金的30%")
    print(f"  止损建议: 累计亏损达到总资金50%时暂停")
    print(f"  适用周期: 中长期投注（≥100期）")
    print()
    
    # 对比固定投注
    fixed_bet_total = test_periods * 15
    fixed_wins = result['wins']
    fixed_reward = fixed_wins * 47
    fixed_profit = fixed_reward - fixed_bet_total
    fixed_roi = (fixed_profit / fixed_bet_total * 100) if fixed_bet_total > 0 else 0
    
    print(f"【对比固定投注】")
    print(f"  固定投注策略（每期15元）:")
    print(f"    总投注: {fixed_bet_total}元")
    print(f"    净利润: {fixed_profit:+.0f}元")
    print(f"    ROI: {fixed_roi:+.2f}%")
    print()
    print(f"  概率预测策略:")
    print(f"    总投注: {result['total_bet']:.0f}元")
    print(f"    净利润: {result['total_profit']:+.0f}元")
    print(f"    ROI: {result['roi']:+.2f}%")
    print()
    
    profit_diff = result['total_profit'] - fixed_profit
    roi_diff = result['roi'] - fixed_roi
    
    if profit_diff > 0:
        print(f"  ✅ 概率策略优势: 收益增加{profit_diff:+.0f}元, ROI提升{roi_diff:+.2f}%")
    else:
        print(f"  ⚠️  固定投注优势: 收益减少{abs(profit_diff):.0f}元, ROI降低{abs(roi_diff):.2f}%")
    print()
    
    # 保存详细数据到CSV
    print("=" * 80)
    print("保存详细数据")
    print("=" * 80)
    print()
    
    # 创建DataFrame
    detail_data = []
    cumulative_profit = 0
    for i, h in enumerate(history, 1):
        cumulative_profit += h['profit']
        period_idx = len(df) - test_periods + i - 1
        date = df.iloc[period_idx]['date'] if period_idx < len(df) else 'N/A'
        
        detail_data.append({
            '期数': i,
            '日期': date,
            '实际号码': h['actual'],
            'TOP15预测': str(h['predictions']),
            '预测概率': f"{h['predicted_prob']:.4f}",
            '倍数': f"{h['multiplier']:.2f}",
            '投注金额': h['bet'],
            '是否命中': '是' if h['hit'] else '否',
            '单期盈亏': h['profit'],
            '累计盈亏': cumulative_profit,
            '近期命中率': f"{h.get('recent_rate', 0):.4f}"
        })
    
    detail_df = pd.DataFrame(detail_data)
    csv_filename = f'probability_betting_300periods_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    detail_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    print(f"✅ 详细数据已保存到: {csv_filename}")
    print(f"   共{len(detail_data)}条记录")
    print()
    
    print("=" * 80)
    print("✅ 300期详细验证完成！")
    print("=" * 80)
    print()
    
    print(f"📊 核心数据总结:")
    print(f"   验证期数: {test_periods}期")
    print(f"   命中率: {result['hit_rate']*100:.2f}%")
    print(f"   ROI: {result['roi']:+.2f}%")
    print(f"   净利润: {result['total_profit']:+.0f}元")
    print(f"   最大回撤: {result['max_drawdown']:.0f}元")
    print(f"   预测MAE: {result['prediction_accuracy']['mae']:.4f}")
    print()


if __name__ == '__main__':
    main()
