#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
概率预测投注策略 - 300期验证结果查看器
"""

import pandas as pd
from probability_betting_strategy import validate_probability_strategy
from precise_top15_predictor import PreciseTop15Predictor


def main():
    print("=" * 80)
    print("概率预测动态倍投策略 - 300期详细验证")
    print("=" * 80)
    print()
    
    # 加载数据
    file_path = 'data/lucky_numbers.csv'
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    print(f"数据加载完成: {len(df)}期")
    
    # 验证最近300期
    test_periods = min(300, len(df))
    print(f"验证期数: {test_periods}期")
    print()
    
    numbers = df['number'].values
    animals = df['animal'].values
    elements = df['element'].values
    
    # 创建预测器
    predictor = PreciseTop15Predictor()
    
    print("正在执行300期回测...")
    
    # 执行验证
    result = validate_probability_strategy(
        predictor,
        numbers,
        animals,
        elements,
        test_periods=test_periods
    )
    
    print("验证完成!")
    print()
    
    # 输出汇总统计
    print("=" * 80)
    print("汇总统计")
    print("=" * 80)
    print()
    
    print(f"验证期数: {result['total_periods']}期")
    print(f"命中次数: {result['wins']}次")
    print(f"未中次数: {result['losses']}次")
    print(f"命中率: {result['hit_rate']*100:.2f}%")
    print()
    
    print(f"总投注: {result['total_bet']:.0f}元")
    print(f"总收益: {result['total_win']:.0f}元")
    print(f"净利润: {result['total_profit']:+.0f}元")
    print(f"ROI: {result['roi']:+.2f}%")
    print(f"最大回撤: {result['max_drawdown']:.0f}元")
    print()
    
    if result['prediction_accuracy']:
        acc = result['prediction_accuracy']
        print(f"预测MAE: {acc['mae']:.4f}")
        print(f"预测RMSE: {acc['rmse']:.4f}")
        print()
    
    # 保存详细数据
    print("=" * 80)
    print("保存详细数据到CSV")
    print("=" * 80)
    print()
    
    history = result['history']
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
            '预测概率': round(h['predicted_prob'], 4),
            '倍数': round(h['multiplier'], 2),
            '投注金额': h['bet'],
            '是否命中': 'Y' if h['hit'] else 'N',
            '单期盈亏': h['profit'],
            '累计盈亏': cumulative_profit
        })
    
    detail_df = pd.DataFrame(detail_data)
    csv_filename = 'probability_betting_300periods_detail.csv'
    detail_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    print(f"详细数据已保存到: {csv_filename}")
    print(f"共 {len(detail_data)} 条记录")
    print()
    
    # 显示前20期和后20期
    print("=" * 80)
    print("前20期详情")
    print("=" * 80)
    print()
    
    print(f"{'期数':<6} {'日期':<12} {'实际':<6} {'预测概率':<10} {'倍数':<8} {'投注':<8} {'结果':<6} {'盈亏':<10}")
    print("-" * 80)
    
    for i in range(min(20, len(detail_data))):
        row = detail_data[i]
        hit_mark = '命中' if row['是否命中'] == 'Y' else '未中'
        print(
            f"{row['期数']:<6} {str(row['日期']):<12} {row['实际号码']:<6} "
            f"{row['预测概率']:>8.1%}  {row['倍数']:>6.2f}x  {row['投注金额']:>6.0f}元  "
            f"{hit_mark:<6} {row['单期盈亏']:>+8.0f}元"
        )
    
    print()
    print("=" * 80)
    print("后20期详情")
    print("=" * 80)
    print()
    
    print(f"{'期数':<6} {'日期':<12} {'实际':<6} {'预测概率':<10} {'倍数':<8} {'投注':<8} {'结果':<6} {'盈亏':<10}")
    print("-" * 80)
    
    start_idx = max(0, len(detail_data) - 20)
    for i in range(start_idx, len(detail_data)):
        row = detail_data[i]
        hit_mark = '命中' if row['是否命中'] == 'Y' else '未中'
        print(
            f"{row['期数']:<6} {str(row['日期']):<12} {row['实际号码']:<6} "
            f"{row['预测概率']:>8.1%}  {row['倍数']:>6.2f}x  {row['投注金额']:>6.0f}元  "
            f"{hit_mark:<6} {row['单期盈亏']:>+8.0f}元"
        )
    
    print()
    print("=" * 80)
    print("验证完成")
    print("=" * 80)
    print()
    
    print(f"核心数据总结:")
    print(f"  验证期数: {test_periods}期")
    print(f"  命中率: {result['hit_rate']*100:.2f}%")
    print(f"  ROI: {result['roi']:+.2f}%")
    print(f"  净利润: {result['total_profit']:+.0f}元")
    print(f"  最大回撤: {result['max_drawdown']:.0f}元")
    if result['prediction_accuracy']:
        print(f"  预测MAE: {result['prediction_accuracy']['mae']:.4f}")
    print()
    print(f"详细的300期投注历史已保存到: {csv_filename}")
    print()


if __name__ == '__main__':
    main()
