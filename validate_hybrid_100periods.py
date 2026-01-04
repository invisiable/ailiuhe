"""
验证固化混合策略模型 - 最近100期成功率
验证规则：使用当期数据预测下一期，与实际结果比对
"""

import pandas as pd
import numpy as np
from collections import Counter
from final_hybrid_predictor import FinalHybridPredictor
from datetime import datetime


def validate_hybrid_model_100periods(csv_file='data/lucky_numbers.csv'):
    """验证混合模型最近100期的预测成功率"""
    
    print("=" * 80)
    print("固化混合策略模型 - 最近100期验证")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    total_records = len(df)
    
    # 确保有足够的数据进行验证
    if total_records < 101:
        print(f"错误：数据不足100期（当前只有{total_records}期）")
        return
    
    print(f"\n数据信息:")
    print(f"  总记录数: {total_records}")
    print(f"  验证期数: 100期")
    print(f"  验证范围: 第{total_records-100+1}期 到 第{total_records}期")
    
    # 创建预测器实例
    predictor = FinalHybridPredictor()
    
    # 统计结果
    results = {
        'top5': [],
        'top10': [],
        'top15': [],
        'top20': []
    }
    
    details = []
    
    print(f"\n{'='*80}")
    print(f"开始验证...")
    print(f"{'='*80}\n")
    
    # 对最近100期进行验证
    for i in range(100):
        # 使用前N期数据预测第N+1期
        current_idx = total_records - 100 + i
        
        # 获取当期之前的所有数据（包括当期）
        train_data = df.iloc[:current_idx + 1]
        
        # 下一期的实际数字
        if current_idx + 1 < total_records:
            next_actual = int(df.iloc[current_idx + 1]['number'])
            next_date = df.iloc[current_idx + 1]['date']
            current_date = df.iloc[current_idx]['date']
        else:
            break
        
        # 使用训练数据进行预测
        numbers = train_data['number'].values
        elements = train_data['element'].values
        
        # 策略A：全部历史数据（稳定覆盖）
        top15_a = predictor._predict_strategy_a(numbers)
        
        # 策略B：最近10期数据（精准预测）
        top15_b = predictor._predict_strategy_b(numbers, elements)
        
        # 混合策略：TOP1-5使用策略B，其余使用策略A
        hybrid_top20 = []
        
        # 先添加策略B的前5个
        for num in top15_b[:5]:
            if num not in hybrid_top20:
                hybrid_top20.append(num)
        
        # 再从策略A中补充到20个
        for num in top15_a:
            if num not in hybrid_top20:
                hybrid_top20.append(num)
            if len(hybrid_top20) >= 20:
                break
        
        top20 = hybrid_top20[:20]
        top15 = top20[:15]
        top10 = top15[:10]
        top5 = top10[:5]
        
        # 检查是否命中
        hit_top5 = next_actual in top5
        hit_top10 = next_actual in top10
        hit_top15 = next_actual in top15
        hit_top20 = next_actual in top20
        
        results['top5'].append(hit_top5)
        results['top10'].append(hit_top10)
        results['top15'].append(hit_top15)
        results['top20'].append(hit_top20)
        
        # 记录详细信息
        rank = None
        if hit_top20:
            rank = top20.index(next_actual) + 1
        
        hit_status = "✓" if hit_top20 else "✗"
        
        detail = {
            '期数': current_idx + 2,  # 被预测的期数
            '日期': next_date,
            '实际号码': next_actual,
            '命中': hit_status,
            'TOP5': '✓' if hit_top5 else '',
            'TOP10': '✓' if hit_top10 else '',
            'TOP15': '✓' if hit_top15 else '',
            'TOP20': '✓' if hit_top20 else '',
            '排名': rank if rank else '-',
            '预测列表': str(top20)
        }
        details.append(detail)
        
        # 每20期输出一次进度
        if (i + 1) % 20 == 0:
            current_top15_rate = sum(results['top15']) / len(results['top15']) * 100
            print(f"已验证 {i+1}/100 期，当前TOP15成功率: {current_top15_rate:.1f}%")
    
    # 计算成功率
    total_tests = len(results['top5'])
    
    print(f"\n{'='*80}")
    print("验证结果统计")
    print(f"{'='*80}\n")
    
    print(f"总验证期数: {total_tests}")
    print(f"\n成功率统计:")
    print(f"  TOP 5  成功率: {sum(results['top5'])/total_tests*100:.2f}% ({sum(results['top5'])}/{total_tests})")
    print(f"  TOP 10 成功率: {sum(results['top10'])/total_tests*100:.2f}% ({sum(results['top10'])}/{total_tests})")
    print(f"  TOP 15 成功率: {sum(results['top15'])/total_tests*100:.2f}% ({sum(results['top15'])}/{total_tests})")
    print(f"  TOP 20 成功率: {sum(results['top20'])/total_tests*100:.2f}% ({sum(results['top20'])}/{total_tests})")
    
    # 分段统计（每25期一个区间）
    print(f"\n{'='*80}")
    print("分段成功率分析（每25期）")
    print(f"{'='*80}\n")
    
    segments = [
        ('第1-25期', 0, 25),
        ('第26-50期', 25, 50),
        ('第51-75期', 50, 75),
        ('第76-100期', 75, 100)
    ]
    
    for seg_name, start, end in segments:
        seg_top5 = results['top5'][start:end]
        seg_top10 = results['top10'][start:end]
        seg_top15 = results['top15'][start:end]
        seg_top20 = results['top20'][start:end]
        
        print(f"{seg_name}:")
        print(f"  TOP 5:  {sum(seg_top5)/len(seg_top5)*100:.1f}% ({sum(seg_top5)}/{len(seg_top5)})")
        print(f"  TOP 10: {sum(seg_top10)/len(seg_top10)*100:.1f}% ({sum(seg_top10)}/{len(seg_top10)})")
        print(f"  TOP 15: {sum(seg_top15)/len(seg_top15)*100:.1f}% ({sum(seg_top15)}/{len(seg_top15)})")
        print(f"  TOP 20: {sum(seg_top20)/len(seg_top20)*100:.1f}% ({sum(seg_top20)}/{len(seg_top20)})")
        print()
    
    # 保存详细结果到CSV
    df_details = pd.DataFrame(details)
    output_file = 'hybrid_validation_100periods.csv'
    df_details.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"详细验证结果已保存到: {output_file}")
    
    # 显示最近10期的详细情况
    print(f"\n{'='*80}")
    print("最近10期详细情况")
    print(f"{'='*80}\n")
    
    print(f"{'期数':<8} {'日期':<12} {'实际':<6} {'命中':<6} {'TOP5':<6} {'TOP10':<7} {'TOP15':<7} {'TOP20':<7} {'排名':<6}")
    print("-" * 80)
    for detail in details[-10:]:
        print(f"{detail['期数']:<8} {detail['日期']:<12} {detail['实际号码']:<6} "
              f"{detail['命中']:<6} {detail['TOP5']:<6} {detail['TOP10']:<7} "
              f"{detail['TOP15']:<7} {detail['TOP20']:<7} {detail['排名']:<6}")
    
    print(f"\n{'='*80}")
    print("验证完成！")
    print(f"{'='*80}")
    
    return results, details


if __name__ == '__main__':
    results, details = validate_hybrid_model_100periods()
