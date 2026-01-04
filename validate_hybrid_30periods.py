"""
验证固化混合策略模型 - 最近30期成功率
验证规则：使用当期数据预测下一期，与实际结果比对
"""

import pandas as pd
import numpy as np
from collections import Counter
from final_hybrid_predictor import FinalHybridPredictor


def validate_hybrid_model_30periods(csv_file='data/lucky_numbers.csv'):
    """验证混合模型最近30期的预测成功率"""
    
    print("=" * 80)
    print("固化混合策略模型 - 最近30期验证")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    total_records = len(df)
    
    print(f"\n数据信息:")
    print(f"  总记录数: {total_records}")
    print(f"  验证期数: 30期")
    print(f"  验证范围: 第{total_records-30+1}期 到 第{total_records}期")
    
    # 创建预测器实例
    predictor = FinalHybridPredictor()
    
    # 统计结果
    results = {
        'top5': [],
        'top10': [],
        'top15': []
    }
    
    details = []
    
    print(f"\n{'='*80}")
    print(f"开始验证...")
    print(f"{'='*80}\n")
    
    # 对最近30期进行验证
    for i in range(30):
        # 使用前N期数据预测第N+1期
        current_idx = total_records - 30 + i
        
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
        hybrid_top15 = []
        
        # 先添加策略B的前5个
        for num in top15_b[:5]:
            if num not in hybrid_top15:
                hybrid_top15.append(num)
        
        # 再从策略A中补充到15个
        for num in top15_a:
            if num not in hybrid_top15:
                hybrid_top15.append(num)
            if len(hybrid_top15) >= 15:
                break
        
        top15 = hybrid_top15[:15]
        top10 = top15[:10]
        top5 = top15[:5]
        
        # 检查是否命中
        hit_top5 = next_actual in top5
        hit_top10 = next_actual in top10
        hit_top15 = next_actual in top15
        
        results['top5'].append(hit_top5)
        results['top10'].append(hit_top10)
        results['top15'].append(hit_top15)
        
        # 记录详细信息
        rank = None
        if hit_top15:
            rank = top15.index(next_actual) + 1
        
        detail = {
            'period': i + 1,
            'current_date': current_date,
            'predict_date': next_date,
            'actual': next_actual,
            'top5': top5,
            'top10': top10,
            'top15': top15,
            'hit_top5': hit_top5,
            'hit_top10': hit_top10,
            'hit_top15': hit_top15,
            'rank': rank
        }
        details.append(detail)
        
        # 实时显示
        status = ""
        if hit_top5:
            status = f"✅ TOP5命中 (#{rank})"
        elif hit_top10:
            status = f"✅ TOP10命中 (#{rank})"
        elif hit_top15:
            status = f"✅ TOP15命中 (#{rank})"
        else:
            status = "❌ 未命中"
        
        print(f"期数 {i+1:>2}/30 | {current_date} 预测 {next_date} | 实际: {next_actual:>2} | {status}")
    
    # 计算成功率
    print(f"\n{'='*80}")
    print("验证结果统计")
    print(f"{'='*80}\n")
    
    top5_success = sum(results['top5'])
    top10_success = sum(results['top10'])
    top15_success = sum(results['top15'])
    
    total = len(results['top5'])
    
    top5_rate = top5_success / total * 100
    top10_rate = top10_success / total * 100
    top15_rate = top15_success / total * 100
    
    print(f"验证期数: {total} 期")
    print(f"\n成功率统计:")
    print(f"  TOP 5  命中: {top5_success:>2}/{total} = {top5_rate:>5.1f}%")
    print(f"  TOP 10 命中: {top10_success:>2}/{total} = {top10_rate:>5.1f}%")
    print(f"  TOP 15 命中: {top15_success:>2}/{total} = {top15_rate:>5.1f}%")
    
    # 详细命中记录
    print(f"\n{'='*80}")
    print("详细命中记录")
    print(f"{'='*80}\n")
    
    hit_details = [d for d in details if d['hit_top15']]
    print(f"总命中次数: {len(hit_details)}/{total}")
    
    if hit_details:
        print(f"\n命中详情:")
        for d in hit_details:
            marker = "🏆" if d['rank'] == 1 else "⭐" if d['rank'] <= 5 else "✓"
            print(f"  {marker} 期数{d['period']:>2}: {d['current_date']} → {d['predict_date']} | "
                  f"预测命中 #{d['actual']} (第{d['rank']}名)")
    
    # 未命中记录
    miss_details = [d for d in details if not d['hit_top15']]
    if miss_details:
        print(f"\n未命中详情: ({len(miss_details)}次)")
        for d in miss_details:
            print(f"  ❌ 期数{d['period']:>2}: {d['current_date']} → {d['predict_date']} | "
                  f"实际 #{d['actual']} | TOP5: {d['top5']}")
    
    # 排名分布
    print(f"\n{'='*80}")
    print("命中排名分布")
    print(f"{'='*80}\n")
    
    rank_dist = {}
    for d in hit_details:
        rank = d['rank']
        rank_dist[rank] = rank_dist.get(rank, 0) + 1
    
    if rank_dist:
        for rank in sorted(rank_dist.keys()):
            count = rank_dist[rank]
            bar = '█' * count
            print(f"  第 {rank:>2} 名: {count:>2} 次 {bar}")
    
    print(f"\n{'='*80}")
    print("验证完成")
    print(f"{'='*80}")
    
    return {
        'total': total,
        'top5_success': top5_success,
        'top10_success': top10_success,
        'top15_success': top15_success,
        'top5_rate': top5_rate,
        'top10_rate': top10_rate,
        'top15_rate': top15_rate,
        'details': details
    }


if __name__ == '__main__':
    results = validate_hybrid_model_30periods()
