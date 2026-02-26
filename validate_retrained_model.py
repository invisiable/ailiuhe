"""
验证重训练模型在最近30期的效果
对比旧模型(EnsembleZodiacPredictor)和新模型(RetrainedZodiacPredictor)
"""

import pandas as pd
import numpy as np
from retrained_zodiac_predictor import RetrainedZodiacPredictor
from ensemble_zodiac_predictor import EnsembleZodiacPredictor


def validate_model(predictor, csv_file, start_period, end_period, model_name):
    """验证模型在指定期数的命中率"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    results = []
    hits = 0
    
    print(f"\n{'='*80}")
    print(f"{model_name} - 第{start_period}-{end_period}期验证")
    print(f"{'='*80}\n")
    
    for period in range(start_period, end_period + 1):
        # 使用period之前的数据预测
        history_df = df.iloc[:period-1]
        animals = [str(a).strip() for a in history_df['animal'].values]
        
        # 预测
        pred_result = predictor.predict_from_history(animals, top_n=4, debug=False)
        top4_pred = pred_result['top4']
        
        # 获取实际结果
        actual_row = df.iloc[period-1]
        actual_zodiac = str(actual_row['animal']).strip()
        actual_date = actual_row['date']
        actual_number = actual_row['number']
        
        # 判断命中
        is_hit = actual_zodiac in top4_pred
        if is_hit:
            hits += 1
        
        results.append({
            'period': period,
            'date': actual_date,
            'actual_number': actual_number,
            'actual_zodiac': actual_zodiac,
            'predicted_top4': ', '.join(top4_pred),
            'is_hit': is_hit
        })
        
        # 打印
        result_str = '✅' if is_hit else '❌'
        print(f"{period:>3}  {actual_date:>10}  {actual_number:>2}  {actual_zodiac:>2}  "
              f"{', '.join(top4_pred):20}  {result_str}")
    
    hit_rate = hits / len(results) * 100
    
    print(f"\n{'='*80}")
    print(f"【验证结果】")
    print(f"命中: {hits}/{len(results)}")
    print(f"命中率: {hit_rate:.1f}%")
    print(f"{'='*80}\n")
    
    return {
        'model_name': model_name,
        'results': results,
        'hits': hits,
        'total': len(results),
        'hit_rate': hit_rate
    }


def compare_models(csv_file='data/lucky_numbers.csv'):
    """对比新旧模型"""
    print("="*80)
    print("新旧模型对比验证 - 最近30期（第359-388期）")
    print("="*80)
    
    # 旧模型
    old_predictor = EnsembleZodiacPredictor()
    old_result = validate_model(old_predictor, csv_file, 359, 388, '旧模型(集成预测器)')
    
    # 新模型
    new_predictor = RetrainedZodiacPredictor()
    new_result = validate_model(new_predictor, csv_file, 359, 388, '新模型(重训练v2.0)')
    
    # 对比总结
    print("="*80)
    print("【对比总结】")
    print("="*80)
    
    print(f"\n旧模型(集成预测器):")
    print(f"  命中率: {old_result['hit_rate']:.1f}% ({old_result['hits']}/{old_result['total']})")
    
    print(f"\n新模型(重训练v2.0):")
    print(f"  命中率: {new_result['hit_rate']:.1f}% ({new_result['hits']}/{new_result['total']})")
    
    improvement = new_result['hit_rate'] - old_result['hit_rate']
    print(f"\n提升幅度: {improvement:+.1f}%")
    
    # 固定1倍投注收益对比
    print(f"\n【固定1倍投注效果对比】")
    single_bet = 16  # 4个生肖×4元
    single_win = 46  # 单次中奖金额
    total_periods = old_result['total']
    
    # 旧模型
    old_total_bet = single_bet * total_periods
    old_total_win = old_result['hits'] * single_win
    old_profit = old_total_win - old_total_bet
    old_roi = (old_profit / old_total_bet) * 100
    
    print(f"\n旧模型:")
    print(f"  总投注: {old_total_bet}元")
    print(f"  总盈利: {old_profit:+d}元")
    print(f"  ROI: {old_roi:.1f}%")
    
    # 新模型
    new_total_bet = single_bet * total_periods
    new_total_win = new_result['hits'] * single_win
    new_profit = new_total_win - new_total_bet
    new_roi = (new_profit / new_total_bet) * 100
    
    print(f"\n新模型:")
    print(f"  总投注: {new_total_bet}元")
    print(f"  总盈利: {new_profit:+d}元")
    print(f"  ROI: {new_roi:.1f}%")
    
    profit_improvement = new_profit - old_profit
    print(f"\n收益提升: {profit_improvement:+d}元")
    
    # 保存详细结果
    results_df = pd.DataFrame(new_result['results'])
    results_df.to_csv('retrained_model_validation_30periods.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 详细结果已保存: retrained_model_validation_30periods.csv")
    
    return {
        'old_model': old_result,
        'new_model': new_result,
        'improvement': improvement
    }


if __name__ == '__main__':
    compare_models()
