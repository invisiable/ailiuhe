"""
测试生肖TOP5止损策略集成
验证：2期止损 + 3期恢复 + 斐波那契倍投
"""
import pandas as pd
from optimized_zodiac_predictor import OptimizedZodiacPredictor

def fibonacci_multiplier(losses):
    """斐波那契数列倍数"""
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    return fib[losses] if losses < len(fib) else fib[-1]

def calculate_stop_loss_strategy(hit_records, stop_loss_threshold=2, base_bet=20, 
                                 win_amount=47, auto_resume_after=3):
    """计算止损策略结果（与GUI中的函数一致）"""
    total_profit = 0
    total_investment = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    max_bet = base_bet
    
    is_betting = True
    paused_periods = 0
    paused_count = 0  # 当前连续暂停期数
    actual_betting_periods = 0
    hits = 0
    
    period_details = []
    
    for i, hit in enumerate(hit_records):
        if is_betting:
            # 当前在投注状态
            paused_count = 0  # 重置暂停计数
            
            multiplier = fibonacci_multiplier(consecutive_losses)
            current_bet = base_bet * multiplier
            total_investment += current_bet
            actual_betting_periods += 1
            
            if hit:
                profit = win_amount * multiplier - current_bet
                total_profit += profit
                consecutive_losses = 0
                hits += 1
                period_profit = profit
                status = '✓中'
            else:
                total_profit -= current_bet
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                period_profit = -current_bet
                status = '✗失'
                
                if consecutive_losses >= stop_loss_threshold:
                    is_betting = False
                    paused_count = 0
                    status = '✗失(止损)'
            
            max_bet = max(max_bet, current_bet)
            
            period_details.append({
                'period': i,
                'multiplier': multiplier,
                'bet': current_bet,
                'status': status,
                'profit': period_profit,
                'cumulative': total_profit,
                'is_betting': True
            })
        else:
            # 暂停投注状态
            paused_periods += 1
            
            if hit:
                # 命中时立即恢复
                is_betting = True
                consecutive_losses = 0
                paused_count = 0
                status = '⏸暂停(恢复)'
            else:
                # 没中，计数暂停期数
                paused_count += 1
                if paused_count >= auto_resume_after:
                    # 自动恢复
                    is_betting = True
                    paused_count = 0
                    status = '⏸暂停(自动恢复)'
                else:
                    status = '⏸暂停'
            
            period_details.append({
                'period': i,
                'multiplier': 0,
                'bet': 0,
                'status': status,
                'profit': 0,
                'cumulative': total_profit,
                'is_betting': False
            })
    
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    hit_rate = (hits / actual_betting_periods) if actual_betting_periods > 0 else 0
    
    return {
        'total_profit': total_profit,
        'total_investment': total_investment,
        'roi': roi,
        'hit_rate': hit_rate,
        'hits': hits,
        'max_consecutive_losses': max_consecutive_losses,
        'max_bet': max_bet,
        'actual_betting_periods': actual_betting_periods,
        'paused_periods': paused_periods,
        'period_details': period_details
    }

def main():
    print("="*80)
    print("🐉 生肖TOP5止损策略集成测试")
    print("="*80)
    print()
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"✅ 数据加载完成: {len(df)}期")
    
    # 测试期数
    test_periods = 200
    print(f"📊 测试期数: 最近{test_periods}期")
    print()
    
    # 生成预测
    predictor = OptimizedZodiacPredictor()
    all_animals = df['animal'].tolist()
    
    predictions_top5 = []
    actuals = []
    hit_records = []
    
    start_idx = max(0, len(df) - test_periods)
    
    print("🔄 生成TOP5预测...")
    for i in range(start_idx, len(df)):
        train_data = all_animals[:i]
        result = predictor.predict_from_history(train_data, top_n=5, debug=False)
        predicted_top5 = result['top5']
        actual = all_animals[i]
        
        predictions_top5.append(predicted_top5)
        actuals.append(actual)
        hit = actual in predicted_top5
        hit_records.append(hit)
    
    base_hit_rate = sum(hit_records) / len(hit_records)
    print(f"✅ 基础命中率: {base_hit_rate*100:.2f}%")
    print()
    
    # 计算止损策略
    print("="*80)
    print("🏆 止损策略计算（2期止损 + 3期恢复 + 斐波那契倍投）")
    print("="*80)
    print()
    
    result = calculate_stop_loss_strategy(
        hit_records,
        stop_loss_threshold=2,
        base_bet=20,
        win_amount=47,
        auto_resume_after=3
    )
    
    print(f"总收益: {result['total_profit']:+.2f}元")
    print(f"总投入: {result['total_investment']:.2f}元")
    print(f"ROI: {result['roi']:+.2f}%")
    print(f"命中率: {result['hit_rate']*100:.2f}%")
    print(f"实际投注: {result['actual_betting_periods']}期")
    print(f"暂停期数: {result['paused_periods']}期")
    print(f"最大连亏: {result['max_consecutive_losses']}期")
    print(f"最大单期投注: {result['max_bet']:.2f}元")
    print()
    
    # 显示详细记录（最近20期）
    print("="*80)
    print("📋 最近20期详细记录")
    print("="*80)
    print()
    print(f"{'期数':<8} {'日期':<12} {'实际':<6} {'预测TOP5':<30} {'倍数':<6} {'投注':<8} {'结果':<12} {'当期收益':<10} {'累计收益':<10}")
    print("-" * 120)
    
    period_details = result['period_details']
    for i in range(max(0, len(period_details)-20), len(period_details)):
        detail = period_details[i]
        idx = start_idx + detail['period']
        actual_row = df.iloc[idx]
        date_str = actual_row['date']
        actual_animal = actual_row['animal']
        predicted_top5 = predictions_top5[detail['period']]
        
        multiplier = detail['multiplier']
        current_bet = detail['bet']
        status = detail['status']
        period_profit = detail['profit']
        cumulative_profit = detail['cumulative']
        
        if period_profit > 0:
            profit_str = f"+{period_profit:.2f}"
        elif period_profit < 0:
            profit_str = f"{period_profit:.2f}"
        else:
            profit_str = "0.00"
        
        top5_str = ','.join(predicted_top5[:5])
        
        if detail['is_betting']:
            print(f"第{idx+1:<5}期 {date_str:<12} {actual_animal:<6} {top5_str:<30} {multiplier:<6.1f} {current_bet:<8.0f} {status:<12} {profit_str:<10} {cumulative_profit:>+10.2f}")
        else:
            print(f"第{idx+1:<5}期 {date_str:<12} {actual_animal:<6} {top5_str:<30} {'--':<6} {'--':<8} {status:<12} {profit_str:<10} {cumulative_profit:>+10.2f}")
    
    print("-" * 120)
    print()
    
    # 对比目标
    print("="*80)
    print("🎯 目标对比")
    print("="*80)
    print()
    print(f"目标ROI:  26.06%")
    print(f"实际ROI:  {result['roi']:+.2f}%")
    print(f"差异:     {result['roi']-26.06:+.2f}%")
    print()
    
    if result['roi'] >= 25.0:
        print("✅ 测试通过！ROI达到预期范围（25-27%）")
    else:
        print("⚠️  警告：ROI低于预期")
    print()
    
    print("="*80)
    print("✅ 测试完成")
    print("="*80)

if __name__ == "__main__":
    main()
