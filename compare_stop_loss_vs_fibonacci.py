"""
对比止损策略 vs 纯斐波那契倍投的收益
分析止损策略对收益的影响
"""
import pandas as pd
from optimized_zodiac_predictor import OptimizedZodiacPredictor

def fibonacci_multiplier(losses):
    """斐波那契数列倍数"""
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    return fib[losses] if losses < len(fib) else fib[-1]

def calculate_fibonacci_no_stop_loss(hit_records, base_bet=20, win_amount=47):
    """纯斐波那契倍投（无止损）"""
    total_profit = 0
    total_investment = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    max_bet = base_bet
    hits = 0
    
    period_details = []
    
    for i, hit in enumerate(hit_records):
        multiplier = fibonacci_multiplier(consecutive_losses)
        current_bet = base_bet * multiplier
        total_investment += current_bet
        
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
    
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    hit_rate = hits / len(hit_records) if len(hit_records) > 0 else 0
    
    return {
        'total_profit': total_profit,
        'total_investment': total_investment,
        'roi': roi,
        'hit_rate': hit_rate,
        'hits': hits,
        'max_consecutive_losses': max_consecutive_losses,
        'max_bet': max_bet,
        'actual_betting_periods': len(hit_records),
        'paused_periods': 0,
        'period_details': period_details
    }

def calculate_stop_loss_strategy(hit_records, stop_loss_threshold=2, base_bet=20, 
                                 win_amount=47, auto_resume_after=3):
    """止损策略（2期止损+3期恢复+斐波那契）"""
    total_profit = 0
    total_investment = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    max_bet = base_bet
    
    is_betting = True
    paused_periods = 0
    paused_count = 0
    actual_betting_periods = 0
    hits = 0
    
    period_details = []
    
    for i, hit in enumerate(hit_records):
        if is_betting:
            paused_count = 0
            
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
            paused_periods += 1
            
            if hit:
                is_betting = True
                consecutive_losses = 0
                paused_count = 0
                status = '⏸暂停(恢复)'
            else:
                paused_count += 1
                if paused_count >= auto_resume_after:
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
    print("="*100)
    print("📊 止损策略 vs 纯斐波那契倍投 - 收益对比分析")
    print("="*100)
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
    hits_count = sum(hit_records)
    print(f"✅ 基础命中率: {base_hit_rate*100:.2f}% ({hits_count}/{len(hit_records)}期)")
    print()
    
    # 计算两种策略
    print("="*100)
    print("🔥 策略1: 纯斐波那契倍投（无止损）")
    print("="*100)
    print()
    
    fib_result = calculate_fibonacci_no_stop_loss(hit_records, base_bet=20, win_amount=47)
    
    print(f"总收益:       {fib_result['total_profit']:+.2f}元")
    print(f"总投入:       {fib_result['total_investment']:.2f}元")
    print(f"ROI:          {fib_result['roi']:+.2f}%")
    print(f"命中率:       {fib_result['hit_rate']*100:.2f}%")
    print(f"命中次数:     {fib_result['hits']}次")
    print(f"投注期数:     {fib_result['actual_betting_periods']}期")
    print(f"最大连亏:     {fib_result['max_consecutive_losses']}期")
    print(f"最大单期投注: {fib_result['max_bet']:.2f}元")
    print()
    
    print("="*100)
    print("🛡️ 策略2: 止损策略（2期止损+3期恢复+斐波那契）")
    print("="*100)
    print()
    
    stop_result = calculate_stop_loss_strategy(
        hit_records,
        stop_loss_threshold=2,
        base_bet=20,
        win_amount=47,
        auto_resume_after=3
    )
    
    print(f"总收益:       {stop_result['total_profit']:+.2f}元")
    print(f"总投入:       {stop_result['total_investment']:.2f}元")
    print(f"ROI:          {stop_result['roi']:+.2f}%")
    print(f"命中率:       {stop_result['hit_rate']*100:.2f}%")
    print(f"命中次数:     {stop_result['hits']}次")
    print(f"实际投注:     {stop_result['actual_betting_periods']}期")
    print(f"暂停期数:     {stop_result['paused_periods']}期")
    print(f"最大连亏:     {stop_result['max_consecutive_losses']}期")
    print(f"最大单期投注: {stop_result['max_bet']:.2f}元")
    print()
    
    # 详细对比
    print("="*100)
    print("📈 收益对比分析")
    print("="*100)
    print()
    
    profit_diff = stop_result['total_profit'] - fib_result['total_profit']
    profit_diff_pct = (profit_diff / abs(fib_result['total_profit']) * 100) if fib_result['total_profit'] != 0 else 0
    
    roi_diff = stop_result['roi'] - fib_result['roi']
    
    investment_diff = fib_result['total_investment'] - stop_result['total_investment']
    investment_saved_pct = (investment_diff / fib_result['total_investment'] * 100) if fib_result['total_investment'] > 0 else 0
    
    max_bet_diff = fib_result['max_bet'] - stop_result['max_bet']
    
    print(f"{'指标':<20} {'纯斐波那契':<20} {'止损策略':<20} {'差异':<20}")
    print("-"*100)
    print(f"{'总收益':<20} {fib_result['total_profit']:>+15.2f}元   {stop_result['total_profit']:>+15.2f}元   {profit_diff:>+15.2f}元 ({profit_diff_pct:+.1f}%)")
    print(f"{'ROI':<20} {fib_result['roi']:>+15.2f}%    {stop_result['roi']:>+15.2f}%    {roi_diff:>+15.2f}%")
    print(f"{'总投入':<20} {fib_result['total_investment']:>15.2f}元   {stop_result['total_investment']:>15.2f}元   {investment_diff:>+15.2f}元 (省{investment_saved_pct:.1f}%)")
    print(f"{'命中次数':<20} {fib_result['hits']:>15}次    {stop_result['hits']:>15}次    {stop_result['hits']-fib_result['hits']:>+15}次")
    print(f"{'投注期数':<20} {fib_result['actual_betting_periods']:>15}期    {stop_result['actual_betting_periods']:>15}期    {stop_result['actual_betting_periods']-fib_result['actual_betting_periods']:>+15}期")
    print(f"{'暂停期数':<20} {fib_result['paused_periods']:>15}期    {stop_result['paused_periods']:>15}期    {stop_result['paused_periods']-fib_result['paused_periods']:>+15}期")
    print(f"{'最大连亏':<20} {fib_result['max_consecutive_losses']:>15}期    {stop_result['max_consecutive_losses']:>15}期    {stop_result['max_consecutive_losses']-fib_result['max_consecutive_losses']:>+15}期")
    print(f"{'最大单期投注':<20} {fib_result['max_bet']:>15.2f}元   {stop_result['max_bet']:>15.2f}元   {max_bet_diff:>+15.2f}元 (降{max_bet_diff/fib_result['max_bet']*100:.1f}%)")
    print()
    
    # 分析
    print("="*100)
    print("💡 策略分析")
    print("="*100)
    print()
    
    if profit_diff < 0:
        print(f"⚠️  止损策略收益比纯斐波那契少 {abs(profit_diff):.2f}元 ({abs(profit_diff_pct):.1f}%)")
        print()
        print("📊 收益降低的原因：")
        print(f"   1. 暂停投注错失 {stop_result['paused_periods']} 期")
        print(f"   2. 总投入减少 {investment_diff:.2f}元 ({investment_saved_pct:.1f}%)")
        print(f"   3. 实际投注期数减少 {fib_result['actual_betting_periods'] - stop_result['actual_betting_periods']} 期")
        print()
        print("🛡️ 止损策略的优势：")
        print(f"   1. 最大单期投注降低 {max_bet_diff:.2f}元 ({max_bet_diff/fib_result['max_bet']*100:.1f}%)")
        print(f"   2. 最大连亏控制在 {stop_result['max_consecutive_losses']} 期（vs {fib_result['max_consecutive_losses']} 期）")
        print(f"   3. 总投入减少 {investment_saved_pct:.1f}%，资金压力更小")
        print(f"   4. 风险更可控，避免深度亏损")
    else:
        print(f"✅ 止损策略收益比纯斐波那契多 {profit_diff:.2f}元 ({profit_diff_pct:.1f}%)")
        print(f"   同时降低了 {investment_saved_pct:.1f}% 的资金投入")
    
    print()
    
    # 找出关键差异期
    print("="*100)
    print("🔍 关键差异分析 - 止损错失的重要时机")
    print("="*100)
    print()
    
    missed_opportunities = []
    for i, detail in enumerate(stop_result['period_details']):
        if not detail['is_betting'] and hit_records[i]:
            # 暂停期间命中的情况
            idx = start_idx + i
            actual_row = df.iloc[idx]
            date_str = actual_row['date']
            actual_animal = actual_row['animal']
            predicted_top5 = predictions_top5[i]
            
            # 计算如果没暂停会获得的收益
            fib_detail = fib_result['period_details'][i]
            potential_profit = fib_detail['profit']
            
            missed_opportunities.append({
                'period': idx + 1,
                'date': date_str,
                'actual': actual_animal,
                'predicted': ','.join(predicted_top5[:5]),
                'status': detail['status'],
                'missed_profit': potential_profit
            })
    
    if missed_opportunities:
        print(f"暂停期间共错失 {len(missed_opportunities)} 次命中机会：")
        print()
        print(f"{'期数':<8} {'日期':<12} {'实际':<6} {'预测TOP5':<30} {'状态':<15} {'错失收益':<10}")
        print("-"*100)
        total_missed = 0
        for opp in missed_opportunities[:10]:  # 只显示前10个
            print(f"第{opp['period']:<5}期 {opp['date']:<12} {opp['actual']:<6} {opp['predicted']:<30} {opp['status']:<15} {opp['missed_profit']:>+8.2f}元")
            total_missed += opp['missed_profit']
        
        if len(missed_opportunities) > 10:
            print(f"... 还有 {len(missed_opportunities)-10} 次未显示")
            for opp in missed_opportunities[10:]:
                total_missed += opp['missed_profit']
        
        print("-"*100)
        print(f"错失收益合计: {total_missed:+.2f}元")
        print()
    else:
        print("✅ 暂停期间没有错失命中机会")
        print()
    
    # 结论
    print("="*100)
    print("🎯 结论与建议")
    print("="*100)
    print()
    
    if abs(profit_diff_pct) < 10:
        print("✅ 两种策略收益相近，差异在可接受范围内")
        print()
        print("🏆 推荐使用止损策略，因为：")
        print(f"   • 最大单期投注降低 {max_bet_diff/fib_result['max_bet']*100:.1f}%")
        print(f"   • 总投入减少 {investment_saved_pct:.1f}%")
        print("   • 风险更可控，适合长期稳定")
    elif profit_diff < 0 and abs(profit_diff_pct) > 20:
        print(f"⚠️  止损策略收益显著降低 {abs(profit_diff_pct):.1f}%")
        print()
        print("📋 可能的优化方向：")
        print("   1. 调整止损阈值：从2期改为3期")
        print("   2. 调整恢复期数：从3期改为2期")
        print("   3. 改进预测模型，提升命中率至60%+")
        print("   4. 仅在高风险时段启用止损")
    else:
        print("✅ 止损策略在降低风险的同时保持了良好收益")
    
    print()
    print("="*100)

if __name__ == "__main__":
    main()
