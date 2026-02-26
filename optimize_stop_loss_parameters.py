"""
优化止损参数 - 寻找平衡点
测试不同止损阈值，找到收益与风险的最佳平衡
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
    
    for hit in hit_records:
        multiplier = fibonacci_multiplier(consecutive_losses)
        current_bet = base_bet * multiplier
        total_investment += current_bet
        
        if hit:
            profit = win_amount * multiplier - current_bet
            total_profit += profit
            consecutive_losses = 0
            hits += 1
        else:
            total_profit -= current_bet
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        max_bet = max(max_bet, current_bet)
    
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    
    return {
        'total_profit': total_profit,
        'total_investment': total_investment,
        'roi': roi,
        'max_consecutive_losses': max_consecutive_losses,
        'max_bet': max_bet,
        'paused_periods': 0,
        'actual_betting_periods': len(hit_records)
    }

def calculate_stop_loss_strategy(hit_records, stop_loss_threshold, base_bet=20, 
                                 win_amount=47, auto_resume_after=3):
    """止损策略"""
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
    
    for hit in hit_records:
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
            else:
                total_profit -= current_bet
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                
                if consecutive_losses >= stop_loss_threshold:
                    is_betting = False
                    paused_count = 0
            
            max_bet = max(max_bet, current_bet)
        else:
            paused_periods += 1
            
            if hit:
                is_betting = True
                consecutive_losses = 0
                paused_count = 0
            else:
                paused_count += 1
                if paused_count >= auto_resume_after:
                    is_betting = True
                    paused_count = 0
    
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    
    return {
        'total_profit': total_profit,
        'total_investment': total_investment,
        'roi': roi,
        'max_consecutive_losses': max_consecutive_losses,
        'max_bet': max_bet,
        'paused_periods': paused_periods,
        'actual_betting_periods': actual_betting_periods
    }

def main():
    print("="*100)
    print("🔍 止损参数优化分析 - 寻找收益与风险的平衡点")
    print("="*100)
    print()
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    test_periods = 200
    
    # 生成预测
    predictor = OptimizedZodiacPredictor()
    all_animals = df['animal'].tolist()
    
    hit_records = []
    start_idx = max(0, len(df) - test_periods)
    
    for i in range(start_idx, len(df)):
        train_data = all_animals[:i]
        result = predictor.predict_from_history(train_data, top_n=5, debug=False)
        predicted_top5 = result['top5']
        actual = all_animals[i]
        hit = actual in predicted_top5
        hit_records.append(hit)
    
    base_hit_rate = sum(hit_records) / len(hit_records)
    print(f"✅ 基础命中率: {base_hit_rate*100:.2f}% ({sum(hit_records)}/{len(hit_records)}期)")
    print()
    
    # 基准：纯斐波那契
    fib_result = calculate_fibonacci_no_stop_loss(hit_records)
    
    print("="*100)
    print("📊 不同止损阈值对比（恢复期固定为3期）")
    print("="*100)
    print()
    
    print(f"{'策略':<25} {'总收益':<15} {'ROI':<10} {'总投入':<12} {'暂停期':<10} {'最大连亏':<10} {'最大投注':<12} {'vs纯斐波那契':<15}")
    print("-"*100)
    
    # 纯斐波那契（基准）
    print(f"{'🔥 纯斐波那契(无止损)':<25} {fib_result['total_profit']:>+12.2f}元  {fib_result['roi']:>+7.2f}%  {fib_result['total_investment']:>9.2f}元  {fib_result['paused_periods']:>7}期  {fib_result['max_consecutive_losses']:>7}期  {fib_result['max_bet']:>9.2f}元  {'基准':<15}")
    
    # 测试不同止损阈值
    stop_loss_configs = [
        (3, 3, "3期止损+3期恢复"),
        (4, 3, "4期止损+3期恢复"),
        (5, 3, "5期止损+3期恢复"),
        (6, 3, "6期止损+3期恢复"),
        (2, 2, "2期止损+2期恢复"),
        (3, 2, "3期止损+2期恢复"),
        (4, 2, "4期止损+2期恢复"),
    ]
    
    results = []
    
    for threshold, resume, name in stop_loss_configs:
        result = calculate_stop_loss_strategy(hit_records, threshold, auto_resume_after=resume)
        
        profit_diff = result['total_profit'] - fib_result['total_profit']
        profit_diff_pct = (profit_diff / abs(fib_result['total_profit']) * 100) if fib_result['total_profit'] != 0 else 0
        roi_diff = result['roi'] - fib_result['roi']
        
        vs_text = f"{profit_diff:+.0f}元({profit_diff_pct:+.1f}%)"
        
        results.append({
            'name': name,
            'threshold': threshold,
            'resume': resume,
            'result': result,
            'profit_diff': profit_diff,
            'profit_diff_pct': profit_diff_pct
        })
        
        emoji = "🏆" if abs(profit_diff_pct) < 5 else ("⚠️" if profit_diff_pct < -10 else "")
        
        print(f"{emoji + ' ' + name:<25} {result['total_profit']:>+12.2f}元  {result['roi']:>+7.2f}%  {result['total_investment']:>9.2f}元  {result['paused_periods']:>7}期  {result['max_consecutive_losses']:>7}期  {result['max_bet']:>9.2f}元  {vs_text:<15}")
    
    print()
    
    # 推荐分析
    print("="*100)
    print("💡 优化建议")
    print("="*100)
    print()
    
    # 找出收益最接近纯斐波那契的配置
    best_balance = min(results, key=lambda x: abs(x['profit_diff_pct']))
    
    # 找出最大投注最低的配置
    lowest_risk = min(results, key=lambda x: x['result']['max_bet'])
    
    print("🎯 推荐方案1: 最接近纯斐波那契收益")
    print(f"   配置: {best_balance['name']}")
    print(f"   总收益: {best_balance['result']['total_profit']:+.2f}元")
    print(f"   vs纯斐波那契: {best_balance['profit_diff']:+.2f}元 ({best_balance['profit_diff_pct']:+.1f}%)")
    print(f"   最大投注: {best_balance['result']['max_bet']:.2f}元 (vs 纯斐波那契 {fib_result['max_bet']:.2f}元)")
    print(f"   最大连亏: {best_balance['result']['max_consecutive_losses']}期 (vs 纯斐波那契 {fib_result['max_consecutive_losses']}期)")
    print()
    
    print("🛡️ 推荐方案2: 最低风险")
    print(f"   配置: {lowest_risk['name']}")
    print(f"   总收益: {lowest_risk['result']['total_profit']:+.2f}元")
    print(f"   vs纯斐波那契: {lowest_risk['profit_diff']:+.2f}元 ({lowest_risk['profit_diff_pct']:+.1f}%)")
    print(f"   最大投注: {lowest_risk['result']['max_bet']:.2f}元 (降低 {(fib_result['max_bet']-lowest_risk['result']['max_bet'])/fib_result['max_bet']*100:.1f}%)")
    print(f"   最大连亏: {lowest_risk['result']['max_consecutive_losses']}期")
    print()
    
    print("="*100)
    print("📋 结论")
    print("="*100)
    print()
    
    if best_balance['profit_diff_pct'] > -5:
        print(f"✅ 找到了平衡方案：{best_balance['name']}")
        print(f"   • 收益接近纯斐波那契（差异仅{abs(best_balance['profit_diff_pct']):.1f}%）")
        print(f"   • 最大投注从{fib_result['max_bet']:.0f}元降至{best_balance['result']['max_bet']:.0f}元")
        print(f"   • 最大连亏从{fib_result['max_consecutive_losses']}期降至{best_balance['result']['max_consecutive_losses']}期")
    else:
        print("⚠️  所有止损配置都显著降低收益")
        print()
        print("建议：")
        print("   1. 不使用止损，直接用纯斐波那契倍投")
        print("   2. 或仅在极端情况（连亏7+期）启用止损")
        print("   3. 优先提升预测准确率至60%+")
    
    print()

if __name__ == "__main__":
    main()
