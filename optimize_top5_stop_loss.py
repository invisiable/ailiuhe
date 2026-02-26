"""
优化生肖TOP5止损策略分析
目标：找到ROI达到50%以上的最佳止损策略
"""
import pandas as pd

def fibonacci_multiplier(losses):
    """斐波那契数列倍数"""
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    return fib[losses] if losses < len(fib) else fib[-1]

def martingale_multiplier(losses):
    """马丁格尔倍数 (2^n)"""
    return min(2 ** losses, 64)

def conservative_multiplier(losses):
    """保守倍投 (1+n*0.5)"""
    return 1 + losses * 0.5 if losses <= 6 else 4

def dalembert_multiplier(losses):
    """达朗贝尔倍投 (1+n)"""
    return 1 + losses if losses <= 10 else 11

def aggressive_multiplier(losses):
    """激进倍投 (1.5^n)"""
    return min(1.5 ** losses, 100)

def ultra_aggressive_multiplier(losses):
    """超激进倍投 (3^n)"""
    return min(3 ** losses, 243)

def calculate_stop_loss_strategy(hit_records, stop_loss_threshold=3, base_bet=20, 
                                 win_amount=47, multiplier_func=None, 
                                 auto_resume_after=5):
    """计算止损策略结果"""
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
    
    get_multiplier = multiplier_func if multiplier_func else fibonacci_multiplier
    
    for hit in hit_records:
        if is_betting:
            paused_count = 0
            multiplier = get_multiplier(consecutive_losses)
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
    hit_rate = (hits / actual_betting_periods * 100) if actual_betting_periods > 0 else 0
    
    return {
        'total_profit': total_profit,
        'total_investment': total_investment,
        'roi': roi,
        'hit_rate': hit_rate,
        'hits': hits,
        'actual_betting_periods': actual_betting_periods,
        'paused_periods': paused_periods,
        'max_consecutive_losses': max_consecutive_losses,
        'max_bet': max_bet
    }

def generate_top5_predictions(df):
    """生成TOP5预测"""
    from optimized_zodiac_predictor import OptimizedZodiacPredictor
    
    predictor = OptimizedZodiacPredictor()
    hit_records = []
    
    for i in range(len(df)):
        train_animals = df['animal'].iloc[:i].tolist()
        if len(train_animals) < 10:
            continue
            
        result = predictor.predict_from_history(train_animals, top_n=5, debug=False)
        top5 = result['top5']
        
        actual = df.iloc[i]['animal']
        hit = actual in top5
        hit_records.append(hit)
    
    return hit_records

def main():
    print("="*80)
    print("生肖TOP5止损策略优化分析 - 目标ROI 50%+")
    print("="*80)
    print()
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"✅ 数据加载完成: {len(df)}期")
    
    test_periods = min(200, len(df))
    start_idx = len(df) - test_periods
    df_test = df.iloc[start_idx:].reset_index(drop=True)
    
    print(f"📊 分析期数: 最近{test_periods}期")
    print()
    
    print("🔄 生成TOP5预测数据...")
    hit_records = generate_top5_predictions(df_test)
    
    base_hit_rate = sum(hit_records) / len(hit_records) * 100
    print(f"✅ 基础命中率: {base_hit_rate:.2f}%")
    print(f"✅ 有效测试期数: {len(hit_records)}期")
    print()
    
    print("="*80)
    print("开始测试不同止损策略组合...")
    print("="*80)
    print()
    
    multiplier_strategies = {
        '固定1倍': lambda x: 1,
        '斐波那契': fibonacci_multiplier,
        '马丁格尔': martingale_multiplier,
        '保守倍投': conservative_multiplier,
        '达朗贝尔': dalembert_multiplier,
        '激进倍投': aggressive_multiplier,
        '超激进倍投': ultra_aggressive_multiplier,
    }
    
    all_strategies = []
    
    for stop_threshold in [2, 3, 4, 5]:
        for auto_resume in [3, 5, 7, 10]:
            for strategy_name, multiplier_func in multiplier_strategies.items():
                result = calculate_stop_loss_strategy(
                    hit_records,
                    stop_loss_threshold=stop_threshold,
                    base_bet=20,
                    win_amount=47,
                    multiplier_func=multiplier_func,
                    auto_resume_after=auto_resume
                )
                
                all_strategies.append({
                    'stop_threshold': stop_threshold,
                    'auto_resume': auto_resume,
                    'strategy_name': strategy_name,
                    **result
                })
    
    all_strategies.sort(key=lambda x: x['roi'], reverse=True)
    best_strategies = [s for s in all_strategies if s['roi'] >= 50]
    
    print(f"{'='*80}")
    print(f"找到 {len(best_strategies)} 个ROI>=50%的策略")
    print(f"{'='*80}")
    print()
    
    print("TOP 20 ROI最高的策略：")
    print(f"{'排名':<6} {'止损':<8} {'恢复':<8} {'倍投策略':<14} {'ROI':<10} {'收益':<10} {'投入':<10} {'命中率':<10}")
    print("-"*90)
    
    for i, s in enumerate(all_strategies[:20], 1):
        print(f"{i:<6} {s['stop_threshold']}期{'':<4} {s['auto_resume']}期{'':<4} "
              f"{s['strategy_name']:<14} {s['roi']:>6.2f}% "
              f"{s['total_profit']:>8.0f}元 {s['total_investment']:>8.0f}元 "
              f"{s['hit_rate']:>6.2f}%")
    print()
    
    if best_strategies:
        best = best_strategies[0]
        print("="*80)
        print("🏆 推荐策略（ROI最高）")
        print("="*80)
        print(f"止损阈值: 连续失败{best['stop_threshold']}期后暂停投注")
        print(f"自动恢复: 暂停后连续错误{best['auto_resume']}期自动恢复")
        print(f"倍投策略: {best['strategy_name']}")
        print(f"基础投注: 20元 (TOP5，每个生肖4元)")
        print()
        print(f"📊 回测结果：")
        print(f"  总收益: {best['total_profit']:+.0f}元")
        print(f"  总投入: {best['total_investment']:.0f}元")
        print(f"  ROI: {best['roi']:+.2f}%")
        print(f"  命中率: {best['hit_rate']:.2f}%")
        print(f"  最大单期投注: {best['max_bet']:.0f}元")
    else:
        best = all_strategies[0]
        print("⚠️ 未找到ROI>=50%的策略")
        print(f"最高ROI策略: {best['roi']:.2f}% ({best['strategy_name']})")

if __name__ == '__main__':
    main()
