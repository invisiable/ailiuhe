"""
测试N=3止损策略v2.0 - 智能观察恢复版本

新规则：
1. 连续4期不中 → 进入暂停观察状态
2. 暂停期间不投注，持续观察：
   - 观察到命中 → 立即恢复投注
   - 观察到连续3期不中 → 恢复投注
3. 恢复时倍数重置为1倍
"""

import pandas as pd
import numpy as np
from zodiac_simple_smart import ZodiacSimpleSmart


def test_n3_v2_strategy():
    """测试N=3止损v2.0策略"""
    
    print("="*80)
    print("N=3止损策略v2.0 - 智能观察恢复测试")
    print("="*80)
    print()
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    animals = df['animal'].tolist()
    
    total_periods = len(animals)
    test_periods = 200
    start = total_periods - test_periods
    
    print(f"数据总期数: {total_periods}")
    print(f"测试期数: {test_periods}")
    print(f"起始索引: {start}")
    print()
    
    # 初始化预测器
    predictor = ZodiacSimpleSmart()
    
    # 斐波那契数列
    def fibonacci_multiplier(losses):
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        if losses < len(fib):
            return fib[losses]
        return fib[-1]
    
    # 策略状态
    total_profit = 0
    total_investment = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    max_bet = 20
    max_drawdown = 0
    peak_balance = 0
    
    is_paused = False
    observe_losses = 0
    paused_periods = 0
    actual_betting_periods = 0
    hits = 0
    
    period_details = []
    
    # 回测
    for i in range(start, total_periods):
        history = animals[:i]
        actual_animal = animals[i]
        
        # 预测TOP5
        if len(history) >= 30:
            result = predictor.predict_from_history(history, top_n=5)
            predicted_top5 = result['top5']
        else:
            predicted_top5 = predictor.zodiac_list[:5]
        
        # 判断命中
        hit = actual_animal in predicted_top5
        
        if is_paused:
            # 暂停观察期间不投注
            paused_periods += 1
            
            if hit:
                # 观察到命中 → 立即恢复
                is_paused = False
                consecutive_losses = 0
                observe_losses = 0
                status = '⏸观察→✓命中恢复'
            else:
                # 观察到不中
                observe_losses += 1
                
                if observe_losses >= 3:
                    # 观察到连续3期不中 → 恢复
                    is_paused = False
                    consecutive_losses = 0
                    observe_losses = 0
                    status = '⏸观察→3期不中恢复'
                else:
                    status = f'⏸观察中({observe_losses}/3)'
            
            period_details.append({
                'period': i - start,
                'date': df.iloc[i]['date'],
                'actual': actual_animal,
                'predicted': ','.join(predicted_top5[:3]),
                'multiplier': 0,
                'bet': 0,
                'status': status,
                'profit': 0,
                'cumulative': total_profit
            })
            continue
        
        # 正常投注期
        multiplier = fibonacci_multiplier(consecutive_losses)
        current_bet = 20 * multiplier
        total_investment += current_bet
        actual_betting_periods += 1
        max_bet = max(max_bet, current_bet)
        
        if hit:
            profit = 47 * multiplier - current_bet
            total_profit += profit
            consecutive_losses = 0
            observe_losses = 0
            hits += 1
            status = '✓中'
        else:
            profit = -current_bet
            total_profit += profit
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            status = '✗失'
            
            # 检查是否触发观察（4连败）
            if consecutive_losses >= 4:
                is_paused = True
                observe_losses = 0
                status = '✗失(4连败→观察)'
        
        period_details.append({
            'period': i - start,
            'date': df.iloc[i]['date'],
            'actual': actual_animal,
            'predicted': ','.join(predicted_top5[:3]),
            'multiplier': multiplier,
            'bet': current_bet,
            'status': status,
            'profit': profit,
            'cumulative': total_profit
        })
        
        peak_balance = max(peak_balance, total_profit)
        drawdown = peak_balance - total_profit
        max_drawdown = max(max_drawdown, drawdown)
    
    # 计算结果
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    hit_rate = (hits / actual_betting_periods) if actual_betting_periods > 0 else 0
    paused_rate = (paused_periods / test_periods) * 100
    
    print("v2.0策略测试结果:")
    print(f"  测试期数: {test_periods}期")
    print(f"  实际投注期数: {actual_betting_periods}期")
    print(f"  观察期数: {paused_periods}期 ({paused_rate:.1f}%)")
    print(f"  命中次数: {hits}次")
    print(f"  命中率: {hit_rate*100:.2f}%")
    print(f"  总投入: {total_investment:.0f}元")
    print(f"  净利润: {total_profit:+.0f}元")
    print(f"  ROI: {roi:.2f}%")
    print(f"  最大回撤: {max_drawdown:.0f}元")
    print(f"  最大连败: {max_consecutive_losses}期")
    print(f"  最大投注: {max_bet:.0f}元")
    print()
    
    # 显示前20期和后20期详情
    print("="*100)
    print("详细投注记录（前20期 + 后20期）:")
    print("="*100)
    print(f"{'期号':<6}{'日期':<12}{'实际':<6}{'预测':<15}{'倍数':<8}{'投注':<8}{'状态':<25}{'盈亏':<10}{'累计':<10}")
    print("-"*100)
    
    # 前20期
    for detail in period_details[:20]:
        period = detail['period'] + 1
        date_str = detail['date']
        actual = detail['actual']
        predicted = detail['predicted']
        multiplier = detail['multiplier']
        bet = detail['bet']
        status = detail['status']
        profit = detail['profit']
        cumulative = detail['cumulative']
        
        mult_str = f"{multiplier:.0f}" if multiplier > 0 else "-"
        bet_str = f"{bet:.0f}" if bet > 0 else "-"
        profit_str = f"{profit:+.0f}" if profit != 0 else "-"
        
        print(f"{period:<6}{date_str:<12}{actual:<6}{predicted:<15}{mult_str:<8}{bet_str:<8}{status:<25}{profit_str:<10}{cumulative:+10.0f}")
    
    print("...")
    
    # 后20期
    for detail in period_details[-20:]:
        period = detail['period'] + 1
        date_str = detail['date']
        actual = detail['actual']
        predicted = detail['predicted']
        multiplier = detail['multiplier']
        bet = detail['bet']
        status = detail['status']
        profit = detail['profit']
        cumulative = detail['cumulative']
        
        mult_str = f"{multiplier:.0f}" if multiplier > 0 else "-"
        bet_str = f"{bet:.0f}" if bet > 0 else "-"
        profit_str = f"{profit:+.0f}" if profit != 0 else "-"
        
        print(f"{period:<6}{date_str:<12}{actual:<6}{predicted:<15}{mult_str:<8}{bet_str:<8}{status:<25}{profit_str:<10}{cumulative:+10.0f}")
    
    print()
    print("="*100)
    print()
    
    # 分析观察恢复情况
    print("智能观察分析:")
    hit_resume_count = sum(1 for d in period_details if '命中恢复' in d['status'])
    miss3_resume_count = sum(1 for d in period_details if '3期不中恢复' in d['status'])
    observe_trigger_count = sum(1 for d in period_details if '4连败→观察' in d['status'])
    
    print(f"  触发观察次数: {observe_trigger_count}次")
    print(f"  观察到命中恢复: {hit_resume_count}次 ({hit_resume_count/observe_trigger_count*100 if observe_trigger_count > 0 else 0:.1f}%)")
    print(f"  观察到3期不中恢复: {miss3_resume_count}次 ({miss3_resume_count/observe_trigger_count*100 if observe_trigger_count > 0 else 0:.1f}%)")
    print()
    
    print("="*100)
    print("🎉 v2.0策略优势:")
    print("  1. ✅ 智能观察机制：若观察到命中，立即恢复投注，不错过机会")
    print("  2. ✅ 灵活恢复条件：观察到连续3期不中，说明走势稳定，恢复投注")
    print("  3. ✅ 风险控制：4连败后暂停，避免在连败期深陷")
    print("  4. ✅ 倍数重置：恢复投注时重置为1倍，控制单次风险")
    print("="*100)
    print()


if __name__ == "__main__":
    test_n3_v2_strategy()
