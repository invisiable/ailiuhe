"""
生肖TOP4投注策略回测 - 集成优化系统模型（自动恢复版）
使用止损策略：连续失败3期暂停，满足以下任一条件恢复投注：
  1. 预测命中时
  2. 连续暂停8期后自动恢复
"""

import pandas as pd
from ensemble_zodiac_predictor import EnsembleZodiacPredictor

def fibonacci_multiplier(losses):
    """斐波那契倍数计算"""
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    if losses < len(fib):
        return fib[losses]
    return fib[-1]

def backtest_stop_loss_auto_resume(periods=200):
    """回测止损策略（带自动恢复）"""
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    df = df.tail(periods).reset_index(drop=True)
    
    predictor = EnsembleZodiacPredictor()
    
    # 统计变量
    total_profit = 0
    total_investment = 0
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    max_consecutive_wins = 0
    base_bet = 16
    win_amount = 45
    
    # 止损相关变量
    is_betting = True  # 当前是否投注
    paused_periods = 0  # 总暂停期数
    paused_count = 0  # 当前连续暂停期数
    actual_betting_periods = 0  # 实际投注期数
    hits = 0
    stop_loss_threshold = 3  # 连续失败3期触发止损
    max_paused_streak = 8  # 最大连续暂停期数
    
    results = []
    
    print(f"{'='*80}")
    print(" 生肖TOP4投注策略回测 - 集成优化系统（带自动恢复止损）")
    print(f"{'='*80}")
    print(f"回测期数: {periods} 期")
    print(f"基础投注: {base_bet}元/期")
    print(f"命中奖励: {win_amount}元")
    print(f"止损规则: 连续失败 {stop_loss_threshold} 期暂停")
    print(f"恢复规则: 预测命中 OR 连续暂停 {max_paused_streak} 期后自动恢复")
    print(f"{'='*80}\n")
    
    for i in range(len(df)):
        # 获取当前期数据
        current_date = df.iloc[i]['date']
        actual_animal = df.iloc[i]['animal']
        
        # 预测TOP4生肖
        if i >= 10:  # 至少需要10期历史数据
            history = df.iloc[:i]['animal'].tolist()
            prediction = predictor.predict_from_history(history, top_n=4)
            top4_animals = prediction['top4']  # 提取TOP4生肖列表
            
            # 判断是否命中
            hit = actual_animal in top4_animals
            
            # 投注逻辑
            if is_betting:
                # 当前在投注状态
                paused_count = 0  # 重置暂停计数器
                
                # 计算投注倍数
                multiplier = fibonacci_multiplier(consecutive_losses)
                current_bet = base_bet * multiplier
                total_investment += current_bet
                actual_betting_periods += 1
                
                if hit:
                    # 命中
                    profit = win_amount * multiplier - current_bet
                    total_profit += profit
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    hits += 1
                    status = f"[HIT] 命中 +{profit}元 (连胜{consecutive_wins})"
                else:
                    # 未中
                    total_profit -= current_bet
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    status = f"[MISS] 未中 -{current_bet}元 (连败{consecutive_losses})"
                    
                    # 检查是否触发止损
                    if consecutive_losses >= stop_loss_threshold:
                        is_betting = False
                        paused_count = 0
                        status += f" [触发止损，暂停投注]"
            else:
                # 暂停投注状态
                paused_periods += 1
                paused_count += 1
                current_bet = 0
                
                # 检查恢复条件
                resume_reason = ""
                if hit:
                    # 条件1：预测命中，恢复投注
                    is_betting = True
                    consecutive_losses = 0
                    consecutive_wins = 0
                    paused_count = 0
                    resume_reason = "预测命中"
                elif paused_count >= max_paused_streak:
                    # 条件2：连续暂停8期，自动恢复
                    is_betting = True
                    consecutive_losses = 0
                    consecutive_wins = 0
                    paused_count = 0
                    resume_reason = f"连续暂停{max_paused_streak}期"
                
                if resume_reason:
                    status = f"[RESUME] 暂停中(第{paused_count}期) -> 恢复投注（{resume_reason}）"
                else:
                    hit_status = "本期会中" if hit else "本期未中"
                    status = f"[PAUSE] 暂停投注(第{paused_count}期，{hit_status})"
            
            # 记录结果
            results.append({
                '期数': i + 1,
                '日期': current_date,
                '预测TOP4': ', '.join(top4_animals),
                '实际': actual_animal,
                '状态': status,
                '投注': current_bet,
                '余额': total_profit
            })
            
            # 打印每期结果
            roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
            print(f"第{i+1:3d}期 {current_date} | 预测: {','.join(top4_animals):20s} | 实际: {actual_animal:2s} | "
                  f"投注: {current_bet:4d}元 | {status:40s} | 余额: {total_profit:+6.0f}元 | ROI: {roi:+6.2f}%")
    
    # 计算统计指标
    hit_rate = (hits / actual_betting_periods * 100) if actual_betting_periods > 0 else 0
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    avg_bet = total_investment / actual_betting_periods if actual_betting_periods > 0 else 0
    
    # 打印总结
    print(f"\n{'='*80}")
    print(" 回测总结 ")
    print(f"{'='*80}")
    print(f"总期数: {periods} 期")
    print(f"实际投注期数: {actual_betting_periods} 期")
    print(f"暂停期数: {paused_periods} 期 ({paused_periods/periods*100:.1f}%)")
    print(f"命中次数: {hits} 次")
    print(f"命中率: {hit_rate:.2f}%")
    print(f"\n投注统计:")
    print(f"  总投入: {total_investment:.0f} 元")
    print(f"  平均单期投入: {avg_bet:.2f} 元")
    print(f"  总盈利: {total_profit:+.0f} 元")
    print(f"  投资回报率(ROI): {roi:+.2f}%")
    print(f"\n风险指标:")
    print(f"  最大连胜: {max_consecutive_wins} 期")
    print(f"  最大连败: {max_consecutive_losses} 期")
    print(f"\n{'='*80}")
    
    # 保存详细结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('backtest_ensemble_top4_auto_resume_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: backtest_ensemble_top4_auto_resume_results.csv")
    
    return {
        'total_profit': total_profit,
        'total_investment': total_investment,
        'roi': roi,
        'hit_rate': hit_rate,
        'hits': hits,
        'actual_betting_periods': actual_betting_periods,
        'paused_periods': paused_periods,
        'max_consecutive_losses': max_consecutive_losses
    }

if __name__ == '__main__':
    backtest_stop_loss_auto_resume(periods=200)
