"""
生肖TOP4稳健动态投注策略 - 200期回测（止损优化版）
投注生肖对应的号码，连续4期失败后暂停投注，直到下次命中再恢复
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from zodiac_enhanced_60_predictor import ZodiacEnhanced60Predictor


class ZodiacTop4StopLossBacktest:
    """生肖TOP4回测验证器 - 止损优化版"""
    
    def __init__(self):
        self.predictor = ZodiacEnhanced60Predictor()
        self.win_reward = 47  # 命中奖励
        self.max_multiplier = 10  # 最大倍数
        
    def calculate_stable_multiplier(self, consecutive_wins, consecutive_losses):
        """
        计算稳健动态投注倍数
        
        连胜时：始终保持1倍
        连败时：快速递增
        """
        base_multiplier = 1.0
        
        # 连胜处理：始终保持1倍
        if consecutive_wins > 0:
            multiplier = base_multiplier
        # 连败处理：快速递增
        elif consecutive_losses >= 3:
            multiplier = 4.0 + (consecutive_losses - 2) * 2
        elif consecutive_losses == 2:
            multiplier = 4.0
        elif consecutive_losses == 1:
            multiplier = 2.0
        else:
            multiplier = base_multiplier
        
        # 限制最大倍数
        multiplier = min(multiplier, self.max_multiplier)
        
        return multiplier
    
    def get_top4_numbers(self, top4_zodiacs):
        """获取TOP4生肖对应的所有数字"""
        all_numbers = []
        for zodiac in top4_zodiacs:
            numbers = self.predictor.zodiac_numbers.get(zodiac, [])
            all_numbers.extend(numbers)
        return sorted(set(all_numbers))
    
    def backtest_200_periods(self, csv_file='data/lucky_numbers.csv'):
        """回测最近200期的投注效果 - 止损优化版"""
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        all_numbers = df['number'].values
        
        if len(all_numbers) < 230:
            print(f"数据不足200期，当前只有{len(all_numbers)}条记录")
            return None
        
        # 使用最近200期进行回测
        test_periods = 200
        start_idx = len(all_numbers) - test_periods
        
        # 初始化统计
        results = []
        total_cost = 0
        total_reward = 0
        total_profit = 0
        hits = 0
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        max_consecutive_wins = 0
        max_drawdown = 0
        current_balance = 0
        
        # 止损相关变量
        is_betting = True  # 是否当前在投注状态
        betting_paused_count = 0  # 暂停投注的期数
        total_paused_periods = 0  # 总暂停期数
        
        print(f"{'='*90}")
        print(f"🎯 生肖TOP4稳健动态投注策略 - 最近200期回测（止损优化版）")
        print(f"{'='*90}\n")
        print(f"📌 策略说明：")
        print(f"  ✓ 预测模型：生肖增强60预测器 (63%命中率)")
        print(f"  ✓ 投注范围：TOP4生肖对应的所有号码（约16个）")
        print(f"  ✓ 动态倍数：连胜保持1倍，连败快速递增")
        print(f"  ✓ 单注奖励：47元")
        print(f"  ✓ 最大倍数：{self.max_multiplier}倍")
        print(f"  ✅ 止损策略：连续4期失败后暂停投注，等待下次命中再恢复\n")
        
        print(f"开始回测...\n")
        
        for i in range(start_idx, len(all_numbers)):
            period_num = i - start_idx + 1
            
            # 使用历史数据预测
            train_data = all_numbers[:i]
            actual = all_numbers[i]
            
            # 预测TOP5生肖，取前4个
            try:
                top5_zodiacs = self.predictor.predict_top5(train_data, recent_periods=100)
                top4_zodiacs = top5_zodiacs[:4]
            except Exception as e:
                print(f"第{period_num}期预测失败: {e}")
                continue
            
            # 获取TOP4对应的所有数字
            bet_numbers = self.get_top4_numbers(top4_zodiacs)
            bet_count = len(bet_numbers)
            
            # 判断是否命中
            is_hit = actual in bet_numbers
            
            # 止损策略逻辑
            bet_amount = 0
            profit = 0
            multiplier = 0
            status = ""
            
            if is_betting:
                # 当前在投注状态
                # 计算稳健动态倍数
                multiplier = self.calculate_stable_multiplier(consecutive_wins, consecutive_losses)
                
                # 计算投注金额
                base_bet = bet_count * 1  # 每个数字1元
                bet_amount = multiplier * base_bet
                
                total_cost += bet_amount
                
                if is_hit:
                    # 命中
                    reward = multiplier * self.win_reward
                    profit = reward - bet_amount
                    
                    total_reward += reward
                    total_profit += profit
                    current_balance += profit
                    
                    hits += 1
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    
                    status = "✅"
                else:
                    # 未中
                    profit = -bet_amount
                    total_profit += profit
                    current_balance += profit
                    
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    
                    # 检查是否触发止损：连续4期失败
                    if consecutive_losses >= 4:
                        is_betting = False
                        betting_paused_count = 0
                        status = "❌ [触发止损，暂停投注]"
                    else:
                        status = "❌"
                
                # 更新最大回撤
                if current_balance < max_drawdown:
                    max_drawdown = current_balance
            else:
                # 暂停投注状态
                betting_paused_count += 1
                total_paused_periods += 1
                
                if is_hit:
                    # 如果这期会命中，恢复投注
                    is_betting = True
                    consecutive_losses = 0
                    consecutive_wins = 0
                    status = f"⏸️ [暂停中-本期会中，恢复投注]"
                else:
                    status = f"⏸️ [暂停投注第{betting_paused_count}期]"
            
            # 记录详细结果
            result = {
                'period': period_num,
                'top4_zodiacs': ', '.join(top4_zodiacs),
                'bet_numbers': ', '.join(map(str, bet_numbers)),
                'bet_count': bet_count,
                'actual': actual,
                'is_hit': '是' if is_hit else '否',
                'is_betting': '是' if (is_betting or bet_amount > 0) else '否',
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'multiplier': multiplier,
                'bet_amount': bet_amount,
                'profit': profit,
                'cumulative_profit': total_profit,
                'balance': current_balance,
                'status': status
            }
            results.append(result)
            
            # 打印关键期数：前20期、后20期、每50期
            if period_num <= 20 or period_num > 180 or period_num % 50 == 0:
                bet_str = f"投注{bet_count:2d}号 倍数={multiplier:.1f}x 投注={bet_amount:6.1f}元" if bet_amount > 0 else "暂停投注                      "
                print(f"第{period_num:3d}期: {status:30} "
                      f"实际={actual:2d} "
                      f"TOP4=[{', '.join(top4_zodiacs)}] "
                      f"{bet_str} "
                      f"盈亏={profit:+7.1f}元 "
                      f"累计={total_profit:+9.1f}元")
        
        # 计算统计指标
        actual_betting_periods = sum(1 for r in results if r['bet_amount'] > 0)
        hit_rate = hits / actual_betting_periods * 100 if actual_betting_periods > 0 else 0
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
        avg_profit_per_period = total_profit / test_periods
        avg_cost_per_period = total_cost / actual_betting_periods if actual_betting_periods > 0 else 0
        
        # 打印汇总统计
        print(f"\n{'='*90}")
        print(f"📊 回测统计汇总 (200期 - 止损优化版)")
        print(f"{'='*90}")
        print(f"测试期数: {test_periods}期")
        print(f"实际投注期数: {actual_betting_periods}期")
        print(f"暂停投注期数: {total_paused_periods}期 ({total_paused_periods/test_periods*100:.2f}%)")
        print(f"命中次数: {hits}次")
        print(f"命中率（实际投注期数）: {hit_rate:.2f}%")
        print(f"最大连胜: {max_consecutive_wins}期")
        print(f"最大连败: {max_consecutive_losses}期")
        
        print(f"\n{'='*90}")
        print(f"💰 收益统计")
        print(f"{'='*90}")
        print(f"总投注: {total_cost:.2f}元")
        print(f"总奖励: {total_reward:.2f}元")
        print(f"净收益: {total_profit:+.2f}元 {'🎉profit!' if total_profit > 0 else '⚠️loss'}")
        print(f"投资回报率(ROI): {roi:+.2f}%")
        print(f"平均每期投注: {avg_cost_per_period:.2f}元")
        print(f"平均每期盈亏: {avg_profit_per_period:+.2f}元")
        
        print(f"\n{'='*90}")
        print(f"📉 风险指标")
        print(f"{'='*90}")
        print(f"最大回撤: {max_drawdown:.2f}元")
        print(f"最终余额: {current_balance:+.2f}元")
        
        # 分析倍数分布
        multiplier_distribution = defaultdict(int)
        for r in results:
            if r['bet_amount'] > 0:
                multiplier_distribution[r['multiplier']] += 1
        
        print(f"\n{'='*90}")
        print(f"📈 倍数分布统计")
        print(f"{'='*90}")
        for mult in sorted(multiplier_distribution.keys()):
            count = multiplier_distribution[mult]
            pct = count / actual_betting_periods * 100 if actual_betting_periods > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"{mult:4.1f}倍: {count:3d}期 ({pct:5.2f}%) {bar}")
        
        # 按100期分段统计
        print(f"\n{'='*90}")
        print(f"📊 分段统计 (每100期)")
        print(f"{'='*90}")
        print(f"{'期数':<15} {'投注期':<10} {'命中率':<12} {'投注成本':<15} {'净收益':<15} {'ROI':<10}")
        print(f"{'-'*90}")
        
        for segment in range(0, test_periods, 100):
            segment_end = min(segment + 100, test_periods)
            segment_results = results[segment:segment_end]
            
            segment_betting = sum(1 for r in segment_results if r['bet_amount'] > 0)
            segment_hits = sum(1 for r in segment_results if r['is_hit'] == '是' and r['bet_amount'] > 0)
            segment_cost = sum(r['bet_amount'] for r in segment_results)
            segment_profit = sum(r['profit'] for r in segment_results)
            segment_hit_rate = segment_hits / segment_betting * 100 if segment_betting > 0 else 0
            segment_roi = (segment_profit / segment_cost * 100) if segment_cost > 0 else 0
            
            print(f"第{segment+1:3d}-{segment_end:3d}期  "
                  f"{segment_betting}期      "
                  f"{segment_hit_rate:5.2f}%      "
                  f"{segment_cost:8.2f}元     "
                  f"{segment_profit:+9.2f}元     "
                  f"{segment_roi:+6.2f}%")
        
        # 对比原策略（无止损）
        print(f"\n{'='*90}")
        print(f"📊 策略对比（止损优化 vs 原策略无止损）")
        print(f"{'='*90}")
        
        # 计算原策略数据（从之前的200期回测结果）
        # 原策略：命中率45%，总投注9645元，净收益+1400元，ROI +14.52%
        original_cost = 9645.00
        original_profit = 1400.00
        original_roi = 14.52
        original_betting_periods = 200
        
        print(f"{'指标':<20} {'止损优化策略':<25} {'原策略（无止损）':<25} {'差异':<15}")
        print(f"{'-'*90}")
        print(f"{'投注期数':<20} {actual_betting_periods:<25} {original_betting_periods:<25} {actual_betting_periods - original_betting_periods:<15}")
        print(f"{'总投注':<20} {total_cost:.2f}元{'':<18} {original_cost:.2f}元{'':<18} {total_cost - original_cost:+.2f}元")
        print(f"{'净收益':<20} {total_profit:+.2f}元{'':<18} {original_profit:+.2f}元{'':<18} {total_profit - original_profit:+.2f}元")
        print(f"{'ROI':<20} {roi:+.2f}%{'':<20} {original_roi:+.2f}%{'':<20} {roi - original_roi:+.2f}%")
        
        improvement = total_profit - original_profit
        print(f"\n💡 止损策略{'提升' if improvement > 0 else '降低'}收益: {improvement:+.2f}元 ({improvement/abs(original_profit)*100:+.2f}%)")
        
        # 保存详细记录到CSV
        df_results = pd.DataFrame(results)
        output_file = 'zodiac_top4_backtest_200periods_stop_loss.csv'
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*90}")
        print(f"✅ 回测完成！详细记录已保存至: {output_file}")
        print(f"{'='*90}\n")
        
        return {
            'hit_rate': hit_rate,
            'total_cost': total_cost,
            'total_reward': total_reward,
            'total_profit': total_profit,
            'roi': roi,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'max_drawdown': max_drawdown,
            'results': results,
            'actual_betting_periods': actual_betting_periods,
            'total_paused_periods': total_paused_periods,
            'improvement': improvement
        }


def main():
    """主函数"""
    from datetime import datetime
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tester = ZodiacTop4StopLossBacktest()
    result = tester.backtest_200_periods()
    
    if result:
        print(f"\n🎯 回测总结（止损优化版）:")
        print(f"  命中率: {result['hit_rate']:.2f}%")
        print(f"  实际投注: {result['actual_betting_periods']}期（暂停{result['total_paused_periods']}期）")
        print(f"  总投注: {result['total_cost']:.2f}元")
        print(f"  净收益: {result['total_profit']:+.2f}元")
        print(f"  ROI: {result['roi']:+.2f}%")
        print(f"  vs 原策略: {result['improvement']:+.2f}元 ({'提升' if result['improvement'] > 0 else '降低'})")
        print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
