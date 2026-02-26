"""
生肖TOP4投注策略分析 - 集成优化系统（200期回测 - 止损优化版）
使用 EnsembleZodiacPredictor 进行生肖预测，直接投注生肖
优化逻辑：连续4期失败后停止投注，直到下次命中再恢复投注
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from ensemble_zodiac_predictor import EnsembleZodiacPredictor
from datetime import datetime


class EnsembleTop4StopLossBacktest:
    """集成预测器TOP4回测验证器 - 止损优化版"""
    
    def __init__(self):
        self.predictor = EnsembleZodiacPredictor()
        
    def backtest_200_periods(self, csv_file='data/lucky_numbers.csv'):
        """回测最近200期的投注效果 - 止损优化版"""
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        if len(df) < 230:
            print(f"数据不足200期，当前只有{len(df)}条记录")
            return None
        
        # 使用最近200期进行回测
        test_periods = 200
        start_idx = len(df) - test_periods
        
        print(f"{'='*90}")
        print(f"🎯 生肖TOP4投注策略分析 - 集成优化系统（200期回测 - 止损优化版）")
        print(f"{'='*90}\n")
        print(f"📌 策略说明：")
        print(f"  ✓ 预测模型：集成预测器 (v10 + 优化版投票)")
        print(f"  ✓ 投注方式：直接投注TOP4生肖（不是号码）")
        print(f"  ✓ 基本倍投：20倍")
        print(f"  ✓ 每期投入：320元 (每个生肖80元 × 4个生肖)")
        print(f"  ✓ 命中奖励：940元 (47元 × 20倍)")
        print(f"  ✓ 净利润：620元 (940 - 320)")
        print(f"  ✓ 未命中亏损：-320元")
        print(f"  ✅ 止损策略：连续4期失败后停止投注，等待下次命中再恢复\n")
        
        print(f"开始回测...\n")
        
        # 初始化统计
        results = []
        total_cost = 0
        total_reward = 0
        total_profit = 0
        hits = 0
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_balance = 0
        monthly_profits = {}
        
        # 止损相关变量
        is_betting = True  # 是否当前在投注状态
        betting_paused_count = 0  # 暂停投注的期数
        total_paused_periods = 0  # 总暂停期数
        
        # 生成预测
        predictions_top4 = []
        actuals = []
        hit_records = []
        
        print("生成历史TOP4生肖预测...\n")
        
        for i in range(start_idx, len(df)):
            period_num = i - start_idx + 1
            
            # 使用i之前的数据进行预测
            train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
            
            try:
                # 使用集成预测器进行预测
                result = self.predictor.predict_from_history(train_animals, top_n=5, debug=False)
                top4 = result['top4']  # 直接取TOP4生肖
            except Exception as e:
                print(f"第{period_num}期预测失败: {e}")
                continue
            
            predictions_top4.append(top4)
            
            # 实际结果（生肖）
            actual_animal = str(df.iloc[i]['animal']).strip()
            actual_number = df.iloc[i]['number']
            date_str = df.iloc[i]['date']
            actuals.append(actual_animal)
            
            # 判断是否命中（生肖是否在TOP4中）
            is_hit = actual_animal in top4
            hit_records.append(is_hit)
            
            # 获取年月用于月度统计
            try:
                date_obj = pd.to_datetime(date_str)
                month_key = date_obj.strftime('%Y/%m')
            except:
                month_key = date_str[:7] if len(date_str) >= 7 else '未知'
            
            if month_key not in monthly_profits:
                monthly_profits[month_key] = 0
            
            # 止损策略逻辑
            bet_amount = 0
            profit = 0
            status = ""
            
            if is_betting:
                # 当前在投注状态
                bet_amount = 320
                total_cost += bet_amount
                
                if is_hit:
                    # 命中
                    reward = 940  # 47元 × 20倍
                    profit = 620  # 净利润
                    
                    total_reward += reward
                    total_profit += profit
                    current_balance += profit
                    monthly_profits[month_key] += profit
                    
                    hits += 1
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    
                    status = "✅"
                else:
                    # 未中
                    profit = -320
                    total_profit += profit
                    current_balance += profit
                    monthly_profits[month_key] += profit
                    
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
            result_record = {
                'period': period_num,
                'date': date_str,
                'actual_number': actual_number,
                'actual_animal': actual_animal,
                'top4_prediction': ', '.join(top4),
                'is_hit': '是' if is_hit else '否',
                'is_betting': '是' if is_betting or (not is_betting and bet_amount > 0) else '否',
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'bet_amount': bet_amount,
                'profit': profit,
                'cumulative_profit': total_profit,
                'balance': current_balance,
                'status': status
            }
            results.append(result_record)
            
            # 打印关键期数
            if period_num <= 20 or period_num > 180 or period_num % 50 == 0:
                bet_str = f"投注={bet_amount:3.0f}元" if bet_amount > 0 else "暂停投注  "
                print(f"第{period_num:3d}期 {date_str:12} {status:30} "
                      f"实际={actual_number:2d}号({actual_animal}) "
                      f"TOP4=[{', '.join(top4)}] "
                      f"{bet_str} "
                      f"盈亏={profit:+6.0f}元 "
                      f"累计={total_profit:+8.0f}元")
        
        # 计算统计指标
        actual_betting_periods = sum(1 for r in results if r['bet_amount'] > 0)
        hit_rate = hits / actual_betting_periods * 100 if actual_betting_periods > 0 else 0
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
        avg_profit_per_period = total_profit / test_periods
        avg_profit_per_betting_period = total_profit / actual_betting_periods if actual_betting_periods > 0 else 0
        
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
        print(f"平均每期盈亏: {avg_profit_per_period:+.2f}元")
        print(f"平均每个投注期盈亏: {avg_profit_per_betting_period:+.2f}元")
        
        print(f"\n{'='*90}")
        print(f"📉 风险指标")
        print(f"{'='*90}")
        print(f"最终余额: {current_balance:+.2f}元")
        
        # 每月收益统计
        print(f"\n{'='*90}")
        print(f"📊 每月收益统计")
        print(f"{'='*90}")
        for month in sorted(monthly_profits.keys()):
            profit = monthly_profits[month]
            print(f"{month}: {profit:+10.2f}元")
        
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
        
        # 对比原策略
        print(f"\n{'='*90}")
        print(f"📊 策略对比（止损优化 vs 原策略）")
        print(f"{'='*90}")
        
        # 计算原策略数据（假设全部投注）
        original_hits = sum(hit_records)
        original_cost = 320 * test_periods
        original_reward = 940 * original_hits
        original_profit = original_reward - original_cost
        original_roi = (original_profit / original_cost * 100)
        
        print(f"{'指标':<20} {'止损优化策略':<20} {'原策略（全投）':<20} {'差异':<15}")
        print(f"{'-'*90}")
        print(f"{'投注期数':<20} {actual_betting_periods:<20} {test_periods:<20} {actual_betting_periods - test_periods:<15}")
        print(f"{'总投注':<20} {total_cost:.2f}元{'':<13} {original_cost:.2f}元{'':<13} {total_cost - original_cost:+.2f}元")
        print(f"{'净收益':<20} {total_profit:+.2f}元{'':<13} {original_profit:+.2f}元{'':<13} {total_profit - original_profit:+.2f}元")
        print(f"{'ROI':<20} {roi:+.2f}%{'':<15} {original_roi:+.2f}%{'':<15} {roi - original_roi:+.2f}%")
        
        improvement = total_profit - original_profit
        print(f"\n💡 止损策略{'提升' if improvement > 0 else '降低'}收益: {improvement:+.2f}元 ({improvement/abs(original_profit)*100:+.2f}%)")
        
        # 分析命中位置分布
        print(f"\n{'='*90}")
        print(f"📈 命中位置分布（当命中时，实际生肖在TOP4中的位置）")
        print(f"{'='*90}")
        position_hits = {1: 0, 2: 0, 3: 0, 4: 0}
        for i, result_record in enumerate(results):
            if result_record['is_hit'] == '是' and result_record['bet_amount'] > 0:
                actual_animal = actuals[i]
                predicted_top4 = predictions_top4[i]
                if actual_animal in predicted_top4:
                    position = predicted_top4.index(actual_animal) + 1
                    position_hits[position] += 1
        
        for pos, count in position_hits.items():
            rate = (count / hits * 100) if hits > 0 else 0
            bar = '█' * int(rate / 5)
            print(f"TOP{pos}: {count:3d}次 ({rate:5.2f}%) {bar}")
        
        # 保存详细记录到CSV
        df_results = pd.DataFrame(results)
        output_file = 'ensemble_top4_backtest_200periods_stop_loss.csv'
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
            'results': results,
            'monthly_profits': monthly_profits,
            'actual_betting_periods': actual_betting_periods,
            'total_paused_periods': total_paused_periods,
            'original_profit': original_profit,
            'original_roi': original_roi,
            'improvement': improvement
        }


def main():
    """主函数"""
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tester = EnsembleTop4StopLossBacktest()
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
