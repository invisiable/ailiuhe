"""
生肖TOP4投注策略分析 - 集成优化系统（200期回测）
使用 EnsembleZodiacPredictor 进行生肖预测，直接投注生肖
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from ensemble_zodiac_predictor import EnsembleZodiacPredictor
from datetime import datetime


class EnsembleTop4Backtest:
    """集成预测器TOP4回测验证器"""
    
    def __init__(self):
        self.predictor = EnsembleZodiacPredictor()
        
    def backtest_200_periods(self, csv_file='data/lucky_numbers.csv'):
        """回测最近200期的投注效果"""
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        if len(df) < 230:
            print(f"数据不足200期，当前只有{len(df)}条记录")
            return None
        
        # 使用最近200期进行回测
        test_periods = 200
        start_idx = len(df) - test_periods
        
        print(f"{'='*90}")
        print(f"🎯 生肖TOP4投注策略分析 - 集成优化系统（200期回测）")
        print(f"{'='*90}\n")
        print(f"📌 策略说明：")
        print(f"  ✓ 预测模型：集成预测器 (v10 + 优化版投票)")
        print(f"  ✓ 投注方式：直接投注TOP4生肖（不是号码）")
        print(f"  ✓ 基本倍投：20倍")
        print(f"  ✓ 每期投入：320元 (每个生肖80元 × 4个生肖)")
        print(f"  ✓ 命中奖励：940元 (47元 × 20倍)")
        print(f"  ✓ 净利润：620元 (940 - 320)")
        print(f"  ✓ 未命中亏损：-320元\n")
        
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
            
            # 判断命中（生肖是否在TOP4中）
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
            
            # 固定投注：每期320元（20倍）
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
                
                result_icon = "✅"
            else:
                # 未中
                profit = -320
                total_profit += profit
                current_balance += profit
                monthly_profits[month_key] += profit
                
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                
                result_icon = "❌"
            
            # 记录详细结果
            result = {
                'period': period_num,
                'date': date_str,
                'actual_number': actual_number,
                'actual_animal': actual_animal,
                'top4_prediction': ', '.join(top4),
                'is_hit': '是' if is_hit else '否',
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'bet_amount': bet_amount,
                'profit': profit,
                'cumulative_profit': total_profit,
                'balance': current_balance
            }
            results.append(result)
            
            # 打印关键期数
            if period_num <= 20 or period_num > 180 or period_num % 50 == 0:
                print(f"第{period_num:3d}期 {date_str:12} {result_icon} "
                      f"实际={actual_number:2d}号({actual_animal}) "
                      f"TOP4=[{', '.join(top4)}] "
                      f"投注=320元 "
                      f"盈亏={profit:+6.0f}元 "
                      f"累计={total_profit:+8.0f}元 "
                      f"连胜={consecutive_wins} 连败={consecutive_losses}")
        
        # 计算统计指标
        hit_rate = hits / test_periods * 100
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
        avg_profit_per_period = total_profit / test_periods
        
        # 打印汇总统计
        print(f"\n{'='*90}")
        print(f"📊 回测统计汇总 (200期)")
        print(f"{'='*90}")
        print(f"测试期数: {test_periods}期")
        print(f"命中次数: {hits}次")
        print(f"命中率: {hit_rate:.2f}%")
        print(f"最大连胜: {max_consecutive_wins}期")
        print(f"最大连败: {max_consecutive_losses}期")
        
        print(f"\n{'='*90}")
        print(f"💰 收益统计")
        print(f"{'='*90}")
        print(f"总投注: {total_cost:.2f}元")
        print(f"总奖励: {total_reward:.2f}元")
        print(f"净收益: {total_profit:+.2f}元 {'🎉profit!' if total_profit > 0 else '⚠️loss'}")
        print(f"投资回报率(ROI): {roi:+.2f}%")
        print(f"平均每期投注: 320.00元")
        print(f"平均每期盈亏: {avg_profit_per_period:+.2f}元")
        
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
        print(f"{'期数':<15} {'命中率':<12} {'投注成本':<15} {'净收益':<15} {'ROI':<10}")
        print(f"{'-'*90}")
        
        for segment in range(0, test_periods, 100):
            segment_end = min(segment + 100, test_periods)
            segment_results = results[segment:segment_end]
            
            segment_hits = sum(1 for r in segment_results if r['is_hit'] == '是')
            segment_cost = sum(r['bet_amount'] for r in segment_results)
            segment_profit = sum(r['profit'] for r in segment_results)
            segment_hit_rate = segment_hits / len(segment_results) * 100
            segment_roi = (segment_profit / segment_cost * 100) if segment_cost > 0 else 0
            
            print(f"第{segment+1:3d}-{segment_end:3d}期  "
                  f"{segment_hit_rate:5.2f}%      "
                  f"{segment_cost:8.2f}元     "
                  f"{segment_profit:+9.2f}元     "
                  f"{segment_roi:+6.2f}%")
        
        # 分析命中位置分布
        print(f"\n{'='*90}")
        print(f"📈 命中位置分布（当命中时，实际生肖在TOP4中的位置）")
        print(f"{'='*90}")
        position_hits = {1: 0, 2: 0, 3: 0, 4: 0}
        for i, is_hit in enumerate(hit_records):
            if is_hit:
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
        output_file = 'ensemble_top4_backtest_200periods.csv'
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
            'monthly_profits': monthly_profits
        }


def main():
    """主函数"""
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tester = EnsembleTop4Backtest()
    result = tester.backtest_200_periods()
    
    if result:
        print(f"\n🎯 回测总结:")
        print(f"  命中率: {result['hit_rate']:.2f}%")
        print(f"  总投注: {result['total_cost']:.2f}元")
        print(f"  净收益: {result['total_profit']:+.2f}元")
        print(f"  ROI: {result['roi']:+.2f}%")
        print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
