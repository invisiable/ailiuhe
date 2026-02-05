"""
生肖TOP4动态投注策略验证
基于连胜连败的智能倍投系统

投注规则：
- 购买TOP4生肖对应的所有数字（约16个号码）
- 每个数字1元
- 命中奖励: 45元
- 未命中亏损: 投注成本

动态策略：
- 连续命中：减少倍数（止盈）
- 连续失败1期：倍数+1
- 连续失败2期：倍数+3（加速回本）
- 连续失败3期+：每期再+2
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from ensemble_zodiac_predictor import EnsembleZodiacPredictor


class ZodiacTop4DynamicBetting:
    """生肖TOP4动态投注验证器"""
    
    def __init__(self):
        self.predictor = EnsembleZodiacPredictor()
        self.win_reward = 47  # 命中奖励
        self.max_multiplier = 10  # 最大倍数
        
    def calculate_dynamic_multiplier(self, consecutive_wins, consecutive_losses):
        """
        计算动态投注倍数
        
        Args:
            consecutive_wins: 连续命中次数
            consecutive_losses: 连续失败次数
            
        Returns:
            投注倍数
        """
        base_multiplier = 1.0
        
        # 连胜处理：每连胜1期，倍数减少0.5
        if consecutive_wins > 0:
            reduction = consecutive_wins * 0.5
            multiplier = max(0.5, base_multiplier - reduction)
        
        # 连败处理
        elif consecutive_losses > 0:
            if consecutive_losses == 1:
                multiplier = base_multiplier + 1  # 2倍
            elif consecutive_losses == 2:
                multiplier = base_multiplier + 3  # 4倍
            else:
                multiplier = base_multiplier + 3 + (consecutive_losses - 2) * 2
        
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
    
    def validate_100_periods(self, csv_file='data/lucky_numbers.csv'):
        """
        验证最近100期的投注效果
        
        Args:
            csv_file: 数据文件路径
            
        Returns:
            详细的投注记录和统计结果
        """
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        all_numbers = df['number'].values
        animals = [str(a).strip() for a in df['animal'].values]
        
        if len(all_numbers) < 130:
            print("数据不足100期，无法验证")
            return None
        
        # 使用最近100期进行验证
        test_periods = 100
        start_idx = len(all_numbers) - test_periods
        
        # 初始化统计
        results = []
        total_cost = 0
        total_reward = 0
        total_profit = 0
        hits = 0
        animal_hits = 0
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        max_drawdown = 0
        current_balance = 0
        
        print(f"{'='*80}")
        print(f"生肖TOP4动态投注策略 - 最近100期验证报告")
        print(f"{'='*80}\n")
        
        for i in range(start_idx, len(all_numbers)):
            period_num = i - start_idx + 1
            
            # 使用历史数据预测
            train_animals = animals[:i]
            actual = all_numbers[i]

            if len(train_animals) == 0:
                continue

            # 预测TOP5生肖，取前4个（与GUI保持一致）
            try:
                prediction = self.predictor.predict_from_history(train_animals, top_n=5, debug=False)
                top4_zodiacs = prediction['top4']
            except Exception as e:
                print(f"第{period_num}期预测失败: {e}")
                continue
            
            # 获取TOP4对应的所有数字
            bet_numbers = self.get_top4_numbers(top4_zodiacs)
            bet_count = len(bet_numbers)
            
            # 计算动态倍数
            multiplier = self.calculate_dynamic_multiplier(consecutive_wins, consecutive_losses)
            
            # 计算投注金额
            base_bet = bet_count * 1  # 每个数字1元
            bet_amount = multiplier * base_bet
            
            total_cost += bet_amount
            
            # 判断是否命中
            hit_number = actual in bet_numbers
            hit_animal = animals[i] in top4_zodiacs
            
            if hit_number:
                # 命中
                reward = multiplier * self.win_reward
                profit = reward - bet_amount
                
                total_reward += reward
                total_profit += profit
                current_balance += profit
                
                hits += 1
                consecutive_wins += 1
                consecutive_losses = 0
                
                result_icon = "✅"
            else:
                # 未中
                profit = -bet_amount
                total_profit += profit
                current_balance += profit
                
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                
                result_icon = "❌"
            
            if hit_animal:
                animal_hits += 1

            # 更新最大回撤
            if current_balance < max_drawdown:
                max_drawdown = current_balance
            
            # 记录详细结果
            current_row = df.iloc[i]
            result = {
                'period': period_num,
                'global_period': i + 1,
                'date': current_row['date'],
                'top4_zodiacs': top4_zodiacs,
                'bet_numbers': bet_numbers,
                'bet_count': bet_count,
                'actual': actual,
                'actual_animal': animals[i],
                'is_hit': hit_number,
                'is_hit_number': hit_number,
                'is_hit_animal': hit_animal,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'multiplier': multiplier,
                'bet_amount': bet_amount,
                'profit': profit,
                'cumulative_profit': total_profit,
                'balance': current_balance,
                'model_used': prediction.get('selected_model', '集成生肖投票模型')
            }
            results.append(result)
            
            # 打印前10期和后10期的详细信息
            if period_num <= 10 or period_num > 90:
                print(f"第{period_num:3d}期: {result_icon} "
                      f"TOP4={top4_zodiacs} "
                      f"投注{bet_count:2d}个号 "
                      f"实际={actual:2d} "
                      f"倍数={multiplier:.1f}x "
                      f"投注={bet_amount:6.1f}元 "
                      f"盈亏={profit:+7.1f}元 "
                      f"累计={total_profit:+8.1f}元 "
                      f"连胜={consecutive_wins} 连败={consecutive_losses}")
        
        # 计算统计指标
        hit_rate = hits / test_periods * 100
        animal_hit_rate = animal_hits / test_periods * 100
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
        avg_profit_per_period = total_profit / test_periods
        
        # 打印汇总统计
        print(f"\n{'='*80}")
        print(f"📊 投注统计汇总")
        print(f"{'='*80}")
        print(f"测试期数: {test_periods}期")
        print(f"命中次数(号码): {hits}次")
        print(f"命中率(号码): {hit_rate:.2f}%")
        print(f"命中次数(生肖): {animal_hits}次")
        print(f"命中率(生肖): {animal_hit_rate:.2f}%")
        print(f"\n{'='*80}")
        print(f"💰 收益统计")
        print(f"{'='*80}")
        print(f"总投注: {total_cost:.2f}元")
        print(f"总奖励: {total_reward:.2f}元")
        print(f"净收益: {total_profit:+.2f}元")
        print(f"投资回报率(ROI): {roi:+.2f}%")
        print(f"平均每期盈亏: {avg_profit_per_period:+.2f}元")
        print(f"\n{'='*80}")
        print(f"📉 风险指标")
        print(f"{'='*80}")
        print(f"最大连败: {max_consecutive_losses}期")
        print(f"最大回撤: {max_drawdown:.2f}元")
        
        # 分析倍数分布
        multiplier_distribution = defaultdict(int)
        for r in results:
            multiplier_distribution[r['multiplier']] += 1
        
        print(f"\n{'='*80}")
        print(f"📈 倍数分布统计")
        print(f"{'='*80}")
        for mult in sorted(multiplier_distribution.keys()):
            count = multiplier_distribution[mult]
            pct = count / test_periods * 100
            print(f"{mult:.1f}倍: {count:3d}期 ({pct:5.2f}%)")
        
        # 保存详细记录到CSV
        df_results = pd.DataFrame(results)
        output_file = 'zodiac_top4_dynamic_betting_100periods.csv'
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细记录已保存至: {output_file}")
        
        return {
            'hit_rate': hit_rate,
            'animal_hit_rate': animal_hit_rate,
            'total_cost': total_cost,
            'total_reward': total_reward,
            'total_profit': total_profit,
            'roi': roi,
            'max_consecutive_losses': max_consecutive_losses,
            'max_drawdown': max_drawdown,
            'animal_hits': animal_hits,
            'results': results
        }
    
    def compare_strategies(self, csv_file='data/lucky_numbers.csv'):
        """
        对比不同投注策略的效果
        
        对比：
        1. 固定1倍投注
        2. 动态投注（本策略）
        3. 马丁格尔倍投
        """
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        all_numbers = df['number'].values
        all_animals = [str(a).strip() for a in df['animal'].values]
        
        if len(all_numbers) < 130:
            print("数据不足，无法对比")
            return
        
        test_periods = 100
        start_idx = len(all_numbers) - test_periods
        
        # 策略1: 固定1倍
        stats_fixed = self._simulate_fixed_bet(all_numbers, all_animals, start_idx)
        
        # 策略2: 动态投注
        stats_dynamic = self._simulate_dynamic_bet(all_numbers, all_animals, start_idx)
        
        # 策略3: 马丁格尔
        stats_martingale = self._simulate_martingale_bet(all_numbers, all_animals, start_idx)
        
        # 打印对比
        print(f"\n{'='*80}")
        print(f"📊 投注策略对比分析 (最近100期)")
        print(f"{'='*80}\n")
        
        strategies = [
            ('固定1倍投注', stats_fixed),
            ('动态智能投注', stats_dynamic),
            ('马丁格尔倍投', stats_martingale)
        ]
        
        print(f"{'策略名称':<20} {'命中率':>10} {'总投注':>12} {'净收益':>12} {'ROI':>10} {'最大回撤':>12}")
        print(f"{'-'*80}")
        
        for name, stats in strategies:
            print(f"{name:<20} "
                  f"{stats['hit_rate']:>9.2f}% "
                  f"{stats['total_cost']:>11.2f}元 "
                  f"{stats['total_profit']:>+11.2f}元 "
                  f"{stats['roi']:>+9.2f}% "
                  f"{stats['max_drawdown']:>11.2f}元")
        
        # 推荐策略
        best_strategy = max(strategies, key=lambda x: x[1]['total_profit'])
        print(f"\n⭐ 推荐策略: {best_strategy[0]}")
        print(f"   净收益: {best_strategy[1]['total_profit']:+.2f}元")
        print(f"   ROI: {best_strategy[1]['roi']:+.2f}%")
    
    def _simulate_fixed_bet(self, all_numbers, all_animals, start_idx):
        """模拟固定1倍投注"""
        total_cost = 0
        total_profit = 0
        hits = 0
        max_drawdown = 0
        current_balance = 0
        
        test_periods = len(all_numbers) - start_idx
        
        for i in range(start_idx, len(all_numbers)):
            train_animals = all_animals[:i]
            actual = all_numbers[i]

            if len(train_animals) == 0:
                continue

            try:
                prediction = self.predictor.predict_from_history(train_animals, top_n=5, debug=False)
                top4_zodiacs = prediction['top4']
                bet_numbers = self.get_top4_numbers(top4_zodiacs)
                
                bet_amount = len(bet_numbers) * 1.0
                total_cost += bet_amount
                
                if actual in bet_numbers:
                    profit = self.win_reward - bet_amount
                    hits += 1
                else:
                    profit = -bet_amount
                
                total_profit += profit
                current_balance += profit
                max_drawdown = min(max_drawdown, current_balance)
            except:
                continue
        
        return {
            'hit_rate': hits / test_periods * 100,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'roi': (total_profit / total_cost * 100) if total_cost > 0 else 0,
            'max_drawdown': max_drawdown
        }
    
    def _simulate_dynamic_bet(self, all_numbers, all_animals, start_idx):
        """模拟动态投注"""
        total_cost = 0
        total_profit = 0
        hits = 0
        consecutive_wins = 0
        consecutive_losses = 0
        max_drawdown = 0
        current_balance = 0
        
        test_periods = len(all_numbers) - start_idx
        
        for i in range(start_idx, len(all_numbers)):
            train_animals = all_animals[:i]
            actual = all_numbers[i]

            if len(train_animals) == 0:
                continue

            try:
                prediction = self.predictor.predict_from_history(train_animals, top_n=5, debug=False)
                top4_zodiacs = prediction['top4']
                bet_numbers = self.get_top4_numbers(top4_zodiacs)
                
                multiplier = self.calculate_dynamic_multiplier(consecutive_wins, consecutive_losses)
                bet_amount = multiplier * len(bet_numbers) * 1.0
                total_cost += bet_amount
                
                if actual in bet_numbers:
                    profit = multiplier * self.win_reward - bet_amount
                    hits += 1
                    consecutive_wins += 1
                    consecutive_losses = 0
                else:
                    profit = -bet_amount
                    consecutive_losses += 1
                    consecutive_wins = 0
                
                total_profit += profit
                current_balance += profit
                max_drawdown = min(max_drawdown, current_balance)
            except:
                continue
        
        return {
            'hit_rate': hits / test_periods * 100,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'roi': (total_profit / total_cost * 100) if total_cost > 0 else 0,
            'max_drawdown': max_drawdown
        }
    
    def _simulate_martingale_bet(self, all_numbers, all_animals, start_idx):
        """模拟马丁格尔倍投"""
        total_cost = 0
        total_profit = 0
        hits = 0
        consecutive_losses = 0
        max_drawdown = 0
        current_balance = 0
        
        test_periods = len(all_numbers) - start_idx
        
        for i in range(start_idx, len(all_numbers)):
            train_animals = all_animals[:i]
            actual = all_numbers[i]

            if len(train_animals) == 0:
                continue

            try:
                prediction = self.predictor.predict_from_history(train_animals, top_n=5, debug=False)
                top4_zodiacs = prediction['top4']
                bet_numbers = self.get_top4_numbers(top4_zodiacs)
                
                # 马丁格尔：连败时倍数翻倍
                multiplier = min(2 ** consecutive_losses, self.max_multiplier)
                bet_amount = multiplier * len(bet_numbers) * 1.0
                total_cost += bet_amount
                
                if actual in bet_numbers:
                    profit = multiplier * self.win_reward - bet_amount
                    hits += 1
                    consecutive_losses = 0
                else:
                    profit = -bet_amount
                    consecutive_losses += 1
                
                total_profit += profit
                current_balance += profit
                max_drawdown = min(max_drawdown, current_balance)
            except:
                continue
        
        return {
            'hit_rate': hits / test_periods * 100,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'roi': (total_profit / total_cost * 100) if total_cost > 0 else 0,
            'max_drawdown': max_drawdown
        }


def main():
    """主函数"""
    validator = ZodiacTop4DynamicBetting()
    
    print("开始验证生肖TOP4动态投注策略...\n")
    
    # 1. 验证最近100期
    result = validator.validate_100_periods()
    
    if result:
        # 2. 对比不同策略
        print(f"\n{'='*80}\n")
        validator.compare_strategies()
    
    print(f"\n{'='*80}")
    print("验证完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
