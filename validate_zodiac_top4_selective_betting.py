"""
生肖TOP4选择性动态投注策略验证
更保守的智能倍投系统

投注规则：
- 默认基础倍投1倍
- 连续失败2期才开始加倍（更谨慎）
- 连续成功2期才减少投注
- 可选择性跳过某些期数不投注

动态策略：
- 初始/连败1期/连胜1期：1倍投注
- 连续成功2期：0.8倍（略微保守）
- 连续成功3期+：0.5倍（大幅保守）
- 连续失败2期：2倍追回
- 连续失败3期：4倍加速
- 连续失败4期+：每期再+2倍
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from zodiac_enhanced_60_predictor import ZodiacEnhanced60Predictor


class ZodiacTop4SelectiveBetting:
    """生肖TOP4选择性动态投注验证器"""
    
    def __init__(self):
        self.predictor = ZodiacEnhanced60Predictor()
        self.win_reward = 47  # 命中奖励
        self.max_multiplier = 10  # 最大倍数
        
    def calculate_selective_multiplier(self, consecutive_wins, consecutive_losses):
        """
        计算选择性动态投注倍数
        
        Args:
            consecutive_wins: 连续命中次数
            consecutive_losses: 连续失败次数
            
        Returns:
            (投注倍数, 是否投注)
        """
        base_multiplier = 1.0
        should_bet = True
        
        # 连胜处理：需要连续2期成功才减少投注
        if consecutive_wins >= 3:
            multiplier = 0.5  # 连胜3期+：大幅保守
        elif consecutive_wins == 2:
            multiplier = 0.8  # 连胜2期：略微保守
        elif consecutive_wins == 1:
            multiplier = base_multiplier  # 保持基础
        
        # 连败处理：需要连续2期失败才开始加倍
        elif consecutive_losses >= 4:
            multiplier = 4.0 + (consecutive_losses - 3) * 2
        elif consecutive_losses == 3:
            multiplier = 4.0
        elif consecutive_losses == 2:
            multiplier = 2.0  # 连败2期才开始加倍
        elif consecutive_losses == 1:
            multiplier = base_multiplier  # 首次失败保持基础
        
        else:
            multiplier = base_multiplier
        
        # 限制最大倍数
        multiplier = min(multiplier, self.max_multiplier)
        
        return multiplier, should_bet
    
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
        skipped_periods = 0
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        max_drawdown = 0
        current_balance = 0
        
        print(f"{'='*80}")
        print(f"生肖TOP4选择性动态投注策略 - 最近100期验证报告")
        print(f"{'='*80}\n")
        print(f"策略特点：")
        print(f"  ✓ 连续失败2期才开始加倍（更谨慎）")
        print(f"  ✓ 连续成功2期才减少投注")
        print(f"  ✓ 失败1期保持1倍投注观察\n")
        
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
            
            # 计算选择性动态倍数
            multiplier, should_bet = self.calculate_selective_multiplier(consecutive_wins, consecutive_losses)
            
            # 判断是否投注
            if not should_bet:
                skipped_periods += 1
                result = {
                    'period': period_num,
                    'top4_zodiacs': top4_zodiacs,
                    'bet_numbers': bet_numbers,
                    'bet_count': bet_count,
                    'actual': actual,
                    'is_hit': False,
                    'consecutive_wins': consecutive_wins,
                    'consecutive_losses': consecutive_losses,
                    'multiplier': 0,
                    'bet_amount': 0,
                    'profit': 0,
                    'cumulative_profit': total_profit,
                    'balance': current_balance,
                    'skipped': True
                }
                results.append(result)
                
                # 打印跳过信息
                if period_num <= 10 or period_num > 90:
                    print(f"第{period_num:3d}期: ⏭️  "
                          f"TOP4={top4_zodiacs} "
                          f"跳过投注（观望期） "
                          f"实际={actual:2d} "
                          f"累计={total_profit:+8.1f}元")
                continue
            
            # 计算投注金额
            base_bet = bet_count * 1  # 每个数字1元
            bet_amount = multiplier * base_bet
            
            total_cost += bet_amount
            
            # 判断是否命中
            is_hit = actual in bet_numbers
            
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
            
            # 更新最大回撤
            if current_balance < max_drawdown:
                max_drawdown = current_balance
            
            # 记录详细结果
            result = {
                'period': period_num,
                'top4_zodiacs': top4_zodiacs,
                'bet_numbers': bet_numbers,
                'bet_count': bet_count,
                'actual': actual,
                'is_hit': is_hit,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'multiplier': multiplier,
                'bet_amount': bet_amount,
                'profit': profit,
                'cumulative_profit': total_profit,
                'balance': current_balance,
                'skipped': False
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
        actual_bet_periods = test_periods - skipped_periods
        hit_rate = (hits / actual_bet_periods * 100) if actual_bet_periods > 0 else 0
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
        avg_profit_per_period = total_profit / test_periods
        
        # 打印汇总统计
        print(f"\n{'='*80}")
        print(f"📊 投注统计汇总")
        print(f"{'='*80}")
        print(f"测试期数: {test_periods}期")
        print(f"实际投注: {actual_bet_periods}期")
        print(f"跳过期数: {skipped_periods}期 ({skipped_periods/test_periods*100:.1f}%)")
        print(f"命中次数: {hits}次")
        print(f"命中率: {hit_rate:.2f}% (基于实际投注期数)")
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
            if not r.get('skipped', False):
                multiplier_distribution[r['multiplier']] += 1
        
        print(f"\n{'='*80}")
        print(f"📈 倍数分布统计")
        print(f"{'='*80}")
        for mult in sorted(multiplier_distribution.keys()):
            count = multiplier_distribution[mult]
            pct = count / actual_bet_periods * 100 if actual_bet_periods > 0 else 0
            print(f"{mult:.1f}倍: {count:3d}期 ({pct:5.2f}%)")
        
        # 保存详细记录到CSV
        df_results = pd.DataFrame(results)
        output_file = 'zodiac_top4_selective_betting_100periods.csv'
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细记录已保存至: {output_file}")
        
        return {
            'hit_rate': hit_rate,
            'total_cost': total_cost,
            'total_reward': total_reward,
            'total_profit': total_profit,
            'roi': roi,
            'max_consecutive_losses': max_consecutive_losses,
            'max_drawdown': max_drawdown,
            'skipped_periods': skipped_periods,
            'results': results
        }
    
    def compare_all_strategies(self, csv_file='data/lucky_numbers.csv'):
        """
        对比所有投注策略的效果
        
        对比：
        1. 固定1倍投注
        2. 动态投注（连败1期即加倍）
        3. 选择性动态投注（连败2期才加倍）★新增
        4. 马丁格尔倍投
        """
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        all_numbers = df['number'].values
        
        if len(all_numbers) < 130:
            print("数据不足，无法对比")
            return
        
        test_periods = 100
        start_idx = len(all_numbers) - test_periods
        
        # 策略1: 固定1倍
        stats_fixed = self._simulate_fixed_bet(all_numbers, start_idx)
        
        # 策略2: 动态投注（原版）
        stats_dynamic = self._simulate_dynamic_bet(all_numbers, start_idx)
        
        # 策略3: 选择性动态投注（新版）
        stats_selective = self._simulate_selective_bet(all_numbers, start_idx)
        
        # 策略4: 马丁格尔
        stats_martingale = self._simulate_martingale_bet(all_numbers, start_idx)
        
        # 打印对比
        print(f"\n{'='*80}")
        print(f"📊 四种投注策略对比分析 (最近100期)")
        print(f"{'='*80}\n")
        
        strategies = [
            ('固定1倍投注', stats_fixed),
            ('动态投注(连败1期加倍)', stats_dynamic),
            ('选择性动态(连败2期加倍)⭐新', stats_selective),
            ('马丁格尔倍投', stats_martingale)
        ]
        
        print(f"{'策略名称':<30} {'命中率':>10} {'总投注':>12} {'净收益':>12} {'ROI':>10} {'最大回撤':>12}")
        print(f"{'-'*90}")
        
        for name, stats in strategies:
            print(f"{name:<30} "
                  f"{stats['hit_rate']:>9.2f}% "
                  f"{stats['total_cost']:>11.2f}元 "
                  f"{stats['total_profit']:>+11.2f}元 "
                  f"{stats['roi']:>+9.2f}% "
                  f"{stats['max_drawdown']:>11.2f}元")
        
        # 推荐策略
        best_roi = max(strategies, key=lambda x: x[1]['roi'])
        best_profit = max(strategies, key=lambda x: x[1]['total_profit'])
        safest = min(strategies, key=lambda x: abs(x[1]['max_drawdown']))
        
        print(f"\n{'='*80}")
        print(f"🏆 策略推荐")
        print(f"{'='*80}")
        print(f"⭐ 最高ROI: {best_roi[0]} (ROI: {best_roi[1]['roi']:+.2f}%)")
        print(f"💰 最高收益: {best_profit[0]} (收益: {best_profit[1]['total_profit']:+.2f}元)")
        print(f"🛡️  最低风险: {safest[0]} (回撤: {safest[1]['max_drawdown']:.2f}元)")
        
        # 选择性策略的特殊说明
        if stats_selective.get('skipped_periods', 0) > 0:
            print(f"\n💡 选择性动态投注特点:")
            print(f"   - 跳过{stats_selective['skipped_periods']}期不投注")
            print(f"   - 更谨慎的倍投策略（连败2期才加倍）")
            print(f"   - 适合保守型投资者")
    
    def _simulate_fixed_bet(self, all_numbers, start_idx):
        """模拟固定1倍投注"""
        total_cost = 0
        total_profit = 0
        hits = 0
        max_drawdown = 0
        current_balance = 0
        
        test_periods = len(all_numbers) - start_idx
        
        for i in range(start_idx, len(all_numbers)):
            train_data = all_numbers[:i]
            actual = all_numbers[i]
            
            try:
                top5_zodiacs = self.predictor.predict_top5(train_data, recent_periods=100)
                top4_zodiacs = top5_zodiacs[:4]
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
    
    def _simulate_dynamic_bet(self, all_numbers, start_idx):
        """模拟动态投注（连败1期即加倍）"""
        total_cost = 0
        total_profit = 0
        hits = 0
        consecutive_wins = 0
        consecutive_losses = 0
        max_drawdown = 0
        current_balance = 0
        
        test_periods = len(all_numbers) - start_idx
        
        for i in range(start_idx, len(all_numbers)):
            train_data = all_numbers[:i]
            actual = all_numbers[i]
            
            try:
                top5_zodiacs = self.predictor.predict_top5(train_data, recent_periods=100)
                top4_zodiacs = top5_zodiacs[:4]
                bet_numbers = self.get_top4_numbers(top4_zodiacs)
                
                # 动态倍数计算（原版：连败1期即加倍）
                base_multiplier = 1.0
                if consecutive_wins > 0:
                    multiplier = max(0.5, base_multiplier - consecutive_wins * 0.5)
                elif consecutive_losses == 1:
                    multiplier = 2.0
                elif consecutive_losses == 2:
                    multiplier = 4.0
                else:
                    multiplier = 4.0 + (consecutive_losses - 2) * 2 if consecutive_losses > 2 else 1.0
                
                multiplier = min(multiplier, self.max_multiplier)
                
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
    
    def _simulate_selective_bet(self, all_numbers, start_idx):
        """模拟选择性动态投注（连败2期才加倍）"""
        total_cost = 0
        total_profit = 0
        hits = 0
        consecutive_wins = 0
        consecutive_losses = 0
        max_drawdown = 0
        current_balance = 0
        skipped_periods = 0
        
        test_periods = len(all_numbers) - start_idx
        
        for i in range(start_idx, len(all_numbers)):
            train_data = all_numbers[:i]
            actual = all_numbers[i]
            
            try:
                top5_zodiacs = self.predictor.predict_top5(train_data, recent_periods=100)
                top4_zodiacs = top5_zodiacs[:4]
                bet_numbers = self.get_top4_numbers(top4_zodiacs)
                
                # 选择性动态倍数计算（新版：连败2期才加倍）
                multiplier, should_bet = self.calculate_selective_multiplier(consecutive_wins, consecutive_losses)
                
                if not should_bet:
                    skipped_periods += 1
                    continue
                
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
            'hit_rate': hits / (test_periods - skipped_periods) * 100 if test_periods > skipped_periods else 0,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'roi': (total_profit / total_cost * 100) if total_cost > 0 else 0,
            'max_drawdown': max_drawdown,
            'skipped_periods': skipped_periods
        }
    
    def _simulate_martingale_bet(self, all_numbers, start_idx):
        """模拟马丁格尔倍投"""
        total_cost = 0
        total_profit = 0
        hits = 0
        consecutive_losses = 0
        max_drawdown = 0
        current_balance = 0
        
        test_periods = len(all_numbers) - start_idx
        
        for i in range(start_idx, len(all_numbers)):
            train_data = all_numbers[:i]
            actual = all_numbers[i]
            
            try:
                top5_zodiacs = self.predictor.predict_top5(train_data, recent_periods=100)
                top4_zodiacs = top5_zodiacs[:4]
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
    validator = ZodiacTop4SelectiveBetting()
    
    print("开始验证生肖TOP4选择性动态投注策略...\n")
    
    # 1. 验证最近100期
    result = validator.validate_100_periods()
    
    if result:
        # 2. 对比所有策略
        print(f"\n{'='*80}\n")
        validator.compare_all_strategies()
    
    print(f"\n{'='*80}")
    print("验证完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
