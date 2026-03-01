"""
智能投注策略模块
基于TOP15预测结果，实现渐进式投注系统以最大化收益
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


class BettingStrategy:
    """
    智能投注策略类
    
    规则:
    - 购买TOP15全部15个数字
    - 每个数字1元
    - 命中奖励: 45元
    - 未命中亏损: 15元/期
    - 使用渐进式投注系统（类似马丁格尔策略）
    """
    
    def __init__(self, base_bet=15, win_reward=47, loss_penalty=15):
        """
        初始化投注策略
        
        Args:
            base_bet: 基础投注金额（购买15个数字）
            win_reward: 命中奖励金额
            loss_penalty: 未命中惩罚金额
        """
        self.base_bet = base_bet  # 15元（15个数字各1元）
        self.win_reward = win_reward  # 47元
        self.loss_penalty = loss_penalty  # 15元
        
        # 策略参数
        self.max_multiplier = 10  # 最大投注倍数
        self.reset_after_win = True  # 命中后重置倍数
        
    def calculate_fixed_bet(self) -> Tuple[int, float]:
        """
        固定倍数策略（最保守）
        
        始终使用1倍投注，不进行倍投
        适合：低风险偏好，稳定收益
        
        Returns:
            (投注倍数, 投注金额)
        """
        return 1, self.base_bet
    
    def calculate_kelly_bet(self, hit_rate: float, consecutive_losses: int = 0) -> Tuple[int, float]:
        """
        凯利公式策略（动态优化）
        
        根据命中率和赔率计算最优投注比例
        公式: f* = (bp - q) / b
        其中: b=赔率, p=胜率, q=败率
        
        Args:
            hit_rate: 历史命中率
            consecutive_losses: 连续亏损次数（用于调整）
            
        Returns:
            (投注倍数, 投注金额)
        """
        # 计算赔率: 奖励/成本 - 1
        odds = (self.win_reward / self.base_bet) - 1  # (47/15) - 1 ≈ 2.13
        
        # 凯利公式
        p = hit_rate  # 胜率
        q = 1 - p     # 败率
        
        kelly_fraction = (odds * p - q) / odds
        
        # 使用半凯利（更保守）
        kelly_fraction = kelly_fraction * 0.5
        
        # 限制在合理范围
        kelly_fraction = max(0.1, min(kelly_fraction, 0.5))
        
        # 转换为倍数
        multiplier = max(1, int(kelly_fraction * 10))
        multiplier = min(multiplier, self.max_multiplier)
        
        bet_amount = multiplier * self.base_bet
        
        return multiplier, bet_amount
    
    def calculate_reverse_bet(self, consecutive_wins: int) -> Tuple[int, float]:
        """
        反向马丁格尔策略（激进）
        
        命中后增加倍数，未中后重置为1倍
        适合：趋势投注，连胜时扩大收益
        
        Args:
            consecutive_wins: 连续命中次数
            
        Returns:
            (投注倍数, 投注金额)
        """
        # 连胜时增加倍数
        multiplier = min(1 + consecutive_wins, self.max_multiplier)
        bet_amount = multiplier * self.base_bet
        
        return multiplier, bet_amount
    
    def calculate_aggressive_martingale_bet(self, consecutive_losses: int) -> Tuple[int, float]:
        """
        激进马丁格尔策略（高风险）
        
        每次亏损倍数翻倍
        适合：短期快速回本，但风险极高
        
        Args:
            consecutive_losses: 连续亏损次数
            
        Returns:
            (投注倍数, 投注金额)
        """
        if consecutive_losses == 0:
            return 1, self.base_bet
        
        # 倍数翻倍: 2^n
        multiplier = min(2 ** consecutive_losses, self.max_multiplier)
        bet_amount = multiplier * self.base_bet
        
        return multiplier, bet_amount
        
    def calculate_optimal_bet(self, consecutive_losses: int, total_loss: float) -> Tuple[int, float]:
        """
        计算最优投注金额
        
        使用修正的马丁格尔策略：
        - 连续亏损时，增加投注倍数
        - 确保下次命中能覆盖之前所有亏损并盈利
        
        Args:
            consecutive_losses: 连续亏损次数
            total_loss: 累计亏损金额
            
        Returns:
            (投注倍数, 投注金额)
        """
        if consecutive_losses == 0:
            return 1, self.base_bet
        
        # 计算需要的倍数来覆盖亏损
        # 公式: (总亏损 + 期望利润) / (奖励 - 单次投注成本)
        # 期望利润设为基础投注的盈利
        expected_profit = self.win_reward - self.base_bet - self.loss_penalty
        
        # 计算需要多少倍数才能覆盖亏损
        required_multiplier = np.ceil(
            (total_loss + expected_profit) / (self.win_reward - self.base_bet)
        )
        
        # 限制最大倍数
        multiplier = min(int(required_multiplier), self.max_multiplier)
        
        # 如果超过最大倍数，使用保守策略
        if required_multiplier > self.max_multiplier:
            # 渐进增加：每亏损一次增加固定倍数
            multiplier = min(1 + consecutive_losses, self.max_multiplier)
        
        bet_amount = multiplier * self.base_bet
        
        return multiplier, bet_amount
    
    def calculate_fibonacci_bet(self, consecutive_losses: int) -> Tuple[int, float]:
        """
        斐波那契投注策略（更保守）
        
        Args:
            consecutive_losses: 连续亏损次数
            
        Returns:
            (投注倍数, 投注金额)
        """
        # 斐波那契序列: 1, 1, 2, 3, 5, 8, 13, 21...
        fib = [1, 1]
        for i in range(2, consecutive_losses + 1):
            fib.append(fib[-1] + fib[-2])
            if fib[-1] > self.max_multiplier:
                break
        
        multiplier = min(fib[min(consecutive_losses, len(fib)-1)], self.max_multiplier)
        bet_amount = multiplier * self.base_bet
        
        return multiplier, bet_amount
    
    def calculate_dalembert_bet(self, consecutive_losses: int) -> Tuple[int, float]:
        """
        达朗贝尔投注策略（最保守）
        
        每次亏损只增加1个单位
        
        Args:
            consecutive_losses: 连续亏损次数
            
        Returns:
            (投注倍数, 投注金额)
        """
        multiplier = min(1 + consecutive_losses, self.max_multiplier)
        bet_amount = multiplier * self.base_bet
        
        return multiplier, bet_amount
    
    def calculate_dynamic_bet(self, consecutive_wins: int, consecutive_losses: int) -> Tuple[int, float]:
        """
        动态投注策略（智能平衡）
        
        核心逻辑：
        - 连续命中时：减少倍数（止盈思维）
        - 连续失败时：增加倍数（回本策略）
        - 连续2期失败：显著增加倍数
        
        规则：
        1. 初始倍数：1倍
        2. 每连胜1期：倍数减少0.5（最低0.5倍，即减半投注）
        3. 连续失败1期：倍数+1
        4. 连续失败2期：倍数+3（加速回本）
        5. 连续失败3期及以上：每期再+2
        
        Args:
            consecutive_wins: 连续命中次数
            consecutive_losses: 连续失败次数
            
        Returns:
            (投注倍数, 投注金额)
        """
        base_multiplier = 1.0
        
        # 连胜处理：每连胜1期，倍数减少0.5
        if consecutive_wins > 0:
            # 连胜越多，投注越保守
            reduction = consecutive_wins * 0.5
            multiplier = max(0.5, base_multiplier - reduction)  # 最低0.5倍
        
        # 连败处理：根据连败次数递增倍数
        elif consecutive_losses > 0:
            if consecutive_losses == 1:
                # 首次失败：倍数+1
                multiplier = base_multiplier + 1
            elif consecutive_losses == 2:
                # 连续2期失败：倍数+3（重点回本）
                multiplier = base_multiplier + 3
            else:
                # 连续3期及以上：+3 + (n-2)*2
                multiplier = base_multiplier + 3 + (consecutive_losses - 2) * 2
        
        else:
            # 初始状态
            multiplier = base_multiplier
        
        # 限制最大倍数
        multiplier = min(multiplier, self.max_multiplier)
        
        # 计算投注金额（支持小数倍数）
        bet_amount = multiplier * self.base_bet
        
        return multiplier, bet_amount
    
    def calculate_selective_dynamic_bet(self, consecutive_wins: int, consecutive_losses: int) -> Tuple[int, float, bool]:
        """
        选择性动态投注策略（更保守）
        
        核心逻辑：
        - 默认1倍投注
        - 连续失败2期才开始加倍（更谨慎）
        - 连续成功2期才减少投注
        - 某些情况可以选择不投注（skip）
        
        规则：
        1. 初始/连胜1期/连败1期：1倍投注
        2. 连续成功2期：下一期0.8倍（略微保守）
        3. 连续成功3期+：0.5倍（大幅保守）
        4. 连续失败2期：开始加倍到2倍
        5. 连续失败3期：4倍
        6. 连续失败4期+：每期再+2倍
        7. 跳过投注条件：连胜3期后的第一次失败可选择观望
        
        Args:
            consecutive_wins: 连续命中次数
            consecutive_losses: 连续失败次数
            
        Returns:
            (投注倍数, 投注金额, 是否投注)
        """
        base_multiplier = 1.0
        should_bet = True  # 默认投注
        
        # 连胜处理：需要连续2期成功才减少投注
        if consecutive_wins >= 3:
            # 连胜3期及以上：大幅保守
            multiplier = 0.5
        elif consecutive_wins == 2:
            # 连胜2期：略微保守
            multiplier = 0.8
        elif consecutive_wins == 1:
            # 连胜1期：保持基础倍数
            multiplier = base_multiplier
        
        # 连败处理：需要连续2期失败才开始加倍
        elif consecutive_losses >= 4:
            # 连败4期及以上：每期+2倍
            multiplier = 4.0 + (consecutive_losses - 3) * 2
        elif consecutive_losses == 3:
            # 连败3期：4倍加速
            multiplier = 4.0
        elif consecutive_losses == 2:
            # 连败2期：开始加倍到2倍
            multiplier = 2.0
        elif consecutive_losses == 1:
            # 首次失败：保持基础倍数，可选择观望
            multiplier = base_multiplier
            # 如果刚从连胜3期转为失败，可以选择跳过本期观望
            # 这里暂时不跳过，保持投注
            should_bet = True
        
        else:
            # 初始状态
            multiplier = base_multiplier
        
        # 限制最大倍数
        multiplier = min(multiplier, self.max_multiplier)
        
        # 计算投注金额
        bet_amount = multiplier * self.base_bet if should_bet else 0
        
        return multiplier, bet_amount, should_bet
    
    def calculate_stable_dynamic_bet(self, consecutive_wins: int, consecutive_losses: int) -> Tuple[int, float]:
        """
        稳健动态投注策略（保守优化版）
        
        核心逻辑：
        - 连胜时：始终保持1倍基础投注（不减少，享受趋势）
        - 连败时：根据连败次数递增倍数（快速回本）
        
        规则：
        1. 连胜任意期：保持1倍投注
        2. 初始状态：1倍投注
        3. 连续失败1期：2倍
        4. 连续失败2期：4倍
        5. 连续失败3期+：每期再+2倍
        
        优势：
        - 连胜期保持稳定收益，不会因保守错失盈利
        - 连败期快速追回，回本效率高
        - 心理压力小，容易坚持执行
        
        Args:
            consecutive_wins: 连续命中次数
            consecutive_losses: 连续失败次数
            
        Returns:
            (投注倍数, 投注金额)
        """
        base_multiplier = 1.0
        
        # 连胜处理：始终保持1倍基础投注
        if consecutive_wins > 0:
            multiplier = base_multiplier  # 不减少，保持基础倍数
        
        # 连败处理：根据连败次数递增
        elif consecutive_losses > 0:
            if consecutive_losses == 1:
                multiplier = 2.0  # 首次失败：2倍
            elif consecutive_losses == 2:
                multiplier = 4.0  # 连败2期：4倍
            else:
                # 连败3期及以上：4 + (n-2)*2
                multiplier = 4.0 + (consecutive_losses - 2) * 2
        
        else:
            # 初始状态
            multiplier = base_multiplier
        
        # 限制最大倍数
        multiplier = min(multiplier, self.max_multiplier)
        
        # 计算投注金额
        bet_amount = multiplier * self.base_bet
        
        return multiplier, bet_amount
    
    def simulate_strategy(self, 
                         predictions: List[List[int]], 
                         actuals: List[int],
                         strategy_type: str = 'martingale',
                         hit_rate: float = 0.5,
                         pause_config: Optional[Dict[str, int]] = None) -> Dict:
        """
        模拟投注策略效果
        
        Args:
            predictions: 每期TOP15预测列表
            actuals: 实际中奖号码列表
            strategy_type: 策略类型 ('fixed', 'kelly', 'reverse', 'aggressive', 
                                     'martingale', 'fibonacci', 'dalembert')
            hit_rate: 历史命中率（用于kelly策略）
            
        Returnif strategy_type == 'stable':
                multiplier, bet_amount = self.calculate_stable_dynamic_bet(consecutive_wins, consecutive_losses)
            els:
            包含详细统计信息的字典
        """
        if len(predictions) != len(actuals):
            raise ValueError("预测和实际结果数量不匹配")
        
        # 初始化统计
        total_profit = 0
        total_cost = 0
        total_reward = 0
        consecutive_losses = 0
        consecutive_wins = 0
        total_loss_accumulation = 0
        
        wins = 0
        losses = 0
        max_consecutive_losses = 0
        max_drawdown = 0
        current_balance = 0
        
        history = []
        use_pause_rule = pause_config is not None
        pause_trigger_hits = 0
        pause_length = 0
        pause_cooldown = 0
        pause_trigger_count = 0
        paused_hit_count = 0
        pause_periods = 0
        hits_since_pause_trigger = 0
        if use_pause_rule:
            pause_trigger_hits = max(1, pause_config.get('trigger_hits', 1))
            pause_length = max(1, pause_config.get('pause_length', 1))
        
        for i, (pred_top15, actual) in enumerate(zip(predictions, actuals)):
            if use_pause_rule and pause_cooldown > 0:
                pause_cooldown -= 1
                pause_periods += 1
                is_hit_when_paused = actual in pred_top15
                if is_hit_when_paused:
                    paused_hit_count += 1
                history.append({
                    'period': i + 1,
                    'prediction': pred_top15,
                    'actual': actual,
                    'is_hit': is_hit_when_paused,
                    'multiplier': 0,
                    'bet_amount': 0,
                    'consecutive_losses': consecutive_losses,
                    'consecutive_wins': consecutive_wins,
                    'skipped': True,
                    'pause_remaining': pause_cooldown,
                    'result': 'SKIP',
                    'profit': 0.0,
                    'total_profit': total_profit,
                    'current_balance': current_balance
                })
                continue
            # 选择策略计算投注金额
            if strategy_type == 'fixed':
                multiplier, bet_amount = self.calculate_fixed_bet()
            elif strategy_type == 'kelly':
                multiplier, bet_amount = self.calculate_kelly_bet(hit_rate, consecutive_losses)
            elif strategy_type == 'reverse':
                multiplier, bet_amount = self.calculate_reverse_bet(consecutive_wins)
            elif strategy_type == 'aggressive':
                multiplier, bet_amount = self.calculate_aggressive_martingale_bet(consecutive_losses)
            elif strategy_type == 'fibonacci':
                multiplier, bet_amount = self.calculate_fibonacci_bet(consecutive_losses)
            elif strategy_type == 'dalembert':
                multiplier, bet_amount = self.calculate_dalembert_bet(consecutive_losses)
            elif strategy_type == 'dynamic':
                multiplier, bet_amount = self.calculate_dynamic_bet(consecutive_wins, consecutive_losses)
            elif strategy_type == 'selective':
                multiplier, bet_amount, should_bet = self.calculate_selective_dynamic_bet(consecutive_wins, consecutive_losses)
                if not should_bet:
                    # 跳过本期投注
                    period_result = {
                        'period': i + 1,
                        'prediction': pred_top15,
                        'actual': actual,
                        'is_hit': False,
                        'multiplier': 0,
                        'bet_amount': 0,
                        'consecutive_losses': consecutive_losses,
                        'consecutive_wins': consecutive_wins,
                        'skipped': True
                    }
                    history.append(period_result)
                    continue
            else:  # martingale
                multiplier, bet_amount = self.calculate_optimal_bet(
                    consecutive_losses, total_loss_accumulation
                )
            
            total_cost += bet_amount
            
            # 检查是否命中
            is_hit = actual in pred_top15
            
            period_result = {
                'period': i + 1,
                'prediction': pred_top15,
                'actual': actual,
                'is_hit': is_hit,
                'multiplier': multiplier,
                'bet_amount': bet_amount,
                'consecutive_losses': consecutive_losses,
                'consecutive_wins': consecutive_wins
            }
            
            if is_hit:
                # 命中：获得奖励
                reward = multiplier * self.win_reward
                profit = reward - bet_amount
                
                total_reward += reward
                total_profit += profit
                current_balance += profit
                
                wins += 1
                consecutive_losses = 0
                consecutive_wins += 1
                total_loss_accumulation = 0
                
                period_result['reward'] = reward
                period_result['profit'] = profit
                period_result['result'] = 'WIN'
                if use_pause_rule:
                    hits_since_pause_trigger += 1
                    if pause_trigger_hits == 1 or hits_since_pause_trigger >= pause_trigger_hits:
                        pause_cooldown = pause_length
                        pause_trigger_count += 1
                        hits_since_pause_trigger = 0
                
            else:
                # 未命中：亏损
                loss = multiplier * self.loss_penalty
                total_profit -= loss
                current_balance -= loss
                
                losses += 1
                consecutive_losses += 1
                consecutive_wins = 0
                total_loss_accumulation += loss
                
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                max_drawdown = min(max_drawdown, current_balance)
                
                period_result['loss'] = loss
                period_result['profit'] = -loss
                period_result['result'] = 'LOSS'
                if use_pause_rule:
                    hits_since_pause_trigger = 0
            
            period_result['total_profit'] = total_profit
            period_result['current_balance'] = current_balance
            history.append(period_result)
        
        # 计算统计指标
        hit_rate = wins / len(predictions) if predictions else 0
        avg_profit_per_period = total_profit / len(predictions) if predictions else 0
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'strategy_type': strategy_type,
            'total_periods': len(predictions),
            'wins': wins,
            'losses': losses,
            'hit_rate': hit_rate,
            'total_cost': total_cost,
            'total_reward': total_reward,
            'total_profit': total_profit,
            'avg_profit_per_period': avg_profit_per_period,
            'max_consecutive_losses': max_consecutive_losses,
            'max_drawdown': max_drawdown,
            'final_balance': current_balance,
            'roi': roi,
            'history': history,
            'pause_stats': {
                'pause_trigger_count': pause_trigger_count,
                'pause_periods': pause_periods,
                'paused_hit_count': paused_hit_count
            } if use_pause_rule else None
        }
    
    def recommend_strategy(self, 
                          predictions: List[List[int]], 
                          actuals: List[int]) -> Dict:
        """
        对比所有策略，推荐最优策略
        
        Args:
            predictions: 每期TOP15预测列表
            actuals: 实际中奖号码列表
            
        Returns:
            包含所有策略对比和推荐的字典
        """
        strategies = ['martingale', 'fibonacci', 'dalembert']
        results = {}
        
        for strategy in strategies:
            results[strategy] = self.simulate_strategy(predictions, actuals, strategy)
        
        # 找出最优策略
        best_strategy = max(results.items(), key=lambda x: x[1]['total_profit'])
        
        return {
            'results': results,
            'recommended': best_strategy[0],
            'reason': f"最高总收益: {best_strategy[1]['total_profit']:.2f}元"
        }
    
    def print_strategy_report(self, result: Dict):
        """
        打印策略分析报告
        
        Args:
            result: simulate_strategy返回的结果
        """
        print("=" * 80)
        print(f"投注策略分析报告 - {result['strategy_type'].upper()}策略")
        print("=" * 80)
        print(f"\n【基础统计】")
        print(f"  总期数: {result['total_periods']}")
        print(f"  命中次数: {result['wins']}")
        print(f"  未命中次数: {result['losses']}")
        print(f"  命中率: {result['hit_rate']*100:.2f}%")
        
        print(f"\n【财务统计】")
        print(f"  总投注: {result['total_cost']:.2f}元")
        print(f"  总奖励: {result['total_reward']:.2f}元")
        print(f"  总收益: {result['total_profit']:.2f}元")
        print(f"  平均每期收益: {result['avg_profit_per_period']:.2f}元")
        print(f"  投资回报率(ROI): {result['roi']:.2f}%")
        
        print(f"\n【风险指标】")
        print(f"  最大连续亏损: {result['max_consecutive_losses']}期")
        print(f"  最大回撤: {result['max_drawdown']:.2f}元")
        print(f"  最终余额: {result['final_balance']:.2f}元")
        
        # 显示最近10期详情
        print(f"\n【最近10期详情】")
        print(f"{'期号':<6} {'倍数':<6} {'投注':<8} {'结果':<6} {'盈亏':<10} {'累计':<10}")
        print("-" * 60)
        
        for period in result['history'][-10:]:
            print(f"{period['period']:<6} "
                  f"{period['multiplier']:<6} "
                  f"{period['bet_amount']:<8.2f} "
                  f"{period['result']:<6} "
                  f"{period['profit']:>+10.2f} "
                  f"{period['total_profit']:>10.2f}")
        
        print("=" * 80)
    
    def generate_next_bet_recommendation(self, consecutive_losses: int, 
                                        total_loss: float,
                                        strategy_type: str = 'martingale',
                                        consecutive_wins: int = 0,
                                        hit_rate: float = 0.5) -> Dict:
        """
        为下一期生成投注建议
        
        Args:
            consecutive_losses: 当前连续亏损次数
            total_loss: 累计亏损金额
            strategy_type: 策略类型
            consecutive_wins: 连续命中次数（用于反向策略）
            hit_rate: 历史命中率（用于凯利策略）
            
        Returns:
            投注建议字典
        """
        if strategy_type == 'fixed':
            multiplier, bet_amount = self.calculate_fixed_bet()
        elif strategy_type == 'kelly':
            multiplier, bet_amount = self.calculate_kelly_bet(hit_rate, consecutive_losses)
        elif strategy_type == 'reverse':
            multiplier, bet_amount = self.calculate_reverse_bet(consecutive_wins)
        elif strategy_type == 'aggressive':
            multiplier, bet_amount = self.calculate_aggressive_martingale_bet(consecutive_losses)
        elif strategy_type == 'fibonacci':
            multiplier, bet_amount = self.calculate_fibonacci_bet(consecutive_losses)
        elif strategy_type == 'dalembert':
            multiplier, bet_amount = self.calculate_dalembert_bet(consecutive_losses)
        else:
            multiplier, bet_amount = self.calculate_optimal_bet(consecutive_losses, total_loss)
        
        # 计算潜在收益
        if_win_reward = multiplier * self.win_reward
        if_win_profit = if_win_reward - bet_amount
        if_loss_penalty = multiplier * self.loss_penalty
        
        return {
            'strategy': strategy_type,
            'consecutive_losses': consecutive_losses,
            'current_total_loss': total_loss,
            'recommended_multiplier': multiplier,
            'recommended_bet': bet_amount,
            'bet_per_number': bet_amount / 15,
            'potential_reward_if_win': if_win_reward,
            'potential_profit_if_win': if_win_profit,
            'potential_loss_if_miss': if_loss_penalty,
            'risk_reward_ratio': if_win_profit / if_loss_penalty if if_loss_penalty > 0 else 0
        }
    
    def print_next_bet_recommendation(self, recommendation: Dict):
        """打印下期投注建议"""
        print("\n" + "=" * 80)
        print("📊 下期投注建议")
        print("=" * 80)
        print(f"\n【当前状态】")
        print(f"  策略类型: {recommendation['strategy'].upper()}")
        print(f"  连续亏损: {recommendation['consecutive_losses']}期")
        print(f"  累计亏损: {recommendation['current_total_loss']:.2f}元")
        
        print(f"\n【投注建议】")
        print(f"  建议倍数: {recommendation['recommended_multiplier']}倍")
        print(f"  总投注额: {recommendation['recommended_bet']:.2f}元")
        print(f"  每个号码: {recommendation['bet_per_number']:.2f}元 × 15个号码")
        
        print(f"\n【收益预期】")
        print(f"  如果命中:")
        print(f"    - 获得奖励: {recommendation['potential_reward_if_win']:.2f}元")
        print(f"    - 净收益: +{recommendation['potential_profit_if_win']:.2f}元 ✓")
        print(f"  如果未中:")
        print(f"    - 额外亏损: -{recommendation['potential_loss_if_miss']:.2f}元")
        print(f"    - 累计亏损: {recommendation['current_total_loss'] + recommendation['potential_loss_if_miss']:.2f}元")
        
        print(f"\n【风险评估】")
        print(f"  盈亏比: {recommendation['risk_reward_ratio']:.2f}")
        if recommendation['risk_reward_ratio'] > 1.5:
            risk_level = "低风险 ✓"
        elif recommendation['risk_reward_ratio'] > 0.8:
            risk_level = "中等风险 ⚠"
        else:
            risk_level = "高风险 ⚠⚠"
        print(f"  风险等级: {risk_level}")
        
        print("=" * 80)


def demo_betting_strategy():
    """演示投注策略使用"""
    print("\n" + "="*80)
    print("智能投注策略演示")
    print("="*80)
    
    # 创建投注策略实例（使用15元基础投注）
    strategy = BettingStrategy(base_bet=15)
    
    # 模拟100期数据
    # 假设命中率为60%（TOP15的命中率）
    np.random.seed(42)
    n_periods = 100
    
    # 生成模拟预测和实际结果
    predictions = []
    actuals = []
    
    for i in range(n_periods):
        # 随机生成TOP15预测
        top15 = np.random.choice(range(1, 50), size=15, replace=False).tolist()
        predictions.append(top15)
        
        # 60%概率命中（实际号码在TOP15中）
        if np.random.random() < 0.6:
            actual = np.random.choice(top15)
        else:
            # 从TOP15之外随机选择
            others = [x for x in range(1, 50) if x not in top15]
            actual = np.random.choice(others)
        actuals.append(actual)
    
    # 对比所有策略
    comparison = strategy.recommend_strategy(predictions, actuals)
    
    # 打印各策略报告
    for strategy_name, result in comparison['results'].items():
        strategy.print_strategy_report(result)
        print("\n")
    
    # 打印推荐
    print("\n" + "="*80)
    print("【策略推荐】")
    print("="*80)
    print(f"推荐策略: {comparison['recommended'].upper()}")
    print(f"推荐理由: {comparison['reason']}")
    print("="*80)
    
    # 生成下期投注建议
    print("\n示例：下期投注建议")
    print("-"*80)
    
    # 假设当前连续亏损3期，累计亏损45元
    for strat_type in ['martingale', 'fibonacci', 'dalembert']:
        recommendation = strategy.generate_next_bet_recommendation(
            consecutive_losses=3,
            total_loss=45.0,
            strategy_type=strat_type
        )
        strategy.print_next_bet_recommendation(recommendation)


if __name__ == '__main__':
    demo_betting_strategy()
