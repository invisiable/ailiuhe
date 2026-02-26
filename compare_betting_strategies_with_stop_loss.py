"""
倍投策略对比测试 - 基于3期止损策略
对比5种倍投策略在相同止损条件下的收益表现
"""

import pandas as pd
from ensemble_zodiac_predictor import EnsembleZodiacPredictor


class BettingStrategyComparer:
    """倍投策略对比器"""
    
    def __init__(self, base_bet=16, win_amount=45, stop_loss_threshold=3, max_paused_streak=8):
        self.base_bet = base_bet
        self.win_amount = win_amount
        self.stop_loss_threshold = stop_loss_threshold
        self.max_paused_streak = max_paused_streak
        
    def martingale_multiplier(self, consecutive_losses):
        """马丁格尔倍投：2^n（激进）"""
        if consecutive_losses == 0:
            return 1
        # 限制最大倍数避免爆仓
        return min(2 ** consecutive_losses, 89)
    
    def fibonacci_multiplier(self, consecutive_losses):
        """斐波那契倍投：1,1,2,3,5,8,13,21,34,55,89（平衡）"""
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        return fib[min(consecutive_losses, len(fib) - 1)]
    
    def dalembert_multiplier(self, consecutive_losses):
        """达朗贝尔倍投：1+n（保守）"""
        return 1 + consecutive_losses
    
    def conservative_multiplier(self, consecutive_losses):
        """保守倍投：更缓慢的增长"""
        # 使用平方根增长，比线性慢但比固定快
        if consecutive_losses == 0:
            return 1
        elif consecutive_losses == 1:
            return 1
        elif consecutive_losses == 2:
            return 2
        else:
            return 2 + (consecutive_losses - 2)
    
    def fixed_multiplier(self, consecutive_losses):
        """固定投注：始终1倍（最保守）"""
        return 1
    
    def simulate_strategy(self, hit_records, multiplier_func, strategy_name):
        """
        模拟单个倍投策略
        
        Args:
            hit_records: 命中记录列表 (True/False)
            multiplier_func: 倍数计算函数
            strategy_name: 策略名称
        
        Returns:
            策略结果字典
        """
        total_profit = 0
        total_investment = 0
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        max_consecutive_wins = 0
        max_bet = self.base_bet
        balance_history = [0]
        max_drawdown = 0
        peak_balance = 0
        
        # 止损相关变量
        is_betting = True
        paused_periods = 0
        paused_count = 0
        actual_betting_periods = 0
        hits = 0
        
        for hit in hit_records:
            if is_betting:
                # 当前在投注状态
                paused_count = 0
                
                # 计算当前倍数
                multiplier = multiplier_func(consecutive_losses)
                current_bet = self.base_bet * multiplier
                total_investment += current_bet
                actual_betting_periods += 1
                
                if hit:
                    # 命中：获得奖励
                    profit = self.win_amount * multiplier - current_bet
                    total_profit += profit
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    hits += 1
                else:
                    # 未中：亏损
                    total_profit -= current_bet
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    
                    # 检查是否触发止损
                    if consecutive_losses >= self.stop_loss_threshold:
                        is_betting = False
                        paused_count = 0
                
                # 更新最大单期投入
                max_bet = max(max_bet, current_bet)
            else:
                # 暂停投注状态
                paused_periods += 1
                paused_count += 1
                
                # 恢复投注的两个条件
                if hit:
                    # 条件1：预测命中，恢复投注
                    is_betting = True
                    consecutive_losses = 0
                    consecutive_wins = 0
                    paused_count = 0
                elif paused_count >= self.max_paused_streak:
                    # 条件2：连续暂停8期后自动恢复
                    is_betting = True
                    consecutive_losses = 0
                    consecutive_wins = 0
                    paused_count = 0
            
            # 记录余额历史
            balance_history.append(total_profit)
            
            # 计算最大回撤
            peak_balance = max(peak_balance, total_profit)
            drawdown = peak_balance - total_profit
            max_drawdown = max(max_drawdown, drawdown)
        
        roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
        hit_rate = (hits / actual_betting_periods * 100) if actual_betting_periods > 0 else 0
        
        return {
            'strategy_name': strategy_name,
            'total_profit': total_profit,
            'total_investment': total_investment,
            'roi': roi,
            'hit_rate': hit_rate,
            'hits': hits,
            'actual_betting_periods': actual_betting_periods,
            'paused_periods': paused_periods,
            'max_consecutive_losses': max_consecutive_losses,
            'max_consecutive_wins': max_consecutive_wins,
            'max_bet': max_bet,
            'max_drawdown': max_drawdown,
            'balance_history': balance_history
        }
    
    def compare_all_strategies(self, periods=200):
        """对比所有倍投策略"""
        
        # 加载数据
        df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
        df = df.tail(periods).reset_index(drop=True)
        
        predictor = EnsembleZodiacPredictor()
        
        # 生成命中记录
        hit_records = []
        for i in range(len(df)):
            if i >= 10:
                history = df.iloc[:i]['animal'].tolist()
                prediction = predictor.predict_from_history(history, top_n=4)
                top4_animals = prediction['top4']
                actual_animal = df.iloc[i]['animal']
                hit = actual_animal in top4_animals
                hit_records.append(hit)
        
        # 定义所有策略
        strategies = [
            ('固定投注', self.fixed_multiplier),
            ('保守倍投', self.conservative_multiplier),
            ('达朗贝尔倍投', self.dalembert_multiplier),
            ('斐波那契倍投', self.fibonacci_multiplier),
            ('马丁格尔倍投', self.martingale_multiplier),
        ]
        
        results = []
        
        print(f"{'='*100}")
        print(f" 倍投策略对比测试 - 基于3期止损策略")
        print(f"{'='*100}")
        print(f"回测期数: {len(hit_records)} 期")
        print(f"基础投注: {self.base_bet}元/期")
        print(f"命中奖励: {self.win_amount}元")
        print(f"止损规则: 连续失败 {self.stop_loss_threshold} 期暂停")
        print(f"恢复规则: 预测命中 OR 连续暂停 {self.max_paused_streak} 期后自动恢复")
        print(f"{'='*100}\n")
        
        # 测试每个策略
        for strategy_name, multiplier_func in strategies:
            print(f"正在测试: {strategy_name}...")
            result = self.simulate_strategy(hit_records, multiplier_func, strategy_name)
            results.append(result)
        
        # 打印对比结果
        print(f"\n{'='*100}")
        print(f" 策略对比结果汇总")
        print(f"{'='*100}\n")
        
        # 创建对比表格
        comparison_df = pd.DataFrame([{
            '策略名称': r['strategy_name'],
            '投注期数': r['actual_betting_periods'],
            '暂停期数': r['paused_periods'],
            '命中次数': r['hits'],
            '命中率': f"{r['hit_rate']:.2f}%",
            '总投入': f"{r['total_investment']:.0f}元",
            '总盈利': f"{r['total_profit']:+.0f}元",
            'ROI': f"{r['roi']:+.2f}%",
            '最大连败': f"{r['max_consecutive_losses']}期",
            '最大连胜': f"{r['max_consecutive_wins']}期",
            '最大单注': f"{r['max_bet']:.0f}元",
            '最大回撤': f"{r['max_drawdown']:.0f}元"
        } for r in results])
        
        print(comparison_df.to_string(index=False))
        
        # 详细分析
        print(f"\n{'='*100}")
        print(f" 详细分析")
        print(f"{'='*100}\n")
        
        for result in results:
            print(f"【{result['strategy_name']}】")
            print(f"  投资效率: {result['actual_betting_periods']} 期投注 / {periods} 期总期数 = {result['actual_betting_periods']/periods*100:.1f}%")
            print(f"  盈利能力: {result['total_profit']:.0f}元 / {result['actual_betting_periods']} 期 = {result['total_profit']/result['actual_betting_periods']:.2f}元/期")
            print(f"  风险指标: 最大回撤 {result['max_drawdown']:.0f}元, 最大单注 {result['max_bet']:.0f}元")
            
            # 风险收益比
            risk_reward_ratio = result['total_profit'] / result['max_drawdown'] if result['max_drawdown'] > 0 else float('inf')
            print(f"  风险收益比: {risk_reward_ratio:.2f} (收益/最大回撤)")
            print()
        
        # 排名
        print(f"{'='*100}")
        print(f" 策略排名")
        print(f"{'='*100}\n")
        
        # 按ROI排名
        roi_ranking = sorted(results, key=lambda x: x['roi'], reverse=True)
        print("【ROI排名】")
        for i, r in enumerate(roi_ranking, 1):
            print(f"  {i}. {r['strategy_name']:12s} - ROI: {r['roi']:+7.2f}%")
        
        # 按总盈利排名
        profit_ranking = sorted(results, key=lambda x: x['total_profit'], reverse=True)
        print("\n【总盈利排名】")
        for i, r in enumerate(profit_ranking, 1):
            print(f"  {i}. {r['strategy_name']:12s} - 盈利: {r['total_profit']:+7.0f}元")
        
        # 按风险排名（最大回撤越小越好）
        risk_ranking = sorted(results, key=lambda x: x['max_drawdown'])
        print("\n【风险控制排名】（最大回撤越小越好）")
        for i, r in enumerate(risk_ranking, 1):
            print(f"  {i}. {r['strategy_name']:12s} - 最大回撤: {r['max_drawdown']:.0f}元")
        
        # 综合评分
        print(f"\n{'='*100}")
        print(f" 综合评分 (ROI×0.4 + 盈利×0.3 + 风险控制×0.3)")
        print(f"{'='*100}\n")
        
        max_profit = max([r['total_profit'] for r in results])
        min_drawdown = min([r['max_drawdown'] for r in results]) if min([r['max_drawdown'] for r in results]) > 0 else 1
        max_roi = max([r['roi'] for r in results])
        
        scored_results = []
        for r in results:
            # 归一化分数
            roi_score = (r['roi'] / max_roi) * 100 if max_roi > 0 else 0
            profit_score = (r['total_profit'] / max_profit) * 100 if max_profit > 0 else 0
            risk_score = (min_drawdown / r['max_drawdown']) * 100 if r['max_drawdown'] > 0 else 100
            
            # 综合评分
            total_score = roi_score * 0.4 + profit_score * 0.3 + risk_score * 0.3
            
            scored_results.append({
                'strategy': r['strategy_name'],
                'total_score': total_score,
                'roi_score': roi_score,
                'profit_score': profit_score,
                'risk_score': risk_score
            })
        
        scored_results.sort(key=lambda x: x['total_score'], reverse=True)
        
        for i, s in enumerate(scored_results, 1):
            print(f"  {i}. {s['strategy']:12s} - 综合分: {s['total_score']:.1f} "
                  f"(ROI: {s['roi_score']:.1f}, 盈利: {s['profit_score']:.1f}, 风控: {s['risk_score']:.1f})")
        
        # 推荐策略
        print(f"\n{'='*100}")
        print(f" 策略推荐")
        print(f"{'='*100}\n")
        
        best_roi = roi_ranking[0]
        best_profit = profit_ranking[0]
        best_risk = risk_ranking[0]
        best_overall = scored_results[0]
        
        print(f"✅ 【最佳ROI策略】: {best_roi['strategy_name']} - {best_roi['roi']:+.2f}%")
        print(f"   适合: 追求资金使用效率最大化\n")
        
        print(f"✅ 【最高盈利策略】: {best_profit['strategy_name']} - {best_profit['total_profit']:+.0f}元")
        print(f"   适合: 追求绝对收益最大化\n")
        
        print(f"✅ 【最低风险策略】: {best_risk['strategy_name']} - 最大回撤 {best_risk['max_drawdown']:.0f}元")
        print(f"   适合: 风险厌恶型投资者\n")
        
        print(f"⭐ 【综合最优策略】: {best_overall['strategy']} - 综合分 {best_overall['total_score']:.1f}")
        print(f"   适合: 平衡收益与风险的稳健型投资者")
        
        print(f"\n{'='*100}")
        
        # 保存结果
        comparison_df.to_csv('betting_strategies_comparison_stop_loss.csv', index=False, encoding='utf-8-sig')
        print(f"\n详细对比数据已保存到: betting_strategies_comparison_stop_loss.csv")
        
        return results


if __name__ == '__main__':
    comparer = BettingStrategyComparer(
        base_bet=16,
        win_amount=45,
        stop_loss_threshold=3,  # 3期止损
        max_paused_streak=8
    )
    
    comparer.compare_all_strategies(periods=200)
