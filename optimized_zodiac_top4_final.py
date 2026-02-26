"""
生肖TOP4投注 - 实用改进方案
基于分析结果，提供可直接应用的改进代码
"""

import pandas as pd
from collections import Counter
from ensemble_zodiac_predictor import EnsembleZodiacPredictor


class OptimizedZodiacTop4Betting:
    """优化的生肖TOP4投注策略"""
    
    def __init__(self, strategy='balanced'):
        """
        Args:
            strategy: 'conservative'(保守), 'balanced'(平衡, 推荐), 'aggressive'(激进)
        """
        self.predictor = EnsembleZodiacPredictor()
        self.strategy = strategy
        
    def get_betting_multiplier(self, consecutive_losses):
        """
        根据策略返回投注倍数
        """
        if self.strategy == 'conservative':
            # 保守策略: 固定1倍
            return 1.0, 16
            
        elif self.strategy == 'balanced':
            # 平衡策略: 1-1-2-2-3-3
            if consecutive_losses == 0 or consecutive_losses == 1:
                return 1.0, 16
            elif consecutive_losses == 2 or consecutive_losses == 3:
                return 2.0, 32
            else:
                return 3.0, 48  # 最多3倍，比原来的10倍保守得多
                
        else:  # aggressive
            # 激进策略: 马丁格尔但限制在6倍
            multipliers = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
            multiplier = multipliers.get(consecutive_losses, 6)
            bet_amount = 16 * multiplier
            return multiplier, bet_amount
    
    def should_bet(self, prediction_result, history):
        """
        决策是否投注（选择性投注）
        """
        # 检查两个子模型的共识度
        consensus = prediction_result.get('consensus', [])
        
        # 高信心：共识生肖≥2个
        if len(consensus) >= 2:
            return True, "高信心"
        
        # 中信心：top4中有至少2个是top3候选
        top4 = prediction_result['top4']
        v10_top3 = set(prediction_result.get('v10_top3', []))
        opt_top3 = set(prediction_result.get('opt_top3', []))
        
        high_conf_count = sum(1 for z in top4 if z in v10_top3 or z in opt_top3)
        
        if high_conf_count >= 3:
            return True, "中信心"
        
        return False, "低信心-跳过"
    
    def apply_stop_loss(self, consecutive_losses, balance):
        """
        止损机制
        Returns:
            (是否继续投注, 原因)
        """
        # 规则1：连续4次不中，暂停1期
        if consecutive_losses >= 4:
            return False, "连续4次不中-暂停"
        
        # 规则2：余额亏损超过100元
        if balance < -100:
            return False, "亏损超过100元-暂停"
        
        return True, "正常"
    
    def validate_with_improvements(self, csv_file='data/lucky_numbers.csv', 
                                   test_periods=50, use_selective_betting=False):
        """
        验证改进后的策略
        Args:
            use_selective_betting: 是否使用选择性投注
        """
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        results = []
        balance = 0
        consecutive_losses = 0
        skipped_count = 0
        stop_loss_paused = 0
        
        start_idx = len(animals) - test_periods - 1
        
        for i in range(start_idx, len(animals) - 1):
            period = i - start_idx + 1
            history = animals[:i+1]
            actual = animals[i+1]
            
            # 获取预测
            prediction = self.predictor.predict_from_history(history, top_n=5, debug=False)
            top4 = prediction['top4']
            
            # 检查是否暂停中（止损后恢复）
            if stop_loss_paused > 0:
                # 检查本期是否会中
                would_hit = actual in top4
                
                if would_hit:
                    # 本期会中，恢复投注
                    stop_loss_paused = 0
                    consecutive_losses = 0  # 重置
                    results.append({
                        'period': period,
                        'top4': ', '.join(top4),
                        'actual': actual,
                        'is_hit': '⏸️会中',
                        'action': '暂停中-恢复',
                        'multiplier': 0,
                        'bet_amount': 0,
                        'profit': 0,
                        'balance': balance
                    })
                else:
                    # 继续暂停
                    stop_loss_paused -= 1
                    results.append({
                        'period': period,
                        'top4': ', '.join(top4),
                        'actual': actual,
                        'is_hit': '⏸️',
                        'action': f'暂停第{2-stop_loss_paused}期',
                        'multiplier': 0,
                        'bet_amount': 0,
                        'profit': 0,
                        'balance': balance
                    })
                continue
            
            # 选择性投注检查
            if use_selective_betting:
                should_bet, confidence = self.should_bet(prediction, history)
                if not should_bet:
                    skipped_count += 1
                    results.append({
                        'period': period,
                        'top4': ', '.join(top4),
                        'actual': actual,
                        'is_hit': '🚫',
                        'action': confidence,
                        'multiplier': 0,
                        'bet_amount': 0,
                        'profit': 0,
                        'balance': balance
                    })
                    continue
            
            # 止损检查
            can_bet, reason = self.apply_stop_loss(consecutive_losses, balance)
            if not can_bet:
                stop_loss_paused = 1  # 暂停1期
                results.append({
                    'period': period,
                    'top4': ', '.join(top4),
                    'actual': actual,
                    'is_hit': '🛑',
                    'action': reason,
                    'multiplier': 0,
                    'bet_amount': 0,
                    'profit': 0,
                    'balance': balance
                })
                continue
            
            # 正常投注
            multiplier, bet_amount = self.get_betting_multiplier(consecutive_losses)
            is_hit = actual in top4
            
            if is_hit:
                profit = 47 - bet_amount
                consecutive_losses = 0
                action = "✅命中"
            else:
                profit = -bet_amount
                consecutive_losses += 1
                action = "❌未中"
            
            balance += profit
            
            results.append({
                'period': period,
                'top4': ', '.join(top4),
                'actual': actual,
                'is_hit': '✅' if is_hit else '❌',
                'action': action,
                'multiplier': multiplier,
                'bet_amount': bet_amount,
                'profit': profit,
                'balance': balance
            })
        
        return results, skipped_count


def run_comprehensive_test():
    """运行全面测试"""
    print("="*100)
    print("生肖TOP4投注策略 - 全面改进测试")
    print("="*100)
    
    # 原策略数据
    df_old = pd.read_csv('zodiac_top4_backtest_300periods.csv', encoding='utf-8-sig')
    old_last_50 = df_old.tail(50)
    old_hits = (old_last_50['is_hit'] == '是').sum()
    old_hit_rate = old_hits / 50 * 100
    old_profit = old_last_50['profit'].sum()
    old_bet = old_last_50['bet_amount'].sum()
    
    print(f"\n【基准：原策略（马丁格尔10倍）】")
    print(f"  命中率: {old_hit_rate:.1f}%")
    print(f"  投注总额: {old_bet:.0f}元")
    print(f"  净盈利: {old_profit:.0f}元")
    print(f"  投资回报率: {old_profit/old_bet*100:.1f}%")
    
    # 测试各种策略
    strategies = [
        ('conservative', False, '保守型（固定1倍）'),
        ('balanced', False, '平衡型（1-1-2-2-3）'),
        ('balanced', True, '平衡型+选择性投注'),
        ('aggressive', False, '激进型（马丁格尔6倍）'),
    ]
    
    print(f"\n{'='*100}")
    print(f"{'策略':<25} {'命中率':<12} {'投注期数':<10} {'投注总额':<12} {'净盈利':<12} {'投资回报率':<12} {'风险'}")
    print("-"*100)
    
    best_strategy = None
    best_roi = old_profit/old_bet*100
    
    for strategy, selective, name in strategies:
        optimizer = OptimizedZodiacTop4Betting(strategy=strategy)
        results, skipped = optimizer.validate_with_improvements(
            test_periods=50, 
            use_selective_betting=selective
        )
        
        bet_periods = sum(1 for r in results if r['bet_amount'] > 0)
        hits = sum(1 for r in results if r['is_hit'] == '✅')
        total_bet = sum(r['bet_amount'] for r in results)
        total_profit = sum(r['profit'] for r in results)
        
        if bet_periods > 0:
            hit_rate = hits / bet_periods * 100
            roi = total_profit / total_bet * 100 if total_bet > 0 else 0
        else:
            hit_rate = 0
            roi = 0
        
        risk_level = '🟢低' if strategy == 'conservative' else '🟡中' if strategy == 'balanced' else '🔴高'
        
        print(f"{name:<25} {hit_rate:>5.1f}% ({hits}/{bet_periods}) "
              f"{bet_periods:>6}期 {total_bet:>9.0f}元 {total_profit:>9.0f}元 "
              f"{roi:>10.1f}% {risk_level}")
        
        if roi > best_roi:
            best_roi = roi
            best_strategy = name
        
        # 保存结果
        filename = f"zodiac_top4_optimized_{strategy}{'_selective' if selective else ''}_50periods.csv"
        pd.DataFrame(results).to_csv(filename, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*100)
    print("【推荐方案】")
    print("="*100)
    
    if best_strategy:
        print(f"✅ 最佳策略: {best_strategy}")
        print(f"   投资回报率: {best_roi:.1f}%")
    else:
        print(f"✅ 保持原策略")
        print(f"   在当前数据下，原策略表现最优")
    
    print(f"\n💡 实施建议:")
    print(f"   1. 短期（本周）：采用【平衡型策略】，降低风险")
    print(f"   2. 中期（本月）：观察实际效果，记录每期数据")
    print(f"   3. 长期（季度）：根据实际命中率动态调整")
    
    print(f"\n📁 详细数据已保存至:")
    for strategy, selective, name in strategies:
        filename = f"zodiac_top4_optimized_{strategy}{'_selective' if selective else ''}_50periods.csv"
        print(f"   • {filename}")


if __name__ == '__main__':
    run_comprehensive_test()
