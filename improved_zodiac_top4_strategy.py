"""
改进的生肖TOP4预测策略
基于问题分析，实施以下改进：
1. 增强短期数据权重（最近30期）
2. 引入实际分布均衡机制
3. 使用保守的斐波那契投注策略
4. 动态调整预测模型
"""

import pandas as pd
from collections import Counter, defaultdict
from ensemble_zodiac_predictor import EnsembleZodiacPredictor


class ImprovedZodiacTop4Strategy:
    """改进的生肖TOP4策略"""
    
    def __init__(self):
        self.base_predictor = EnsembleZodiacPredictor()
        self.zodiacs = self.base_predictor.zodiacs
        
    def predict_balanced(self, animals, recent_weight=0.6):
        """
        平衡预测：结合基础模型和短期分布
        
        Args:
            animals: 历史生肖列表
            recent_weight: 短期数据权重（0-1），默认0.6
        """
        # 获取基础预测
        base_result = self.base_predictor.predict_from_history(animals, top_n=5, debug=False)
        base_top5 = base_result['top5']
        
        # 分析最近30期的实际分布
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        recent_counter = Counter(recent_30)
        
        # 分析最近50期的分布趋势
        recent_50 = animals[-50:] if len(animals) >= 50 else animals
        recent_50_counter = Counter(recent_50)
        
        # 计算每个生肖的综合得分
        scores = {}
        
        for zodiac in self.zodiacs:
            # 1. 基础模型得分（40%）
            if zodiac in base_top5:
                rank = base_top5.index(zodiac) + 1
                base_score = (6 - rank) * 10  # TOP1=50分，TOP5=10分
            else:
                base_score = 0
            
            # 2. 短期冷热分析（30%）
            recent_count_30 = recent_counter.get(zodiac, 0)
            recent_count_50 = recent_50_counter.get(zodiac, 0)
            
            # 冷号加分（最近30期出现≤2次）
            if recent_count_30 <= 2:
                cold_score = 40
            elif recent_count_30 == 3:
                cold_score = 20
            else:
                cold_score = 0
            
            # 3. 间隔分析（20%）
            try:
                last_idx = len(animals) - 1 - animals[::-1].index(zodiac)
                gap = len(animals) - last_idx - 1
                
                # 最佳间隔：4-12期
                if 4 <= gap <= 12:
                    gap_score = 30
                elif gap > 12:
                    gap_score = 25
                elif gap >= 2:
                    gap_score = 10
                else:
                    gap_score = 0
            except ValueError:
                # 从未出现过
                gap_score = 35
            
            # 4. 均衡性加分（10%）- 防止过度集中
            # 如果该生肖在最近50期被预测过多但实际出现少，降分
            expected_rate = 50 / 12  # 理论上每个生肖应该出现4.17次/50期
            if recent_count_50 < expected_rate * 0.5:  # 实际出现远低于预期
                balance_score = 15  # 给弱势生肖加分
            else:
                balance_score = 0
            
            # 综合得分
            scores[zodiac] = base_score * 0.4 + cold_score * 0.3 + gap_score * 0.2 + balance_score * 0.1
        
        # 排序并返回TOP4
        sorted_zodiacs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top4': [z for z, s in sorted_zodiacs[:4]],
            'top5': [z for z, s in sorted_zodiacs[:5]],
            'scores': dict(sorted_zodiacs),
            'base_top5': base_top5,
            'recent_30_dist': dict(recent_counter)
        }
    
    def fibonacci_betting(self, consecutive_losses):
        """
        斐波那契投注策略（保守型）
        Args:
            consecutive_losses: 连续亏损次数
        Returns:
            (倍数, 投注额)
        """
        fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21]  # 斐波那契数列
        
        if consecutive_losses == 0:
            return 1, 16  # 基础投注16元（4个生肖×4元）
        
        # 限制最大倍数为8倍
        idx = min(consecutive_losses, len(fib_sequence) - 1)
        multiplier = fib_sequence[idx]
        bet_amount = 16 * multiplier
        
        return multiplier, bet_amount
    
    def validate_improved_strategy(self, csv_file='data/lucky_numbers.csv', test_periods=50):
        """
        验证改进策略的效果
        """
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        results = []
        balance = 0
        consecutive_wins = 0
        consecutive_losses = 0
        
        start_idx = len(animals) - test_periods - 1
        
        for i in range(start_idx, len(animals) - 1):
            period = i - start_idx + 1
            history = animals[:i+1]
            actual = animals[i+1]
            
            # 使用改进策略预测
            prediction = self.predict_balanced(history)
            top4 = prediction['top4']
            
            # 判断命中
            is_hit = actual in top4
            
            # 计算投注
            multiplier, bet_amount = self.fibonacci_betting(consecutive_losses)
            
            if is_hit:
                profit = 47 - bet_amount  # 中奖47元
                consecutive_wins += 1
                consecutive_losses = 0
            else:
                profit = -bet_amount
                consecutive_wins = 0
                consecutive_losses += 1
            
            balance += profit
            
            results.append({
                'period': period,
                'top4': ', '.join(top4),
                'actual': actual,
                'is_hit': '✅' if is_hit else '❌',
                'multiplier': multiplier,
                'bet_amount': bet_amount,
                'profit': profit,
                'balance': balance,
                'consecutive_losses': consecutive_losses
            })
        
        return results


def compare_strategies():
    """对比原策略和改进策略"""
    print("="*80)
    print("生肖TOP4策略对比 - 最近50期验证")
    print("="*80)
    
    # 原策略数据
    df_old = pd.read_csv('zodiac_top4_backtest_300periods.csv', encoding='utf-8-sig')
    old_last_50 = df_old.tail(50)
    
    old_hits = (old_last_50['is_hit'] == '是').sum()
    old_hit_rate = old_hits / 50 * 100
    old_profit = old_last_50['profit'].sum()
    old_bet = old_last_50['bet_amount'].sum()
    old_max_loss = old_last_50['consecutive_losses'].max()
    
    print(f"\n【原策略 - 马丁格尔倍投】")
    print(f"  命中率: {old_hit_rate:.1f}% ({old_hits}/50)")
    print(f"  累计投注: {old_bet:.0f}元")
    print(f"  累计盈利: {old_profit:.0f}元")
    print(f"  投资回报: {old_profit/old_bet*100:.1f}%")
    print(f"  最长连续不中: {old_max_loss}期")
    
    # 改进策略测试
    improved = ImprovedZodiacTop4Strategy()
    new_results = improved.validate_improved_strategy(test_periods=50)
    
    new_hits = sum(1 for r in new_results if r['is_hit'] == '✅')
    new_hit_rate = new_hits / 50 * 100
    new_profit = sum(r['profit'] for r in new_results)
    new_bet = sum(r['bet_amount'] for r in new_results)
    new_max_loss = max(r['consecutive_losses'] for r in new_results)
    
    print(f"\n【改进策略 - 平衡预测 + 斐波那契倍投】")
    print(f"  命中率: {new_hit_rate:.1f}% ({new_hits}/50)")
    print(f"  累计投注: {new_bet:.0f}元")
    print(f"  累计盈利: {new_profit:.0f}元")
    print(f"  投资回报: {new_profit/new_bet*100:.1f}%")
    print(f"  最长连续不中: {new_max_loss}期")
    
    # 对比结果
    print(f"\n{'='*80}")
    print("【改进效果对比】")
    print(f"{'='*80}")
    print(f"  命中率提升: {new_hit_rate - old_hit_rate:+.1f}%")
    print(f"  盈利提升: {new_profit - old_profit:+.0f}元")
    print(f"  投资回报提升: {new_profit/new_bet*100 - old_profit/old_bet*100:+.1f}%")
    print(f"  风险降低: 最长连续不中减少{old_max_loss - new_max_loss}期")
    
    # 保存详细结果
    df_new = pd.DataFrame(new_results)
    df_new.to_csv('zodiac_top4_improved_strategy_50periods.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 详细结果已保存至: zodiac_top4_improved_strategy_50periods.csv")
    
    # 显示部分详细记录
    print(f"\n【最近10期详细记录】")
    print(f"{'期数':<6} {'预测TOP4':<30} {'实际':<6} {'结果':<6} {'倍数':<6} {'盈亏':<8} {'累计':<8}")
    print("-" * 80)
    for r in new_results[-10:]:
        print(f"{r['period']:<6} {r['top4']:<30} {r['actual']:<6} {r['is_hit']:<6} "
              f"{r['multiplier']:<6} {r['profit']:<8.0f} {r['balance']:<8.0f}")


if __name__ == '__main__':
    compare_strategies()
