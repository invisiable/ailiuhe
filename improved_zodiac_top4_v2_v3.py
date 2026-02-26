"""
改进生肖TOP4策略 V2
采用更保守的改进方案：
1. 保留基础模型70%权重
2. 增加最近10-20期的短期适应性
3. 使用固定1.5倍投注策略替代马丁格尔
"""

import pandas as pd
from collections import Counter
from ensemble_zodiac_predictor import EnsembleZodiacPredictor


class ImprovedZodiacTop4V2:
    """改进生肖TOP4策略 V2 - 保守版"""
    
    def __init__(self):
        self.base_predictor = EnsembleZodiacPredictor()
        self.zodiacs = self.base_predictor.zodiacs
        
    def predict_enhanced(self, animals):
        """
        增强预测：基础模型 + 短期微调
        """
        # 1. 获取基础预测（70%权重）
        base_result = self.base_predictor.predict_from_history(animals, top_n=6, debug=False)
        base_scores = {}
        for i, zodiac in enumerate(base_result['top5'][:6], 1):
            base_scores[zodiac] = (7 - i) * 10  # 60, 50, 40, 30, 20, 10
        
        # 2. 短期调整（30%权重）
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        recent_counter = Counter(recent_20)
        
        adjustment_scores = {}
        for zodiac in self.zodiacs:
            recent_count = recent_counter.get(zodiac, 0)
            
            # 间隔分析
            try:
                last_idx = len(animals) - 1 - animals[::-1].index(zodiac)
                gap = len(animals) - last_idx - 1
            except ValueError:
                gap = 100  # 从未出现
            
            # 合理的调整分数
            if recent_count == 0 and gap >= 5:  # 冷号但不要过分偏好
                adjustment_scores[zodiac] = 20
            elif recent_count >= 3:  # 过热号降权
                adjustment_scores[zodiac] = -10
            elif gap >= 15:  # 长期未出现
                adjustment_scores[zodiac] = 15
            elif 5 <= gap <= 12:  # 理想间隔
                adjustment_scores[zodiac] = 10
            else:
                adjustment_scores[zodiac] = 0
        
        # 3. 综合得分
        final_scores = {}
        for zodiac in self.zodiacs:
            base = base_scores.get(zodiac, 0) * 0.7
            adj = adjustment_scores.get(zodiac, 0) * 0.3
            final_scores[zodiac] = base + adj
        
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top4': [z for z, s in sorted_zodiacs[:4]],
            'top5': [z for z, s in sorted_zodiacs[:5]],
            'scores': dict(sorted_zodiacs)
        }
    
    def conservative_betting(self, consecutive_losses):
        """
        保守倍投策略
        - 0-1次不中：1倍
        - 2-3次不中：1.5倍
        - 4次以上：2倍
        """
        if consecutive_losses == 0:
            return 1.0, 16
        elif consecutive_losses <= 1:
            return 1.0, 16
        elif consecutive_losses <= 3:
            return 1.5, 24
        else:
            return 2.0, 32
    
    def validate(self, csv_file='data/lucky_numbers.csv', test_periods=50):
        """验证策略"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        results = []
        balance = 0
        consecutive_losses = 0
        
        start_idx = len(animals) - test_periods - 1
        
        for i in range(start_idx, len(animals) - 1):
            period = i - start_idx + 1
            history = animals[:i+1]
            actual = animals[i+1]
            
            # 预测
            prediction = self.predict_enhanced(history)
            top4 = prediction['top4']
            
            # 判断
            is_hit = actual in top4
            
            # 投注
            multiplier, bet_amount = self.conservative_betting(consecutive_losses)
            
            if is_hit:
                profit = 47 - bet_amount
                consecutive_losses = 0
            else:
                profit = -bet_amount
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


class ImprovedZodiacTop4V3:
    """改进生肖TOP4策略 V3 - 固定投注版"""
    
    def __init__(self):
        self.v2_strategy = ImprovedZodiacTop4V2()
    
    def fixed_betting(self, consecutive_losses):
        """固定投注策略 - 不倍投"""
        return 1.0, 16
    
    def validate(self, csv_file='data/lucky_numbers.csv', test_periods=50):
        """验证策略"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        results = []
        balance = 0
        consecutive_losses = 0
        
        start_idx = len(animals) - test_periods - 1
        
        for i in range(start_idx, len(animals) - 1):
            period = i - start_idx + 1
            history = animals[:i+1]
            actual = animals[i+1]
            
            # 使用V2的预测逻辑
            prediction = self.v2_strategy.predict_enhanced(history)
            top4 = prediction['top4']
            
            # 判断
            is_hit = actual in top4
            
            # 固定投注
            multiplier, bet_amount = self.fixed_betting(consecutive_losses)
            
            if is_hit:
                profit = 47 - bet_amount
                consecutive_losses = 0
            else:
                profit = -bet_amount
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


def comprehensive_comparison():
    """全面对比所有策略"""
    print("="*90)
    print("生肖TOP4策略全面对比 - 最近50期验证")
    print("="*90)
    
    # 1. 原策略
    df_old = pd.read_csv('zodiac_top4_backtest_300periods.csv', encoding='utf-8-sig')
    old_last_50 = df_old.tail(50)
    old_hits = (old_last_50['is_hit'] == '是').sum()
    old_hit_rate = old_hits / 50 * 100
    old_profit = old_last_50['profit'].sum()
    old_bet = old_last_50['bet_amount'].sum()
    old_max_loss = old_last_50['consecutive_losses'].max()
    
    # 2. 改进V2策略
    v2 = ImprovedZodiacTop4V2()
    v2_results = v2.validate(test_periods=50)
    v2_hits = sum(1 for r in v2_results if r['is_hit'] == '✅')
    v2_hit_rate = v2_hits / 50 * 100
    v2_profit = sum(r['profit'] for r in v2_results)
    v2_bet = sum(r['bet_amount'] for r in v2_results)
    v2_max_loss = max(r['consecutive_losses'] for r in v2_results)
    
    # 3. 改进V3策略（固定投注）
    v3 = ImprovedZodiacTop4V3()
    v3_results = v3.validate(test_periods=50)
    v3_hits = sum(1 for r in v3_results if r['is_hit'] == '✅')
    v3_hit_rate = v3_hits / 50 * 100
    v3_profit = sum(r['profit'] for r in v3_results)
    v3_bet = sum(r['bet_amount'] for r in v3_results)
    v3_max_loss = max(r['consecutive_losses'] for r in v3_results)
    
    # 打印对比表格
    print(f"\n{'策略':<25} {'命中率':<12} {'累计投注':<12} {'累计盈利':<12} {'投资回报':<12} {'最长连续不中'}")
    print("-" * 90)
    
    print(f"{'原策略(马丁格尔)':<25} {old_hit_rate:>5.1f}% ({old_hits}/50) "
          f"{old_bet:>8.0f}元 {old_profit:>9.0f}元 {old_profit/old_bet*100:>8.1f}% {old_max_loss:>8}期")
    
    print(f"{'V2(保守倍投)':<25} {v2_hit_rate:>5.1f}% ({v2_hits}/50) "
          f"{v2_bet:>8.0f}元 {v2_profit:>9.0f}元 {v2_profit/v2_bet*100:>8.1f}% {v2_max_loss:>8}期")
    
    print(f"{'V3(固定投注)':<25} {v3_hit_rate:>5.1f}% ({v3_hits}/50) "
          f"{v3_bet:>8.0f}元 {v3_profit:>9.0f}元 {v3_profit/v3_bet*100:>8.1f}% {v3_max_loss:>8}期")
    
    # 推荐结论
    print(f"\n{'='*90}")
    print("【策略推荐】")
    print(f"{'='*90}")
    
    best_roi = max(old_profit/old_bet*100, v2_profit/v2_bet*100, v3_profit/v3_bet*100)
    
    if best_roi == old_profit/old_bet*100:
        print("✅ 推荐：原策略（马丁格尔倍投）")
        print(f"   虽然风险较高，但在当前数据下投资回报最好")
    elif best_roi == v2_profit/v2_bet*100:
        print("✅ 推荐：V2策略（保守倍投）")
        print(f"   平衡风险与收益")
    else:
        print("✅ 推荐：V3策略（固定投注）")
        print(f"   风险最低，适合长期稳定投注")
    
    # 保存结果
    pd.DataFrame(v2_results).to_csv('zodiac_top4_v2_strategy_50periods.csv', 
                                     index=False, encoding='utf-8-sig')
    pd.DataFrame(v3_results).to_csv('zodiac_top4_v3_strategy_50periods.csv', 
                                     index=False, encoding='utf-8-sig')
    
    print(f"\n📁 详细结果已保存:")
    print(f"   • zodiac_top4_v2_strategy_50periods.csv")
    print(f"   • zodiac_top4_v3_strategy_50periods.csv")


if __name__ == '__main__':
    comprehensive_comparison()
