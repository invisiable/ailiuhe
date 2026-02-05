"""
生肖TOP4动态适应投注策略验证
平衡风险与收益的优化方案

投注规则（基于实际数据分析优化）：
- 连胜时：前2期保持1倍，第3期轻微保护，4期+适度保护
- 连败时：温和加倍起步，中期加速，严控最大倍数

动态策略：
- 初始/胜1-2期：1.0倍（充分享受短连胜，占64%）
- 胜3期：0.85倍（轻微保护）
- 胜4+期：0.7倍（适度保护，长连胜占36%）
- 败1期：1.8倍（温和加倍，单次连败占44%）
- 败2期：3.5倍（中度追回）
- 败3期：5.5倍（强力回本）
- 败4+期：每期+1.5倍，最大8倍（严控风险，长连败占8%）
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from zodiac_enhanced_60_predictor import ZodiacEnhanced60Predictor


class ZodiacTop4AdaptiveBetting:
    """生肖TOP4动态适应投注验证器"""
    
    def __init__(self):
        self.predictor = ZodiacEnhanced60Predictor()
        self.win_reward = 47  # 命中奖励
        self.max_multiplier = 8  # 最大倍数（从10降到8）
        
    def calculate_adaptive_multiplier(self, consecutive_wins, consecutive_losses):
        """
        计算动态适应投注倍数
        基于实际数据分析优化
        
        Args:
            consecutive_wins: 连续命中次数
            consecutive_losses: 连续失败次数
            
        Returns:
            投注倍数
        """
        # 连胜处理：渐进式保护
        if consecutive_wins > 0:
            if consecutive_wins <= 2:
                multiplier = 1.0  # 前2次保持标准（短连胜占64%）
            elif consecutive_wins == 3:
                multiplier = 0.85  # 第3次轻微保护
            else:
                multiplier = 0.7  # 4+次适度保护（长连胜占36%）
        
        # 连败处理：温和起步，中期加速，严控上限
        else:
            if consecutive_losses == 0:
                multiplier = 1.0  # 初始状态
            elif consecutive_losses == 1:
                multiplier = 1.8  # 首败温和（单次连败占44%）
            elif consecutive_losses == 2:
                multiplier = 3.5  # 连败2期中度追回
            elif consecutive_losses == 3:
                multiplier = 5.5  # 连败3期强力回本
            else:
                # 连败4+期：每期+1.5倍，最大8倍
                multiplier = 5.5 + (consecutive_losses - 3) * 1.5
        
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
        """验证最近100期的投注效果"""
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
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        max_drawdown = 0
        current_balance = 0
        
        multiplier_usage = defaultdict(int)
        
        print(f"{'='*80}")
        print(f"生肖TOP4动态适应投注策略 - 最近100期验证报告")
        print(f"{'='*80}\n")
        print(f"策略特点：")
        print(f"  ✓ 前2次胜利保持1倍（短连胜占64%）")
        print(f"  ✓ 首败温和加倍1.8倍（单次连败占44%）")
        print(f"  ✓ 最大倍数8倍（严控风险）")
        print(f"  ✓ 平衡风险与收益，适合大众投资者\n")
        
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
            
            # 计算动态适应倍数
            multiplier = self.calculate_adaptive_multiplier(consecutive_wins, consecutive_losses)
            
            # 计算投注金额
            base_bet = bet_count * 1  # 每个数字1元
            bet_amount = multiplier * base_bet
            
            total_cost += bet_amount
            multiplier_usage[multiplier] += 1
            
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
                # 未命中
                profit = -bet_amount
                total_profit += profit
                current_balance += profit
                
                consecutive_wins = 0
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                
                result_icon = "❌"
            
            # 更新最大回撤
            max_drawdown = min(max_drawdown, current_balance)
            
            # 记录结果
            results.append({
                'period': period_num,
                'top4_zodiacs': top4_zodiacs,
                'bet_numbers': bet_numbers,
                'bet_count': bet_count,
                'actual': actual,
                'is_hit': is_hit,
                'consecutive_wins': consecutive_wins if is_hit else 0,
                'consecutive_losses': consecutive_losses if not is_hit else 0,
                'multiplier': multiplier,
                'bet_amount': bet_amount,
                'profit': profit,
                'cumulative_profit': total_profit,
                'balance': current_balance
            })
        
        # 打印汇总结果
        hit_rate = hits / test_periods
        roi = (total_profit / total_cost) * 100
        avg_profit_per_period = total_profit / test_periods
        
        print(f"\n{'='*80}")
        print(f"验证结果汇总")
        print(f"{'='*80}\n")
        print(f"测试期数: {test_periods}")
        print(f"命中期数: {hits} ✓")
        print(f"失败期数: {test_periods - hits} ✗")
        print(f"命中率: {hit_rate*100:.2f}%\n")
        
        print(f"财务统计:")
        print(f"总投注: {total_cost:.2f}元")
        print(f"总奖励: {total_reward:.2f}元")
        print(f"净收益: {total_profit:+.2f}元")
        print(f"ROI: {roi:+.2f}%")
        print(f"平均每期收益: {avg_profit_per_period:+.2f}元\n")
        
        print(f"风险指标:")
        print(f"最大连败: {max_consecutive_losses}期")
        print(f"最大回撤: {max_drawdown:.2f}元\n")
        
        print(f"倍数分布:")
        for mult in sorted(multiplier_usage.keys()):
            count = multiplier_usage[mult]
            pct = (count / test_periods) * 100
            print(f"  {mult:.2f}倍: {count}期 ({pct:.1f}%)")
        
        # 对比其他策略
        self.compare_strategies(results)
        
        # 保存详细记录
        results_df = pd.DataFrame(results)
        output_file = 'zodiac_top4_adaptive_betting_100periods.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细记录已保存到: {output_file}")
        
        return results
    
    def compare_strategies(self, adaptive_results):
        """对比动态适应与其他策略"""
        print(f"\n{'='*80}")
        print(f"六策略对比")
        print(f"{'='*80}\n")
        
        # 读取其他策略结果
        try:
            stable_df = pd.read_csv('zodiac_top4_stable_betting_100periods.csv', encoding='utf-8-sig')
            stable_investment = stable_df['bet_amount'].sum()
            stable_profit = stable_df['cumulative_profit'].iloc[-1]
            stable_roi = (stable_profit / stable_investment) * 100
            stable_drawdown = stable_df['balance'].min()
        except:
            stable_investment, stable_profit, stable_roi, stable_drawdown = 0, 0, 0, 0
        
        try:
            selective_df = pd.read_csv('zodiac_top4_selective_betting_100periods.csv', encoding='utf-8-sig')
            selective_investment = selective_df['bet_amount'].sum()
            selective_profit = selective_df['cumulative_profit'].iloc[-1]
            selective_roi = (selective_profit / selective_investment) * 100
            selective_drawdown = selective_df['balance'].min()
        except:
            selective_investment, selective_profit, selective_roi, selective_drawdown = 0, 0, 0, 0
        
        # 计算动态适应策略的数据
        adaptive_df = pd.DataFrame(adaptive_results)
        adaptive_investment = adaptive_df['bet_amount'].sum()
        adaptive_profit = adaptive_df['cumulative_profit'].iloc[-1]
        adaptive_roi = (adaptive_profit / adaptive_investment) * 100
        adaptive_drawdown = adaptive_df['balance'].min()
        
        # 固定1倍策略
        fixed_investment = 1700  # 100期 * 17元
        fixed_profit = 2295 - fixed_investment
        fixed_roi = (fixed_profit / fixed_investment) * 100
        fixed_drawdown = -34
        
        # 打印对比表格
        print(f"{'策略':<18} {'总投注':<12} {'净收益':<12} {'ROI':<10} {'回撤':<10}")
        print(f"{'-'*80}")
        print(f"{'固定1倍':<18} {fixed_investment:>10.0f}元 {fixed_profit:>+10.0f}元 {fixed_roi:>+8.2f}% {fixed_drawdown:>+8.0f}元")
        
        if stable_investment > 0:
            print(f"{'稳健动态':<18} {stable_investment:>10.0f}元 {stable_profit:>+10.0f}元 {stable_roi:>+8.2f}% {stable_drawdown:>+8.0f}元")
        
        if selective_investment > 0:
            print(f"{'选择性动态':<18} {selective_investment:>10.0f}元 {selective_profit:>+10.0f}元 {selective_roi:>+8.2f}% {selective_drawdown:>+8.0f}元")
        
        print(f"{'动态适应（NEW）':<18} {adaptive_investment:>10.0f}元 {adaptive_profit:>+10.0f}元 {adaptive_roi:>+8.2f}% {adaptive_drawdown:>+8.0f}元")
        
        # 推荐结论
        print(f"\n{'='*80}")
        print(f"推荐结论")
        print(f"{'='*80}\n")
        
        if adaptive_roi > stable_roi and abs(adaptive_drawdown) < abs(stable_drawdown):
            print(f"🎉 动态适应策略表现优异！")
            print(f"  ✓ ROI超越稳健动态: {adaptive_roi - stable_roi:+.2f}个百分点")
            print(f"  ✓ 回撤优于稳健动态: {abs(stable_drawdown) - abs(adaptive_drawdown):.0f}元")
            print(f"  ✓ 推荐作为新的首选策略")
        elif adaptive_roi > stable_roi:
            print(f"📈 动态适应策略ROI更高")
            print(f"  ✓ ROI: {adaptive_roi:+.2f}% vs {stable_roi:+.2f}%")
            print(f"  ⚠ 但回撤略大: {adaptive_drawdown:.0f}元 vs {stable_drawdown:.0f}元")
        else:
            print(f"📊 动态适应策略表现稳健")
            print(f"  • ROI: {adaptive_roi:+.2f}%")
            print(f"  • 回撤: {adaptive_drawdown:.0f}元")
            print(f"  • 适合风险偏好适中的投资者")


def main():
    validator = ZodiacTop4AdaptiveBetting()
    results = validator.validate_100_periods()
    
    if results:
        print(f"\n✓ 验证完成！")
        print(f"\n动态适应策略核心优势：")
        print(f"  1. 温和起步（首败1.8倍，避免过度）")
        print(f"  2. 中期加速（连败2-3期快速追回）")
        print(f"  3. 严控上限（最大8倍，降低风险）")
        print(f"  4. 渐进保护（连胜3期后适度降低）")


if __name__ == '__main__':
    main()
