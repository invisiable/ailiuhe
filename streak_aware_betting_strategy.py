"""
连败感知智能倍投策略
基于历史数据分析发现的规律：
1. 连续失败5次后，命中率从39.8%提升到63.6%（+59.7%）
2. 连续失败3次后，命中率从39.8%提升到45.5%（+14.1%）
3. 结合时间周期性（周三、周六命中率更高）
"""

import pandas as pd
from collections import deque
from datetime import datetime

class StreakAwareBettingStrategy:
    """连败感知倍投策略"""
    
    def __init__(self, base_bet=20, base_reward=47, max_multiplier=10):
        """
        初始化策略
        
        参数:
            base_bet: 基础投注额（生肖TOP5默认20元）
            base_reward: 命中奖励（生肖TOP5默认47元）
            max_multiplier: 最大倍数限制
        """
        self.base_bet = base_bet
        self.base_reward = base_reward
        self.max_multiplier = max_multiplier
        
        # 投注状态
        self.balance = 0
        self.min_balance = 0
        self.max_drawdown = 0
        self.total_bet = 0
        self.total_profit = 0
        
        # 连败追踪
        self.consecutive_misses = 0
        self.recent_history = deque(maxlen=10)  # 最近10期的命中情况
        
        # 统计数据
        self.hit_count = 0
        self.miss_count = 0
        self.period_count = 0
        
        # 连败倍数映射（基于历史数据分析）
        self.streak_multipliers = {
            0: 1.0,   # 上期命中，正常投注
            1: 1.0,   # 连败1次，正常投注
            2: 1.5,   # 连败2次，小幅加注
            3: 2.0,   # 连败3次，命中率45.5%，明显加注
            4: 2.5,   # 连败4次，继续加注
            5: 3.0,   # 连败5次，命中率63.6%，重点加注！
            6: 3.0,   # 连败6次，保持高倍
            7: 3.5,   # 连败7次，进一步加注
            8: 4.0,   # 连败8次，接近极限
            9: 4.5,   # 连败9次
            10: 5.0   # 连败10次及以上，最高倍数
        }
    
    def calculate_multiplier(self, date_str=None):
        """
        计算当前倍数
        
        参数:
            date_str: 日期字符串（格式：2025/3/5）
        
        返回:
            倍数（float）
        """
        # 基础倍数：根据连败次数
        base_mult = self.streak_multipliers.get(
            min(self.consecutive_misses, 10), 
            5.0
        )
        
        # 时间调整因子
        time_factor = 1.0
        if date_str:
            try:
                date = pd.to_datetime(date_str)
                weekday = date.dayofweek
                month = date.month
                
                # 周三(2)和周六(5)命中率更高，略微降低倍数
                if weekday in [2, 5]:
                    time_factor = 0.95
                
                # 2月、4月、5月命中率更高，略微降低倍数
                if month in [2, 4, 5]:
                    time_factor *= 0.95
                
                # 周一(0)命中率较低，略微提高倍数
                if weekday == 0:
                    time_factor = 1.05
                
            except:
                pass
        
        # 计算最终倍数
        final_mult = base_mult * time_factor
        
        # 应用最大倍数限制
        final_mult = min(final_mult, self.max_multiplier)
        
        return round(final_mult, 1)
    
    def process_period(self, hit, date_str=None):
        """
        处理一期投注
        
        参数:
            hit: 是否命中（bool）
            date_str: 日期字符串
        
        返回:
            dict: {
                'multiplier': 倍数,
                'bet': 投注额,
                'profit': 本期盈亏,
                'consecutive_misses': 当前连败次数,
                'balance': 当前余额,
                'drawdown': 当前回撤,
                'recommendation': 建议等级
            }
        """
        self.period_count += 1
        
        # 计算倍数和投注额
        multiplier = self.calculate_multiplier(date_str)
        bet = self.base_bet * multiplier
        
        # 计算盈亏
        if hit:
            reward = self.base_reward * multiplier
            profit = reward - bet
            self.balance += profit
            self.hit_count += 1
            self.consecutive_misses = 0  # 重置连败
            recommendation = "✅ 命中"
        else:
            profit = -bet
            self.balance += profit
            self.miss_count += 1
            self.consecutive_misses += 1
            
            # 根据连败次数给出建议
            if self.consecutive_misses >= 5:
                recommendation = "⚠️ 连败5+次，下期命中率63.6%！"
            elif self.consecutive_misses >= 3:
                recommendation = "⚠️ 连败3+次，下期命中率45.5%"
            else:
                recommendation = "❌ 未中"
        
        # 更新统计
        self.total_bet += bet
        self.total_profit += profit
        self.recent_history.append(hit)
        
        # 更新回撤
        if self.balance < self.min_balance:
            self.min_balance = self.balance
            self.max_drawdown = abs(self.min_balance)
        
        return {
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'consecutive_misses': self.consecutive_misses,
            'balance': self.balance,
            'drawdown': self.max_drawdown,
            'recommendation': recommendation
        }
    
    def get_statistics(self):
        """获取统计数据"""
        hit_rate = self.hit_count / self.period_count if self.period_count > 0 else 0
        roi = self.total_profit / self.total_bet if self.total_bet > 0 else 0
        
        # 计算近期命中率
        recent_hit_rate = 0
        if len(self.recent_history) > 0:
            recent_hit_rate = sum(self.recent_history) / len(self.recent_history)
        
        return {
            'period_count': self.period_count,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'recent_hit_rate': recent_hit_rate,
            'total_bet': self.total_bet,
            'total_profit': self.total_profit,
            'balance': self.balance,
            'max_drawdown': self.max_drawdown,
            'roi': roi,
            'consecutive_misses': self.consecutive_misses
        }
    
    def get_next_period_advice(self, date_str=None):
        """
        获取下一期投注建议
        
        参数:
            date_str: 预测期的日期
        
        返回:
            dict: 建议信息
        """
        multiplier = self.calculate_multiplier(date_str)
        bet = self.base_bet * multiplier
        
        # 预估命中率（基于历史分析）
        if self.consecutive_misses >= 5:
            estimated_hit_rate = 0.636
            confidence = "⭐⭐⭐"
        elif self.consecutive_misses >= 3:
            estimated_hit_rate = 0.455
            confidence = "⭐⭐"
        else:
            estimated_hit_rate = 0.398
            confidence = "⭐"
        
        # 时间因素调整
        time_note = ""
        if date_str:
            try:
                date = pd.to_datetime(date_str)
                weekday = date.dayofweek
                month = date.month
                
                weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
                time_note = f"{weekday_names[weekday]}"
                
                if weekday in [2, 5]:  # 周三、周六
                    time_note += " (历史命中率较高)"
                    estimated_hit_rate *= 1.1
                elif weekday == 0:  # 周一
                    time_note += " (历史命中率较低)"
                    estimated_hit_rate *= 0.9
                
                if month in [2, 4, 5]:
                    time_note += f" {month}月 (月份命中率较高)"
                    estimated_hit_rate *= 1.05
                    
            except:
                pass
        
        estimated_hit_rate = min(estimated_hit_rate, 0.85)  # 最高不超过85%
        
        return {
            'multiplier': multiplier,
            'bet': bet,
            'estimated_hit_rate': estimated_hit_rate,
            'confidence': confidence,
            'consecutive_misses': self.consecutive_misses,
            'time_note': time_note,
            'advice': self._generate_advice_text(multiplier, estimated_hit_rate)
        }
    
    def _generate_advice_text(self, multiplier, hit_rate):
        """生成建议文本"""
        if multiplier >= 3.0:
            return f"🔥 强烈建议投注 {multiplier}x！预估命中率 {hit_rate:.1%}"
        elif multiplier >= 2.0:
            return f"⚡ 建议加大投注 {multiplier}x，预估命中率 {hit_rate:.1%}"
        else:
            return f"💡 常规投注 {multiplier}x，预估命中率 {hit_rate:.1%}"


def validate_strategy(test_periods=300):
    """验证策略效果"""
    print("="*70)
    print(" "*20 + "连败感知策略验证")
    print("="*70)
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    
    # 生成生肖TOP5预测（使用简单的历史高频方法）
    from zodiac_simple_smart import ZodiacSimpleSmart
    predictor = ZodiacSimpleSmart()
    
    all_numbers = df['number'].tolist()
    all_animals = df['animal'].tolist()
    all_dates = df['date'].dt.strftime('%Y/%m/%d').tolist()
    
    # 初始化策略
    strategy = StreakAwareBettingStrategy(base_bet=20, base_reward=47, max_multiplier=10)
    
    # 用于对比的基础固定倍投策略
    baseline_balance = 0
    baseline_consecutive_losses = 0
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    fib_index = 0
    
    # 回测
    lookback = 50
    results = []
    
    print(f"\n正在回测最近 {test_periods} 期...")
    print(f"数据范围: {all_dates[-test_periods-lookback]} ~ {all_dates[-1]}")
    print("\n" + "-"*70)
    
    for i in range(len(all_numbers) - test_periods, len(all_numbers)):
        train_numbers = all_numbers[:i]
        train_animals = all_animals[:i]
        actual_animal = all_animals[i]
        date_str = all_dates[i]
        
        # 生成预测
        prediction_result = predictor.predict_from_history(train_animals, top_n=5)
        top5_predictions = prediction_result['top5']
        
        # 判断命中
        hit = actual_animal in top5_predictions
        
        # 连败感知策略处理
        result = strategy.process_period(hit, date_str)
        
        # 基础斐波那契策略（用于对比）
        baseline_mult = min(fib_sequence[fib_index], 10)
        baseline_bet = 20 * baseline_mult
        if hit:
            baseline_profit = 47 * baseline_mult - baseline_bet
            baseline_balance += baseline_profit
            fib_index = 0  # 重置
            baseline_consecutive_losses = 0
        else:
            baseline_profit = -baseline_bet
            baseline_balance += baseline_profit
            fib_index = min(fib_index + 1, len(fib_sequence) - 1)
            baseline_consecutive_losses += 1
        
        # 记录详情
        result['date'] = date_str
        result['actual'] = actual_animal
        result['predictions'] = ','.join(top5_predictions)
        result['hit'] = '✅' if hit else '❌'
        result['baseline_mult'] = baseline_mult
        result['baseline_balance'] = baseline_balance
        
        results.append(result)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 显示前20期和后20期
    print("\n【前20期详情】")
    print(results_df.head(20)[['date', 'hit', 'consecutive_misses', 'multiplier', 'bet', 'profit', 'balance', 'recommendation']].to_string(index=False))
    
    print("\n" + "."*70)
    print("\n【后20期详情】")
    print(results_df.tail(20)[['date', 'hit', 'consecutive_misses', 'multiplier', 'bet', 'profit', 'balance', 'recommendation']].to_string(index=False))
    
    # 统计对比
    stats = strategy.get_statistics()
    baseline_roi = (baseline_balance / (20 * test_periods)) * 100
    
    print("\n" + "="*70)
    print("【策略对比】")
    print("="*70)
    
    print(f"\n连败感知策略:")
    print(f"  总期数: {stats['period_count']}")
    print(f"  命中次数: {stats['hit_count']} ({stats['hit_rate']:.1%})")
    print(f"  总投注: {stats['total_bet']:.0f}元")
    print(f"  净利润: {stats['total_profit']:+.0f}元")
    print(f"  ROI: {stats['roi']:.2%}")
    print(f"  最大回撤: {stats['max_drawdown']:.0f}元")
    print(f"  风险收益比: {stats['total_profit']/stats['max_drawdown']:.2f}" if stats['max_drawdown'] > 0 else "  风险收益比: N/A")
    
    print(f"\n基础斐波那契策略:")
    print(f"  总期数: {test_periods}")
    print(f"  净利润: {baseline_balance:+.0f}元")
    print(f"  ROI: {baseline_roi:.2%}")
    
    print("\n" + "="*70)
    improvement = stats['total_profit'] - baseline_balance
    print(f"💡 收益提升: {improvement:+.0f}元 ({(improvement/abs(baseline_balance)*100):+.1f}%)" if baseline_balance != 0 else f"💡 收益提升: {improvement:+.0f}元")
    print("="*70)
    
    # 保存详细结果
    output_file = 'streak_aware_strategy_300periods.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 详细结果已保存到: {output_file}")
    
    # 返回下一期建议
    print("\n" + "="*70)
    print("【下一期投注建议】")
    print("="*70)
    
    # 假设下一期是明天
    from datetime import timedelta
    next_date = (pd.to_datetime(all_dates[-1]) + timedelta(days=1)).strftime('%Y/%m/%d')
    advice = strategy.get_next_period_advice(next_date)
    
    print(f"\n预测日期: {next_date} {advice['time_note']}")
    print(f"当前连败: {advice['consecutive_misses']}次")
    print(f"建议倍数: {advice['multiplier']}x")
    print(f"建议投注: {advice['bet']:.0f}元")
    print(f"预估命中率: {advice['estimated_hit_rate']:.1%} {advice['confidence']}")
    print(f"投注建议: {advice['advice']}")
    print("\n" + "="*70)


if __name__ == '__main__':
    validate_strategy(test_periods=300)
