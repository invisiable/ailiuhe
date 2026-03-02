"""
最优智能投注策略 - 暂停策略全面验证报告
测试不同暂停期长度：停1期、停2期、停3期
"""
import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor


# Fibonacci数列
fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


class SmartDynamicStrategy:
    """最优智能动态倍投策略"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.fib_index = 0
        self.recent_results = []
        self.total_bet = 0
        self.total_win = 0
        self.balance = 0
        self.min_balance = 0
        self.max_drawdown = 0
    
    def get_base_multiplier(self):
        if self.fib_index >= len(fib_sequence):
            return min(fib_sequence[-1], self.cfg['max_multiplier'])
        return min(fib_sequence[self.fib_index], self.cfg['max_multiplier'])
    
    def get_recent_rate(self):
        if len(self.recent_results) == 0:
            return 0.33
        return sum(self.recent_results) / len(self.recent_results)
    
    def process_period(self, hit):
        base_mult = self.get_base_multiplier()
        
        if len(self.recent_results) >= self.cfg['lookback']:
            rate = self.get_recent_rate()
            if rate >= self.cfg['good_thresh']:
                multiplier = min(base_mult * self.cfg['boost_mult'], self.cfg['max_multiplier'])
            elif rate <= self.cfg['bad_thresh']:
                multiplier = max(base_mult * self.cfg['reduce_mult'], 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        bet = self.cfg['base_bet'] * multiplier
        self.total_bet += bet
        
        if hit:
            win = self.cfg['win_reward'] * multiplier
            self.total_win += win
            profit = win - bet
            self.balance += profit
            self.fib_index = 0
        else:
            profit = -bet
            self.balance += profit
            self.fib_index += 1
            
            if self.balance < self.min_balance:
                self.min_balance = self.balance
                self.max_drawdown = abs(self.min_balance)
        
        self.recent_results.append(1 if hit else 0)
        if len(self.recent_results) > self.cfg['lookback']:
            self.recent_results.pop(0)
        
        return {
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'recent_rate': self.get_recent_rate()
        }


def simulate_base_strategy(hit_sequence, config):
    """基础策略（无暂停）"""
    strategy = SmartDynamicStrategy(config)
    total_periods = len(hit_sequence)
    consecutive_losses = 0
    max_consecutive_losses = 0
    hit_10x_count = 0
    
    for entry in hit_sequence:
        hit = entry['hit']
        result = strategy.process_period(hit)
        
        if hit:
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        if result['multiplier'] >= config['max_multiplier']:
            hit_10x_count += 1
    
    wins = sum(1 for entry in hit_sequence if entry['hit'])
    hit_rate = wins / total_periods if total_periods > 0 else 0
    roi = (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0
    
    return {
        'total_periods': total_periods,
        'bet_periods': total_periods,
        'pause_periods': 0,
        'wins': wins,
        'hit_rate': hit_rate,
        'total_cost': strategy.total_bet,
        'total_win': strategy.total_win,
        'total_profit': strategy.balance,
        'roi': roi,
        'max_drawdown': strategy.max_drawdown,
        'max_consecutive_losses': max_consecutive_losses,
        'hit_10x_count': hit_10x_count,
        'pause_trigger_count': 0,
        'paused_hit_count': 0
    }


def simulate_pause_strategy(hit_sequence, config, pause_length=1):
    """命中1停N期策略"""
    strategy = SmartDynamicStrategy(config)
    pause_remaining = 0
    pause_trigger_count = 0
    paused_hit_count = 0
    pause_periods = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    hit_10x_count = 0
    bet_periods = 0
    wins_in_betting = 0
    
    for entry in hit_sequence:
        hit = entry['hit']
        
        # 检查是否在暂停期
        if pause_remaining > 0:
            pause_remaining -= 1
            pause_periods += 1
            if hit:
                paused_hit_count += 1
            continue
        
        # 正常投注
        bet_periods += 1
        result = strategy.process_period(hit)
        
        if hit:
            wins_in_betting += 1
            pause_remaining = pause_length
            pause_trigger_count += 1
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        if result['multiplier'] >= config['max_multiplier']:
            hit_10x_count += 1
    
    hit_rate = wins_in_betting / bet_periods if bet_periods > 0 else 0
    roi = (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0
    
    return {
        'total_periods': len(hit_sequence),
        'bet_periods': bet_periods,
        'pause_periods': pause_periods,
        'wins': wins_in_betting,
        'hit_rate': hit_rate,
        'total_cost': strategy.total_bet,
        'total_win': strategy.total_win,
        'total_profit': strategy.balance,
        'roi': roi,
        'max_drawdown': strategy.max_drawdown,
        'max_consecutive_losses': max_consecutive_losses,
        'hit_10x_count': hit_10x_count,
        'pause_trigger_count': pause_trigger_count,
        'paused_hit_count': paused_hit_count
    }


def generate_report():
    """生成完整验证报告"""
    print("=" * 80)
    print("最优智能投注策略 - 暂停策略全面验证报告")
    print("=" * 80)
    print()
    
    # 策略配置
    config = {
        'name': '最优智能动态倍投策略 v3.1',
        'lookback': 12,
        'good_thresh': 0.35,
        'bad_thresh': 0.20,
        'boost_mult': 1.5,
        'reduce_mult': 0.5,
        'max_multiplier': 10,
        'base_bet': 15,
        'win_reward': 47
    }
    
    print("【策略配置】")
    print(f"  窗口期: {config['lookback']}期")
    print(f"  增强阈值: ≥{config['good_thresh']:.0%} × {config['boost_mult']}")
    print(f"  降低阈值: ≤{config['bad_thresh']:.0%} × {config['reduce_mult']}")
    print(f"  最大倍数: {config['max_multiplier']}倍")
    print(f"  基础投注: {config['base_bet']}元 | 中奖: {config['win_reward']}元")
    print()
    
    # 读取数据
    file_path = 'data/lucky_numbers.csv'
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    print(f"【数据加载】")
    print(f"  总期数: {len(df)}期")
    print(f"  日期范围: {df.iloc[0]['date']} 至 {df.iloc[-1]['date']}")
    print()
    
    # 测试期数
    test_periods = min(300, len(df) - 50)
    start_idx = len(df) - test_periods
    
    print(f"【回测设置】")
    print(f"  测试期数: {test_periods}期")
    print(f"  起始期数: 第{start_idx+1}期")
    print()
    
    # 生成预测序列
    print(f"正在生成预测...")
    predictor = PreciseTop15Predictor()
    hit_sequence = []
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        hit = actual in predictions
        predictor.update_performance(predictions, actual)
        
        hit_sequence.append({
            'period': i - start_idx + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'predictions': predictions,
            'hit': hit
        })
    
    print(f"✓ 预测完成")
    print()
    
    # 测试不同策略
    print("=" * 80)
    print("策略对比测试")
    print("=" * 80)
    print()
    
    strategies = [
        ("基础策略（无暂停）", None),
        ("命中1停1期", 1),
        ("命中1停2期", 2),
        ("命中1停3期", 3)
    ]
    
    results = {}
    
    for name, pause_length in strategies:
        print(f"测试: {name}...")
        
        if pause_length is None:
            result = simulate_base_strategy(hit_sequence, config)
        else:
            result = simulate_pause_strategy(hit_sequence, config, pause_length)
        
        results[name] = result
        print(f"  ✓ 完成")
    
    print()
    
    # 汇总对比表
    print("=" * 80)
    print("策略对比汇总")
    print("=" * 80)
    print()
    
    print(f"{'策略':<20} {'投注期':<8} {'暂停期':<8} {'命中率':<10} {'ROI':<10} {'净利润':<12} {'回撤':<10}")
    print("-" * 80)
    
    for name, pause_length in strategies:
        r = results[name]
        print(f"{name:<20} {r['bet_periods']:<8} {r['pause_periods']:<8} "
              f"{r['hit_rate']*100:<9.2f}% {r['roi']:<9.2f}% "
              f"{r['total_profit']:<+11.0f}元 {r['max_drawdown']:<9.0f}元")
    
    print()
    
    # 详细对比
    base_result = results["基础策略（无暂停）"]
    
    print("=" * 80)
    print("详细对比分析（相对基础策略）")
    print("=" * 80)
    print()
    
    for name, pause_length in strategies[1:]:  # 跳过基础策略
        r = results[name]
        
        print(f"【{name}】")
        
        # 收益对比
        profit_delta = r['total_profit'] - base_result['total_profit']
        profit_delta_pct = (profit_delta / abs(base_result['total_profit']) * 100) if base_result['total_profit'] != 0 else 0
        roi_delta = r['roi'] - base_result['roi']
        
        # 风险对比
        drawdown_delta = base_result['max_drawdown'] - r['max_drawdown']
        drawdown_delta_pct = (drawdown_delta / base_result['max_drawdown'] * 100) if base_result['max_drawdown'] > 0 else 0
        
        # 成本对比
        cost_delta = base_result['total_cost'] - r['total_cost']
        cost_delta_pct = (cost_delta / base_result['total_cost'] * 100) if base_result['total_cost'] > 0 else 0
        
        # 综合评分
        score_profit = 1 if profit_delta > 0 else (-1 if profit_delta < 0 else 0)
        score_roi = 1 if roi_delta > 0 else (-1 if roi_delta < 0 else 0)
        score_drawdown = 1 if drawdown_delta > 0 else (-1 if drawdown_delta < 0 else 0)
        total_score = score_profit + score_roi + score_drawdown
        
        print(f"  收益：")
        print(f"    净利润: {r['total_profit']:+.0f}元 ({profit_delta:+.0f}元, {profit_delta_pct:+.1f}%)")
        print(f"    ROI: {r['roi']:+.2f}% ({roi_delta:+.2f}%)")
        
        print(f"  风险：")
        print(f"    最大回撤: {r['max_drawdown']:.0f}元 ({drawdown_delta:+.0f}元, {drawdown_delta_pct:+.1f}%)")
        print(f"    最长连亏: {r['max_consecutive_losses']}期 (基础{base_result['max_consecutive_losses']}期)")
        print(f"    触及10倍: {r['hit_10x_count']}次 (基础{base_result['hit_10x_count']}次)")
        
        print(f"  成本：")
        print(f"    总投注: {r['total_cost']:.0f}元 ({cost_delta:+.0f}元, {cost_delta_pct:+.1f}%)")
        print(f"    投注期数: {r['bet_periods']}期（减少{base_result['bet_periods']-r['bet_periods']}期）")
        print(f"    暂停期数: {r['pause_periods']}期")
        
        print(f"  暂停效果：")
        print(f"    暂停触发: {r['pause_trigger_count']}次")
        print(f"    暂停期漏中: {r['paused_hit_count']}次")
        if r['pause_periods'] > 0:
            print(f"    漏中率: {r['paused_hit_count']/r['pause_periods']*100:.1f}%")
        
        print(f"  综合评分: {total_score}/3")
        
        if total_score >= 2:
            verdict = "✅ 明显优于基础策略"
        elif total_score == 1:
            verdict = "🟡 略优于基础策略"
        elif total_score == 0:
            verdict = "➖ 与基础策略相近"
        elif total_score == -1:
            verdict = "🟡 略逊于基础策略"
        else:
            verdict = "⚠️  明显逊于基础策略"
        
        print(f"  结论: {verdict}")
        print()
    
    # 最佳策略推荐
    print("=" * 80)
    print("最佳策略推荐")
    print("=" * 80)
    print()
    
    # 按不同维度排名
    sorted_by_profit = sorted(results.items(), key=lambda x: x[1]['total_profit'], reverse=True)
    sorted_by_roi = sorted(results.items(), key=lambda x: x[1]['roi'], reverse=True)
    sorted_by_drawdown = sorted(results.items(), key=lambda x: x[1]['max_drawdown'])
    
    # 计算风险调整收益
    risk_adjusted = {}
    for name, r in results.items():
        if r['max_drawdown'] > 0:
            risk_adjusted[name] = r['total_profit'] / r['max_drawdown']
        else:
            risk_adjusted[name] = r['total_profit']
    sorted_by_risk_adj = sorted(risk_adjusted.items(), key=lambda x: x[1], reverse=True)
    
    print("【各维度最佳策略】")
    print(f"  净利润最高: {sorted_by_profit[0][0]} ({sorted_by_profit[0][1]['total_profit']:+.0f}元)")
    print(f"  ROI最高: {sorted_by_roi[0][0]} ({sorted_by_roi[0][1]['roi']:+.2f}%)")
    print(f"  回撤最低: {sorted_by_drawdown[0][0]} ({sorted_by_drawdown[0][1]['max_drawdown']:.0f}元)")
    print(f"  风险调整收益最高: {sorted_by_risk_adj[0][0]} ({sorted_by_risk_adj[0][1]:.2f})")
    print()
    
    # 综合推荐
    print("【综合推荐】")
    
    best_strategy = sorted_by_risk_adj[0][0]
    best_result = results[best_strategy]
    
    print(f"  🏆 最佳策略: {best_strategy}")
    print(f"     净利润: {best_result['total_profit']:+.0f}元")
    print(f"     ROI: {best_result['roi']:+.2f}%")
    print(f"     最大回撤: {best_result['max_drawdown']:.0f}元")
    print(f"     风险调整收益: {risk_adjusted[best_strategy]:.2f}")
    print()
    
    # 保存结果
    print("=" * 80)
    print("保存结果...")
    print()
    
    summary_data = []
    for name, r in results.items():
        summary_data.append({
            '策略': name,
            '总期数': r['total_periods'],
            '投注期数': r['bet_periods'],
            '暂停期数': r['pause_periods'],
            '命中次数': r['wins'],
            '命中率': f"{r['hit_rate']*100:.2f}%",
            '总投注': f"{r['total_cost']:.0f}",
            '总收益': f"{r['total_win']:.0f}",
            '净利润': f"{r['total_profit']:+.0f}",
            'ROI': f"{r['roi']:+.2f}%",
            '最大回撤': f"{r['max_drawdown']:.0f}",
            '最长连亏': r['max_consecutive_losses'],
            '触及10倍': r['hit_10x_count'],
            '暂停触发': r['pause_trigger_count'],
            '暂停期漏中': r['paused_hit_count'],
            '风险调整收益': f"{risk_adjusted[name]:.2f}" if name in risk_adjusted else "0"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('optimal_smart_pause_strategies_comparison.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 对比汇总: optimal_smart_pause_strategies_comparison.csv")
    print()
    
    print("=" * 80)
    print("验证完成！")
    print("=" * 80)


if __name__ == '__main__':
    generate_report()
