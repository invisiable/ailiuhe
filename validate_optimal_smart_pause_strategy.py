"""
验证最优智能投注策略 - 命中1期停1期暂停逻辑
测试目标：验证暂停策略是否能提升收益及减低回撤
"""
import sys
import io
import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

# 设置UTF-8编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


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
        # 获取基础倍数
        base_mult = self.get_base_multiplier()
        
        # 根据最近命中率计算动态倍数（使用投注前的历史数据）
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
        
        # 计算投注和收益
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
        
        # 添加结果到历史（在投注和结算之后）
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
    history = []
    consecutive_losses = 0
    max_consecutive_losses = 0
    hit_10x_count = 0
    
    for entry in hit_sequence:
        hit = entry['hit']
        # 保存投注前的fib_index（这是本期投注实际使用的索引）
        betting_fib_index = strategy.fib_index
        
        result = strategy.process_period(hit)
        
        if hit:
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        if result['multiplier'] >= config['max_multiplier']:
            hit_10x_count += 1
        
        history.append({
            **entry,
            **result,
            'fib_index': betting_fib_index,  # 记录投注时的索引
            'cumulative_profit': strategy.balance
        })
    
    wins = sum(1 for h in history if h['hit'])
    losses = len(history) - wins
    hit_rate = wins / len(history) if history else 0
    roi = (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0
    
    return {
        'history': history,
        'total_periods': len(history),
        'bet_periods': len(history),
        'wins': wins,
        'losses': losses,
        'hit_rate': hit_rate,
        'total_cost': strategy.total_bet,
        'total_win': strategy.total_win,
        'total_profit': strategy.balance,
        'roi': roi,
        'max_drawdown': strategy.max_drawdown,
        'max_consecutive_losses': max_consecutive_losses,
        'hit_10x_count': hit_10x_count
    }


def simulate_pause_strategy(hit_sequence, config, pause_length=1):
    """命中1停N期策略"""
    strategy = SmartDynamicStrategy(config)
    history = []
    pause_remaining = 0
    pause_trigger_count = 0
    paused_hit_count = 0
    pause_periods = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    hit_10x_count = 0
    
    for entry in hit_sequence:
        period = entry['period']
        date = entry['date']
        actual = entry['actual']
        predictions = entry['predictions']
        hit = entry['hit']
        pred_str = str(predictions[:5]) + "..."
        
        # 检查是否在暂停期
        if pause_remaining > 0:
            pause_remaining -= 1
            pause_periods += 1
            if hit:
                paused_hit_count += 1
            
            history.append({
                'period': period,
                'date': date,
                'actual': actual,
                'predictions': predictions,
                'predictions_str': pred_str,
                'hit': hit,
                'multiplier': 0,
                'bet': 0,
                'profit': 0,
                'cumulative_profit': strategy.balance,
                'recent_rate': strategy.get_recent_rate(),
                'fib_index': strategy.fib_index,
                'result': 'SKIP',
                'paused': True,
                'pause_remaining': pause_remaining
            })
            continue
        
        # 正常投注
        # 保存投注前的fib_index（这是本期投注实际使用的索引）
        betting_fib_index = strategy.fib_index
        
        result = strategy.process_period(hit)
        status = 'WIN' if hit else 'LOSS'
        
        if hit:
            pause_remaining = pause_length
            pause_trigger_count += 1
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        if result['multiplier'] >= config['max_multiplier']:
            hit_10x_count += 1
        
        history.append({
            'period': period,
            'date': date,
            'actual': actual,
            'predictions': predictions,
            'predictions_str': pred_str,
            'hit': hit,
            'multiplier': result['multiplier'],
            'bet': result['bet'],
            'profit': result['profit'],
            'cumulative_profit': strategy.balance,
            'recent_rate': result['recent_rate'],
            'fib_index': betting_fib_index,  # 记录投注时的索引
            'result': status,
            'paused': False,
            'pause_remaining': pause_remaining
        })
    
    bet_periods = len(history) - pause_periods
    wins = sum(1 for h in history if h.get('result') == 'WIN')
    losses = sum(1 for h in history if h.get('result') == 'LOSS')
    hit_rate = wins / bet_periods if bet_periods > 0 else 0
    roi = (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0
    
    return {
        'history': history,
        'total_periods': len(history),
        'bet_periods': bet_periods,
        'pause_periods': pause_periods,
        'paused_hit_count': paused_hit_count,
        'pause_trigger_count': pause_trigger_count,
        'wins': wins,
        'losses': losses,
        'hit_rate': hit_rate,
        'total_cost': strategy.total_bet,
        'total_win': strategy.total_win,
        'total_profit': strategy.balance,
        'roi': roi,
        'max_drawdown': strategy.max_drawdown,
        'max_consecutive_losses': max_consecutive_losses,
        'hit_10x_count': hit_10x_count
    }


def validate_pause_strategy():
    """验证暂停策略效果"""
    print("=" * 80)
    print("最优智能投注策略 - 命中1期停1期暂停逻辑验证")
    print("=" * 80)
    print()
    
    # 最优策略配置
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
    
    print(f"【策略配置】")
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
    
    # 初始化预测器
    predictor = PreciseTop15Predictor()
    
    # 生成预测序列
    print(f"正在生成预测...")
    hit_sequence = []
    for i in range(start_idx, len(df)):
        period_num = i - start_idx + 1
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        date = df.iloc[i]['date']
        hit = actual in predictions
        
        predictor.update_performance(predictions, actual)
        
        hit_sequence.append({
            'period': period_num,
            'date': date,
            'actual': actual,
            'predictions': predictions,
            'hit': hit
        })
    
    print(f"✓ 预测完成")
    print()
    
    # 测试基础策略（无暂停）
    print("=" * 80)
    print("测试1：基础策略（无暂停）")
    print("=" * 80)
    base_result = simulate_base_strategy(hit_sequence, config)
    
    print(f"【投注统计】")
    print(f"  投注期数: {base_result['bet_periods']}期")
    print(f"  命中次数: {base_result['wins']}期")
    print(f"  命中率: {base_result['hit_rate']*100:.2f}%")
    print()
    
    print(f"【收益统计】")
    print(f"  总投注: {base_result['total_cost']:.0f}元")
    print(f"  总收益: {base_result['total_win']:.0f}元")
    print(f"  净利润: {base_result['total_profit']:+.0f}元")
    print(f"  ROI: {base_result['roi']:+.2f}%")
    print()
    
    print(f"【风险指标】")
    print(f"  最大回撤: {base_result['max_drawdown']:.0f}元")
    print(f"  最长连亏: {base_result['max_consecutive_losses']}期")
    print(f"  触及10倍: {base_result['hit_10x_count']}次")
    print()
    
    # 测试暂停策略
    print("=" * 80)
    print("测试2：命中1期停1期策略")
    print("=" * 80)
    pause_result = simulate_pause_strategy(hit_sequence, config, pause_length=1)
    
    print(f"【投注统计】")
    print(f"  总期数: {pause_result['total_periods']}期")
    print(f"  投注期数: {pause_result['bet_periods']}期")
    print(f"  暂停期数: {pause_result['pause_periods']}期")
    print(f"  暂停比例: {pause_result['pause_periods']/pause_result['total_periods']*100:.1f}%")
    print(f"  命中次数: {pause_result['wins']}期")
    print(f"  命中率: {pause_result['hit_rate']*100:.2f}%")
    print()
    
    print(f"【暂停效果】")
    print(f"  暂停触发: {pause_result['pause_trigger_count']}次（每次命中后暂停）")
    print(f"  暂停期漏中: {pause_result['paused_hit_count']}次")
    print(f"  漏中率: {pause_result['paused_hit_count']/pause_result['pause_periods']*100:.1f}%" if pause_result['pause_periods'] > 0 else "  漏中率: 0.0%")
    print()
    
    print(f"【收益统计】")
    print(f"  总投注: {pause_result['total_cost']:.0f}元")
    print(f"  总收益: {pause_result['total_win']:.0f}元")
    print(f"  净利润: {pause_result['total_profit']:+.0f}元")
    print(f"  ROI: {pause_result['roi']:+.2f}%")
    print()
    
    print(f"【风险指标】")
    print(f"  最大回撤: {pause_result['max_drawdown']:.0f}元")
    print(f"  最长连亏: {pause_result['max_consecutive_losses']}期")
    print(f"  触及10倍: {pause_result['hit_10x_count']}次")
    print()
    
    # 对比分析
    print("=" * 80)
    print("对比分析：暂停策略 vs 基础策略")
    print("=" * 80)
    
    # 计算差异
    profit_delta = pause_result['total_profit'] - base_result['total_profit']
    profit_delta_pct = (profit_delta / abs(base_result['total_profit']) * 100) if base_result['total_profit'] != 0 else 0
    roi_delta = pause_result['roi'] - base_result['roi']
    drawdown_delta = base_result['max_drawdown'] - pause_result['max_drawdown']
    drawdown_delta_pct = (drawdown_delta / base_result['max_drawdown'] * 100) if base_result['max_drawdown'] > 0 else 0
    cost_delta = base_result['total_cost'] - pause_result['total_cost']
    cost_delta_pct = (cost_delta / base_result['total_cost'] * 100) if base_result['total_cost'] > 0 else 0
    
    print(f"【收益对比】")
    print(f"  净利润差异: {profit_delta:+.0f}元 ({profit_delta_pct:+.1f}%)")
    if profit_delta > 0:
        print(f"    ✅ 暂停策略收益更高")
    elif profit_delta < 0:
        print(f"    ⚠️  基础策略收益更高")
    else:
        print(f"    ➖ 两者收益相同")
    print()
    
    print(f"  ROI差异: {roi_delta:+.2f}%")
    if roi_delta > 0:
        print(f"    ✅ 暂停策略ROI更高")
    elif roi_delta < 0:
        print(f"    ⚠️  基础策略ROI更高")
    else:
        print(f"    ➖ 两者ROI相同")
    print()
    
    print(f"【风险对比】")
    print(f"  回撤差异: {drawdown_delta:+.0f}元 ({drawdown_delta_pct:+.1f}%)")
    if drawdown_delta > 0:
        print(f"    ✅ 暂停策略回撤更低（风险降低）")
    elif drawdown_delta < 0:
        print(f"    ⚠️  基础策略回撤更低")
    else:
        print(f"    ➖ 两者回撤相同")
    print()
    
    print(f"【成本对比】")
    print(f"  投注成本差异: {cost_delta:+.0f}元 ({cost_delta_pct:+.1f}%)")
    print(f"    暂停策略投注期数: {pause_result['bet_periods']}期")
    print(f"    基础策略投注期数: {base_result['bet_periods']}期")
    print(f"    减少投注: {base_result['bet_periods'] - pause_result['bet_periods']}期")
    print()
    
    print(f"【综合评估】")
    
    # 综合得分
    score_profit = 1 if profit_delta > 0 else (-1 if profit_delta < 0 else 0)
    score_roi = 1 if roi_delta > 0 else (-1 if roi_delta < 0 else 0)
    score_drawdown = 1 if drawdown_delta > 0 else (-1 if drawdown_delta < 0 else 0)
    total_score = score_profit + score_roi + score_drawdown
    
    if total_score >= 2:
        conclusion = "✅ 暂停策略明显优于基础策略"
        recommendation = "强烈建议使用暂停策略"
    elif total_score == 1:
        conclusion = "🟡 暂停策略略优于基础策略"
        recommendation = "建议使用暂停策略"
    elif total_score == 0:
        conclusion = "➖ 两种策略表现相近"
        recommendation = "根据个人偏好选择"
    elif total_score == -1:
        conclusion = "🟡 基础策略略优于暂停策略"
        recommendation = "建议使用基础策略"
    else:
        conclusion = "⚠️  基础策略明显优于暂停策略"
        recommendation = "建议使用基础策略"
    
    print(f"  综合得分: {total_score}/3")
    print(f"  结论: {conclusion}")
    print(f"  建议: {recommendation}")
    print()
    
    # 详细说明
    print(f"【暂停策略优缺点分析】")
    print(f"  优点:")
    print(f"    • 减少投注频率，降低总成本 {cost_delta_pct:.1f}%")
    print(f"    • 在命中后暂停，避免连续小额亏损累积")
    print(f"    • 重置Fibonacci序列，从低倍数开始")
    if drawdown_delta > 0:
        print(f"    • 显著降低最大回撤 {drawdown_delta:.0f}元")
    
    print(f"  缺点:")
    print(f"    • 暂停期可能错过连续命中机会（漏中{pause_result['paused_hit_count']}次）")
    if profit_delta < 0:
        print(f"    • 可能减少总收益 {abs(profit_delta):.0f}元")
    print()
    
    # 保存详细结果到CSV
    print("=" * 80)
    print("保存详细结果...")
    
    # 保存基础策略历史
    base_df = pd.DataFrame(base_result['history'])
    base_df.to_csv('validate_optimal_smart_base_300periods.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 基础策略详情: validate_optimal_smart_base_300periods.csv")
    
    # 保存暂停策略历史
    pause_df = pd.DataFrame(pause_result['history'])
    pause_df.to_csv('validate_optimal_smart_pause_300periods.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 暂停策略详情: validate_optimal_smart_pause_300periods.csv")
    
    # 保存对比摘要
    summary = {
        '策略': ['基础策略', '暂停策略', '差异', '差异%'],
        '测试期数': [base_result['bet_periods'], pause_result['total_periods'], 
                   pause_result['total_periods'] - base_result['bet_periods'], ''],
        '投注期数': [base_result['bet_periods'], pause_result['bet_periods'], 
                   pause_result['bet_periods'] - base_result['bet_periods'],
                   f"{(pause_result['bet_periods'] - base_result['bet_periods']) / base_result['bet_periods'] * 100:.1f}%"],
        '命中次数': [base_result['wins'], pause_result['wins'],
                   pause_result['wins'] - base_result['wins'], ''],
        '命中率': [f"{base_result['hit_rate']*100:.2f}%", f"{pause_result['hit_rate']*100:.2f}%", '', ''],
        '总投注': [f"{base_result['total_cost']:.0f}", f"{pause_result['total_cost']:.0f}",
                 f"{pause_result['total_cost'] - base_result['total_cost']:+.0f}",
                 f"{cost_delta_pct:+.1f}%"],
        '净利润': [f"{base_result['total_profit']:+.0f}", f"{pause_result['total_profit']:+.0f}",
                 f"{profit_delta:+.0f}", f"{profit_delta_pct:+.1f}%"],
        'ROI': [f"{base_result['roi']:+.2f}%", f"{pause_result['roi']:+.2f}%",
               f"{roi_delta:+.2f}%", ''],
        '最大回撤': [f"{base_result['max_drawdown']:.0f}", f"{pause_result['max_drawdown']:.0f}",
                  f"{drawdown_delta:+.0f}", f"{drawdown_delta_pct:+.1f}%"],
        '最长连亏': [base_result['max_consecutive_losses'], pause_result['max_consecutive_losses'],
                  pause_result['max_consecutive_losses'] - base_result['max_consecutive_losses'], ''],
        '触及10倍': [base_result['hit_10x_count'], pause_result['hit_10x_count'],
                  pause_result['hit_10x_count'] - base_result['hit_10x_count'], '']
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('validate_optimal_smart_comparison.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 对比摘要: validate_optimal_smart_comparison.csv")
    print()
    
    print("=" * 80)
    print("验证完成！")
    print("=" * 80)
    print(f"最终建议: {recommendation}")
    print()


if __name__ == '__main__':
    validate_pause_strategy()
