"""
完整扫描所有触发≥8倍的情况（包括接近10倍的）
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor


def find_all_high_multipliers():
    """找出所有高倍数投注（≥8x）"""
    
    print("=" * 100)
    print("完整扫描：最近300期所有高倍数投注（≥8倍）")
    print("=" * 100)
    print()
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    test_periods = 300
    start_idx = len(df) - test_periods
    
    print(f"数据范围: {df.iloc[start_idx]['date']} ~ {df.iloc[-1]['date']}")
    print(f"总期数: {test_periods}期")
    print()
    
    # 初始化预测器
    predictor = PreciseTop15Predictor()
    
    # 智能动态配置
    config = {
        'base_bet': 15,
        'win_reward': 45,
        'lookback': 8,
        'good_thresh': 0.35,
        'bad_thresh': 0.20,
        'boost_mult': 1.5,
        'reduce_mult': 0.6,
        'max_multiplier': 10
    }
    
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    # 回测
    results = []
    recent_results = []
    balance = 0
    total_bet = 0
    fib_index = 0
    
    all_high_mult = []
    
    for i in range(start_idx, len(df)):
        period_num = i - start_idx + 1
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        date = df.iloc[i]['date']
        hit = actual in predictions
        
        predictor.update_performance(predictions, actual)
        
        # ===== 在投注前计算连败 =====
        consecutive_losses_before = 0
        for r in reversed(recent_results):
            if r == 0:
                consecutive_losses_before += 1
            else:
                break
        
        # 计算倍数
        if fib_index >= len(fib_sequence):
            base_mult = min(fib_sequence[-1], config['max_multiplier'])
        else:
            base_mult = min(fib_sequence[fib_index], config['max_multiplier'])
        
        # 动态调整
        adjustment_type = "无调整"
        multiplier_before_cap = base_mult
        
        if len(recent_results) >= config['lookback']:
            recent_hits = sum(recent_results[-config['lookback']:])
            rate = recent_hits / config['lookback']
            
            if rate >= config['good_thresh']:
                multiplier_before_cap = base_mult * config['boost_mult']
                adjustment_type = f"增强×{config['boost_mult']}"
                multiplier = min(multiplier_before_cap, config['max_multiplier'])
            elif rate <= config['bad_thresh']:
                multiplier_before_cap = base_mult * config['reduce_mult']
                adjustment_type = f"降低×{config['reduce_mult']}"
                multiplier = max(multiplier_before_cap, 1)
            else:
                multiplier = base_mult
                adjustment_type = "保持基础"
        else:
            multiplier = base_mult
            adjustment_type = "数据不足"
        
        bet = config['base_bet'] * multiplier
        total_bet += bet
        
        if hit:
            win = config['win_reward'] * multiplier
            profit = win - bet
            balance += profit
            fib_index = 0
            recent_results.append(1)
        else:
            profit = -bet
            balance += profit
            fib_index += 1
            recent_results.append(0)
        
        if len(recent_results) >= config['lookback']:
            recent_rate = sum(recent_results[-config['lookback']:]) / config['lookback']
        else:
            recent_rate = sum(recent_results) / len(recent_results) if recent_results else 0
        
        # 检查是否≥8倍
        if multiplier >= 8:
            trigger_info = {
                'period': period_num,
                'date': date,
                'actual': actual,
                'predictions': predictions[:5],
                'hit': hit,
                'fib_index': fib_index - 1 if not hit else 0,  # 显示投注时的索引
                'base_mult': base_mult,
                'multiplier_before_cap': multiplier_before_cap,
                'adjustment_type': adjustment_type,
                'multiplier': multiplier,
                'bet': bet,
                'profit': profit,
                'balance': balance,
                'recent_rate': recent_rate,
                'consecutive_losses': consecutive_losses_before,
                'recent_8': list(recent_results[-8:]) if len(recent_results) >= 8 else list(recent_results)
            }
            all_high_mult.append(trigger_info)
        
        results.append({
            'period': period_num,
            'date': date,
            'actual': actual,
            'hit': hit,
            'multiplier': multiplier,
            'profit': profit,
            'balance': balance
        })
    
    # 输出所有高倍数
    print("=" * 100)
    print(f"【所有≥8倍的投注】共 {len(all_high_mult)} 次")
    print("=" * 100)
    print()
    
    if not all_high_mult:
        print("✅ 300期内未触发8倍及以上")
        return
    
    # 统计≥10倍
    at_10x = [t for t in all_high_mult if t['multiplier'] >= 9.99]
    at_8_9x = [t for t in all_high_mult if 8 <= t['multiplier'] < 9.99]
    
    print(f"达到10倍上限: {len(at_10x)}次")
    print(f"8-9.9倍范围: {len(at_8_9x)}次")
    print()
    
    # 详细列表
    print(f"{'期号':<6} {'日期':<12} {'开奖':<6} {'连败':<6} {'Fib':<5} {'基础':<8} {'调整类型':<12} {'调整前':<10} {'最终':<8} {'结果':<6} {'盈亏':<10} {'余额':<10}")
    print("-" * 115)
    
    for t in all_high_mult:
        result_str = "✅命中" if t['hit'] else "❌未中"
        marker = "🔴" if t['multiplier'] >= 9.99 else "🟡"
        
        print(f"{marker} {t['period']:<6} {t['date']:<12} {t['actual']:<6} {t['consecutive_losses']:<6} "
              f"{t['fib_index']:<5} {t['base_mult']:<8.2f} {t['adjustment_type']:<12} "
              f"{t['multiplier_before_cap']:<10.2f} {t['multiplier']:<8.2f} {result_str:<6} "
              f"{t['profit']:>+10.0f} {t['balance']:>+10.0f}")
    
    print()
    print("🔴 = 达到10倍上限")
    print("🟡 = 8-9.9倍")
    print()
    
    # 统计分析
    print("=" * 100)
    print("【高倍数统计】")
    print("=" * 100)
    print()
    
    print(f"≥8倍频率: {len(all_high_mult)}/{test_periods}期 ({len(all_high_mult)/test_periods*100:.1f}%)")
    print(f"其中≥10倍: {len(at_10x)}次 ({len(at_10x)/test_periods*100:.1f}%)")
    print()
    
    # 命中情况
    hits_all = sum(1 for t in all_high_mult if t['hit'])
    hit_rate_all = hits_all / len(all_high_mult) if all_high_mult else 0
    
    hits_10x = sum(1 for t in at_10x if t['hit'])
    hit_rate_10x = hits_10x / len(at_10x) if at_10x else 0
    
    hits_8_9x = sum(1 for t in at_8_9x if t['hit'])
    hit_rate_8_9x = hits_8_9x / len(at_8_9x) if at_8_9x else 0
    
    print(f"≥8倍命中率: {hit_rate_all*100:.1f}% ({hits_all}/{len(all_high_mult)}次)")
    print(f"  10倍命中率: {hit_rate_10x*100:.1f}% ({hits_10x}/{len(at_10x)}次)")
    print(f"  8-9.9倍命中率: {hit_rate_8_9x*100:.1f}% ({hits_8_9x}/{len(at_8_9x)}次)")
    print()
    
    # 盈亏统计
    total_profit_all = sum(t['profit'] for t in all_high_mult)
    total_profit_10x = sum(t['profit'] for t in at_10x)
    total_profit_8_9x = sum(t['profit'] for t in at_8_9x)
    
    print(f"≥8倍总盈亏: {total_profit_all:+.0f}元")
    print(f"  10倍盈亏: {total_profit_10x:+.0f}元")
    print(f"  8-9.9倍盈亏: {total_profit_8_9x:+.0f}元")
    print()
    
    # 连败分析
    max_consecutive = max(t['consecutive_losses'] for t in all_high_mult)
    avg_consecutive = np.mean([t['consecutive_losses'] for t in all_high_mult])
    
    print(f"连败情况:")
    print(f"  最长连败: {max_consecutive}期")
    print(f"  平均连败: {avg_consecutive:.1f}期")
    print()
    
    # 触发原因分类
    fib_caused_10x = sum(1 for t in at_10x if t['base_mult'] >= 10)
    boost_caused_10x = sum(1 for t in at_10x if t['base_mult'] < 10 and '增强' in t['adjustment_type'])
    
    if at_10x:
        print(f"10倍触发原因:")
        print(f"  Fibonacci基础≥10x: {fib_caused_10x}次 ({fib_caused_10x/len(at_10x)*100:.0f}%)")
        print(f"  动态增强后≥10x: {boost_caused_10x}次 ({boost_caused_10x/len(at_10x)*100:.0f}%)")
        print()
    
    # 整体表现对比
    overall_hit_rate = sum(1 for r in results if r['hit']) / len(results)
    max_drawdown = abs(min(r['balance'] for r in results))
    final_balance = results[-1]['balance']
    roi = (final_balance / total_bet * 100) if total_bet > 0 else 0
    
    print("=" * 100)
    print("【整体表现对比】")
    print("=" * 100)
    print()
    print(f"总命中率: {overall_hit_rate*100:.2f}%")
    print(f"≥8倍命中率: {hit_rate_all*100:.2f}% (差{(hit_rate_all-overall_hit_rate)*100:+.2f}%)")
    print(f"10倍命中率: {hit_rate_10x*100:.2f}% (差{(hit_rate_10x-overall_hit_rate)*100:+.2f}%)")
    print(f"ROI: {roi:.2f}%")
    print(f"净收益: {final_balance:+.0f}元")
    print(f"最大回撤: {max_drawdown:.0f}元")
    
    if at_10x:
        losses_10x = abs(sum(t['profit'] for t in at_10x if t['profit'] < 0))
        print(f"10倍损失占回撤: {losses_10x/max_drawdown*100:.1f}%")
    
    print()


if __name__ == '__main__':
    find_all_high_multipliers()
