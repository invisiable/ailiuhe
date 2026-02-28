"""
完整检查300期中所有触发10倍的情况
包括所有可能的触发路径
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor


def find_all_10x_triggers():
    """找出所有触发10倍的期数"""
    
    print("=" * 100)
    print("完整扫描：最近300期所有10倍触发情况")
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
    
    all_10x_triggers = []
    
    for i in range(start_idx, len(df)):
        period_num = i - start_idx + 1
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        date = df.iloc[i]['date']
        hit = actual in predictions
        
        predictor.update_performance(predictions, actual)
        
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
        
        # 检查是否触发10倍（包括等于和大于）
        hit_10x = multiplier >= 9.99  # 使用浮点数比较，防止精度问题
        
        if hit_10x:
            # 找出连败情况
            consecutive_losses = 0
            for r in reversed(results):
                if not r['hit']:
                    consecutive_losses += 1
                else:
                    break
            
            trigger_info = {
                'period': period_num,
                'date': date,
                'actual': actual,
                'predictions': predictions[:5],
                'hit': hit,
                'fib_index': fib_index if not hit else 0,
                'base_mult': base_mult,
                'multiplier_before_cap': multiplier_before_cap,
                'adjustment_type': adjustment_type,
                'multiplier': multiplier,
                'bet': bet,
                'profit': profit,
                'balance': balance,
                'recent_rate': recent_rate,
                'consecutive_losses': consecutive_losses,
                'recent_8': list(recent_results[-8:]) if len(recent_results) >= 8 else list(recent_results)
            }
            all_10x_triggers.append(trigger_info)
        
        results.append({
            'period': period_num,
            'date': date,
            'actual': actual,
            'hit': hit,
            'multiplier': multiplier,
            'profit': profit,
            'balance': balance
        })
    
    # 输出所有触发
    print("=" * 100)
    print(f"【找到的所有10倍触发】共 {len(all_10x_triggers)} 次")
    print("=" * 100)
    print()
    
    if not all_10x_triggers:
        print("✅ 300期内未触发10倍上限")
        return
    
    # 详细列表
    print(f"{'期号':<6} {'日期':<12} {'开奖':<6} {'连败':<6} {'Fib':<5} {'基础':<8} {'调整类型':<12} {'调整前':<10} {'最终':<8} {'结果':<6} {'盈亏':<10} {'余额':<10}")
    print("-" * 100)
    
    for t in all_10x_triggers:
        result_str = "✅命中" if t['hit'] else "❌未中"
        print(f"{t['period']:<6} {t['date']:<12} {t['actual']:<6} {t['consecutive_losses']:<6} "
              f"{t['fib_index']:<5} {t['base_mult']:<8.2f} {t['adjustment_type']:<12} "
              f"{t['multiplier_before_cap']:<10.2f} {t['multiplier']:<8.2f} {result_str:<6} "
              f"{t['profit']:>+10.0f} {t['balance']:>+10.0f}")
    
    print()
    
    # 统计分析
    print("=" * 100)
    print("【触发统计】")
    print("=" * 100)
    print()
    
    print(f"触发频率: {len(all_10x_triggers)}/{test_periods}期 ({len(all_10x_triggers)/test_periods*100:.1f}%)")
    
    # 命中情况
    hits = sum(1 for t in all_10x_triggers if t['hit'])
    hit_rate = hits / len(all_10x_triggers) if all_10x_triggers else 0
    print(f"触发时命中率: {hit_rate*100:.1f}% ({hits}/{len(all_10x_triggers)}次)")
    
    # 盈亏统计
    total_profit = sum(t['profit'] for t in all_10x_triggers)
    wins_profit = sum(t['profit'] for t in all_10x_triggers if t['hit'])
    losses = sum(t['profit'] for t in all_10x_triggers if not t['hit'])
    
    print(f"10倍总盈亏: {total_profit:+.0f}元")
    print(f"  命中收益: +{wins_profit:.0f}元")
    print(f"  未中损失: {losses:.0f}元")
    print()
    
    # 触发原因分类
    fib_caused = sum(1 for t in all_10x_triggers if t['base_mult'] >= 10)
    boost_caused = sum(1 for t in all_10x_triggers if t['base_mult'] < 10 and '增强' in t['adjustment_type'])
    
    print(f"触发原因:")
    print(f"  Fibonacci基础≥10x: {fib_caused}次 ({fib_caused/len(all_10x_triggers)*100:.0f}%)")
    print(f"  动态增强后≥10x: {boost_caused}次 ({boost_caused/len(all_10x_triggers)*100:.0f}%)")
    print()
    
    # 连败分析
    max_consecutive = max(t['consecutive_losses'] for t in all_10x_triggers)
    avg_consecutive = np.mean([t['consecutive_losses'] for t in all_10x_triggers])
    
    print(f"连败情况:")
    print(f"  最长连败: {max_consecutive}期")
    print(f"  平均连败: {avg_consecutive:.1f}期")
    print()
    
    # 倍数分布
    fib_indices = [t['fib_index'] for t in all_10x_triggers]
    print(f"Fibonacci索引分布: {sorted(set(fib_indices))}")
    print()
    
    # 按日期分组
    print("=" * 100)
    print("【按月份统计】")
    print("=" * 100)
    print()
    
    from collections import defaultdict
    import re
    
    monthly_triggers = defaultdict(list)
    for t in all_10x_triggers:
        date_match = re.match(r'(\d{4})/(\d{1,2})/\d{1,2}', str(t['date']))
        if date_match:
            year_month = f"{date_match.group(1)}年{int(date_match.group(2))}月"
            monthly_triggers[year_month].append(t)
    
    for month in sorted(monthly_triggers.keys()):
        triggers = monthly_triggers[month]
        month_hits = sum(1 for t in triggers if t['hit'])
        month_profit = sum(t['profit'] for t in triggers)
        
        print(f"{month}: {len(triggers)}次触发, 命中{month_hits}次, 盈亏{month_profit:+.0f}元")
        for t in triggers:
            result_str = "✅" if t['hit'] else "❌"
            print(f"  - {t['date']} 第{t['period']}期: {result_str} {t['profit']:+.0f}元 (连败{t['consecutive_losses']}期)")
    
    print()
    
    # 整体表现对比
    overall_hit_rate = sum(1 for r in results if r['hit']) / len(results)
    max_drawdown = abs(min(r['balance'] for r in results))
    final_balance = results[-1]['balance']
    roi = (final_balance / total_bet * 100) if total_bet > 0 else 0
    
    print("=" * 100)
    print("【整体表现】")
    print("=" * 100)
    print()
    print(f"总命中率: {overall_hit_rate*100:.2f}%")
    print(f"10倍命中率: {hit_rate*100:.2f}% (差{(hit_rate-overall_hit_rate)*100:+.2f}%)")
    print(f"ROI: {roi:.2f}%")
    print(f"净收益: {final_balance:+.0f}元")
    print(f"最大回撤: {max_drawdown:.0f}元")
    print(f"10倍损失占回撤: {abs(losses)/max_drawdown*100:.1f}%")
    print()


if __name__ == '__main__':
    find_all_10x_triggers()
