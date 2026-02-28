"""
分析最优智能投注中10倍上限触发的情况
找出触发原因、影响和改进建议
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor


def analyze_10x_triggers():
    """分析10倍上限触发情况"""
    
    print("=" * 100)
    print("最优智能投注 - 10倍上限触发分析")
    print("=" * 100)
    print()
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"总数据: {len(df)}期")
    
    # 300期回测
    test_periods = 300
    start_idx = len(df) - test_periods
    print(f"分析期数: {test_periods}期")
    print(f"起始日期: {df.iloc[start_idx]['date']}")
    print(f"结束日期: {df.iloc[-1]['date']}")
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
    
    for i in range(start_idx, len(df)):
        period_num = i - start_idx + 1
        
        # 预测
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        date = df.iloc[i]['date']
        hit = actual in predictions
        
        # 更新预测器
        predictor.update_performance(predictions, actual)
        
        # 计算倍数
        if fib_index >= len(fib_sequence):
            base_mult = min(fib_sequence[-1], config['max_multiplier'])
        else:
            base_mult = min(fib_sequence[fib_index], config['max_multiplier'])
        
        # 动态调整
        if len(recent_results) >= config['lookback']:
            recent_hits = sum(recent_results[-config['lookback']:])
            rate = recent_hits / config['lookback']
            
            if rate >= config['good_thresh']:
                multiplier = min(base_mult * config['boost_mult'], config['max_multiplier'])
            elif rate <= config['bad_thresh']:
                multiplier = max(base_mult * config['reduce_mult'], 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        # 投注
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
        
        # 计算最近命中率
        if len(recent_results) >= config['lookback']:
            recent_rate = sum(recent_results[-config['lookback']:]) / config['lookback']
        else:
            recent_rate = sum(recent_results) / len(recent_results) if recent_results else 0
        
        # 判断是否触发10倍
        hit_10x = multiplier >= 10
        
        results.append({
            'period': period_num,
            'date': date,
            'actual': actual,
            'predictions': predictions,
            'hit': hit,
            'multiplier': multiplier,
            'base_mult': base_mult,
            'bet': bet,
            'profit': profit,
            'balance': balance,
            'recent_rate': recent_rate,
            'fib_index': fib_index if not hit else 0,
            'hit_10x': hit_10x
        })
    
    # 统计10倍触发情况
    triggers = [r for r in results if r['hit_10x']]
    
    print("=" * 100)
    print(f"【10倍上限触发统计】")
    print("=" * 100)
    print()
    
    print(f"触发次数: {len(triggers)}次 / {test_periods}期 ({len(triggers)/test_periods*100:.1f}%)")
    print()
    
    if not triggers:
        print("✅ 300期内未触发10倍上限，策略控制良好")
        return
    
    # 详细分析每次触发
    print("=" * 100)
    print("【每次触发详情】")
    print("=" * 100)
    print()
    
    for idx, trigger in enumerate(triggers, 1):
        period = trigger['period']
        print(f"\n第{idx}次触发 - 第{period}期 ({trigger['date']})")
        print("-" * 80)
        
        # 找到触发前的连败情况
        consecutive_losses = 0
        for r in reversed(results[:period-1]):
            if not r['hit']:
                consecutive_losses += 1
            else:
                break
        
        print(f"📍 基本信息:")
        print(f"  触发期号: 第{period}期")
        print(f"  日期: {trigger['date']}")
        print(f"  开奖号码: {trigger['actual']}")
        print(f"  预测TOP15: {trigger['predictions'][:5]}...")
        print()
        
        print(f"📊 触发原因:")
        print(f"  Fibonacci索引: {trigger['fib_index']}（基础倍数: {trigger['base_mult']:.2f}x）")
        print(f"  连续未中: {consecutive_losses}期")
        print(f"  最近{config['lookback']}期命中率: {trigger['recent_rate']*100:.1f}%")
        
        # 判断是否有动态调整
        if trigger['recent_rate'] >= config['good_thresh']:
            adjust_type = f"增强倍投（×{config['boost_mult']}）"
        elif trigger['recent_rate'] <= config['bad_thresh']:
            adjust_type = f"降低倍投（×{config['reduce_mult']}）"
        else:
            adjust_type = "保持基础倍数"
        
        print(f"  动态调整: {adjust_type}")
        print(f"  最终倍数: {trigger['multiplier']:.2f}x (触及上限)")
        print()
        
        print(f"💰 投注情况:")
        print(f"  投注金额: {trigger['bet']:.0f}元 (15元 × 10倍)")
        print(f"  命中结果: {'✅ 命中' if trigger['hit'] else '❌ 未中'}")
        print(f"  当期盈亏: {trigger['profit']:+.0f}元")
        print(f"  累计余额: {trigger['balance']:+.0f}元")
        print()
        
        # 分析触发前10期的状态
        start_period = max(0, period - 11)
        before_periods = results[start_period:period-1]
        
        if len(before_periods) >= 10:
            before_10 = before_periods[-10:]
            hits_before_10 = sum(1 for r in before_10 if r['hit'])
            rate_before_10 = hits_before_10 / 10
            
            print(f"📈 触发前10期表现:")
            print(f"  命中次数: {hits_before_10}/10期 ({rate_before_10*100:.1f}%)")
            
            # 显示最近10期详情
            print(f"  期号序列: ", end="")
            for r in before_10:
                symbol = "✓" if r['hit'] else "✗"
                print(f"{symbol}", end=" ")
            print()
            
            avg_mult_before = np.mean([r['multiplier'] for r in before_10])
            print(f"  平均倍数: {avg_mult_before:.2f}x")
            total_loss_before = sum(r['profit'] for r in before_10 if not r['hit'])
            print(f"  累计亏损: {total_loss_before:.0f}元")
        
        print()
        
        # 分析触发后的情况
        if period < len(results):
            after_5 = results[period:min(period+5, len(results))]
            if after_5:
                hits_after_5 = sum(1 for r in after_5 if r['hit'])
                rate_after_5 = hits_after_5 / len(after_5)
                
                print(f"📉 触发后5期表现:")
                print(f"  命中次数: {hits_after_5}/{len(after_5)}期 ({rate_after_5*100:.1f}%)")
                print(f"  期号序列: ", end="")
                for r in after_5:
                    symbol = "✓" if r['hit'] else "✗"
                    print(f"{symbol}", end=" ")
                print()
    
    print()
    print("=" * 100)
    print("【10倍触发综合分析】")
    print("=" * 100)
    print()
    
    # 统计触发时的命中情况
    trigger_hits = sum(1 for t in triggers if t['hit'])
    trigger_hit_rate = trigger_hits / len(triggers) if triggers else 0
    
    print(f"触发时命中率: {trigger_hit_rate*100:.1f}% ({trigger_hits}/{len(triggers)}次)")
    print()
    
    # 计算10倍触发对回撤的影响
    trigger_losses = [t for t in triggers if not t['hit']]
    total_loss_at_10x = sum(t['bet'] for t in trigger_losses)
    
    print(f"10倍触发造成的直接损失:")
    print(f"  未中次数: {len(trigger_losses)}次")
    print(f"  总损失: {total_loss_at_10x:.0f}元 (每次150元)")
    print()
    
    # 分析Fibonacci索引分布
    fib_indices = [t['fib_index'] for t in triggers]
    print(f"触发时的Fibonacci索引: {fib_indices}")
    print(f"  对应连败期数: {fib_indices} 期")
    print()
    
    # 分析动态调整效果
    boost_triggers = sum(1 for t in triggers if t['recent_rate'] >= config['good_thresh'])
    reduce_triggers = sum(1 for t in triggers if t['recent_rate'] <= config['bad_thresh'])
    normal_triggers = len(triggers) - boost_triggers - reduce_triggers
    
    print(f"动态调整状态分布:")
    print(f"  增强倍投: {boost_triggers}次 (命中率≥35%)")
    print(f"  正常倍投: {normal_triggers}次")
    print(f"  降低倍投: {reduce_triggers}次 (命中率≤20%)")
    print()
    
    # 分析是否有改进空间
    print("=" * 100)
    print("【改进建议】")
    print("=" * 100)
    print()
    
    # 建议1：检查是否需要更严格的上限
    if len(trigger_losses) / len(triggers) > 0.5:
        print(f"⚠️ 建议1: 10倍触发时命中率仅{trigger_hit_rate*100:.1f}%，考虑降低上限至8倍或9倍")
    else:
        print(f"✅ 建议1: 10倍触发时命中率{trigger_hit_rate*100:.1f}%表现尚可，保持当前上限")
    
    # 建议2：检查是否应该在特定条件下提前止损
    max_consecutive = max(fib_indices) if fib_indices else 0
    if max_consecutive >= 8:
        print(f"⚠️ 建议2: 出现连败{max_consecutive}期触发10倍，考虑在7期连败后强制重置")
    else:
        print(f"✅ 建议2: 最长连败{max_consecutive}期可控，无需额外止损机制")
    
    # 建议3：检查动态调整是否合理
    if boost_triggers > 0:
        print(f"⚠️ 建议3: 有{boost_triggers}次在增强倍投状态下触发10倍，说明热度判断可能有误")
        print(f"   → 考虑提高增强阈值（从35%提高至40%）或降低增强倍数（从1.5降至1.3）")
    else:
        print(f"✅ 建议3: 未在增强倍投状态触发10倍，动态调整逻辑合理")
    
    # 建议4：风险控制
    print(f"\n💡 建议4: 风险控制优化")
    print(f"   • 当前10倍损失占总回撤比例: {total_loss_at_10x}/{results[-1]['balance']:.0f}元")
    print(f"   • 可考虑在连败7期后，强制降低倍数至0.5x（保守模式）")
    print(f"   • 或设置单日最大投注额上限（如500元），超过则暂停投注")
    
    print()
    
    # 统计总体表现
    total_hits = sum(1 for r in results if r['hit'])
    total_hit_rate = total_hits / len(results)
    final_balance = results[-1]['balance']
    
    # 找最大回撤
    min_balance = 0
    for r in results:
        if r['balance'] < min_balance:
            min_balance = r['balance']
    max_drawdown = abs(min_balance)
    
    roi = (final_balance / total_bet * 100) if total_bet > 0 else 0
    
    print("=" * 100)
    print("【整体策略表现】")
    print("=" * 100)
    print()
    
    print(f"总期数: {len(results)}期")
    print(f"命中率: {total_hit_rate*100:.2f}%")
    print(f"ROI: {roi:.2f}%")
    print(f"净收益: {final_balance:+.0f}元")
    print(f"最大回撤: {max_drawdown:.0f}元")
    print(f"10倍触发: {len(triggers)}次 ({len(triggers)/len(results)*100:.1f}%)")
    print()


if __name__ == '__main__':
    analyze_10x_triggers()
