"""
验证精准TOP15投注与生肖TOP4投注的互补性
分析同时购买两种方案能否提高整体命中率和收益
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor
from ensemble_select_best_predictor import EnsembleSelectBestPredictor

# 生肖到号码的映射
ZODIAC_TO_NUMBERS = {
    '鼠': [1, 13, 25, 37, 49],
    '牛': [2, 14, 26, 38],
    '虎': [3, 15, 27, 39],
    '兔': [4, 16, 28, 40],
    '龙': [5, 17, 29, 41],
    '蛇': [6, 18, 30, 42],
    '马': [7, 19, 31, 43],
    '羊': [8, 20, 32, 44],
    '猴': [9, 21, 33, 45],
    '鸡': [10, 22, 34, 46],
    '狗': [11, 23, 35, 47],
    '猪': [12, 24, 36, 48]
}

def zodiac_to_numbers(zodiacs):
    """将生肖列表转换为号码列表"""
    numbers = []
    for zodiac in zodiacs:
        numbers.extend(ZODIAC_TO_NUMBERS.get(zodiac, []))
    return list(set(numbers))  # 去重

def load_data():
    """加载数据"""
    df = pd.read_csv('data/lucky_numbers.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def analyze_complementary(test_periods=200):
    """分析互补性"""
    print("="*80)
    print("精准TOP15 vs 生肖TOP4 互补性分析")
    print("="*80)
    
    df = load_data()
    total_periods = len(df)
    start_idx = total_periods - test_periods
    
    # 初始化预测器
    top15_predictor = PreciseTop15Predictor()
    top4_predictor = EnsembleSelectBestPredictor()
    
    results = []
    
    # Fibonacci倍投序列
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    # 统计变量
    top15_only_hits = 0  # 仅TOP15命中
    top4_only_hits = 0   # 仅TOP4命中
    both_hits = 0        # 两者都命中
    both_miss = 0        # 两者都不中
    
    # 投注统计（独立倍投）
    top15_multiplier_idx = 0
    top4_multiplier_idx = 0
    top15_total_bet = 0
    top15_total_reward = 0
    top4_total_bet = 0
    top4_total_reward = 0
    
    # 组合策略统计
    combo_total_bet = 0
    combo_total_reward = 0
    combo_multiplier_idx = 0  # 组合策略：只要有一个中就重置
    
    print(f"\n测试周期: 最近{test_periods}期")
    print(f"开始分析...\n")
    
    for i in range(start_idx, total_periods):
        train_data = df.iloc[:i]
        test_row = df.iloc[i]
        
        # TOP15预测
        history_numbers = train_data['number'].tolist()
        top15_predictions = top15_predictor.predict(history_numbers)
        
        # TOP4预测
        history_animals = train_data['animal'].tolist()
        top4_result = top4_predictor.predict_top4(history_animals)
        top4_zodiacs = top4_result['top4']
        top4_numbers = zodiac_to_numbers(top4_zodiacs)  # 转换为号码
        
        # 实际结果
        actual_number = test_row['number']
        actual_zodiac = test_row['animal']
        
        # 判断命中情况
        top15_hit = actual_number in top15_predictions
        top4_hit = actual_zodiac in top4_zodiacs
        
        # 统计互补性
        if top15_hit and top4_hit:
            both_hits += 1
            hit_type = "双中"
        elif top15_hit and not top4_hit:
            top15_only_hits += 1
            hit_type = "仅TOP15"
        elif not top15_hit and top4_hit:
            top4_only_hits += 1
            hit_type = "仅TOP4"
        else:
            both_miss += 1
            hit_type = "都不中"
        
        # 计算独立投注（各自倍投）
        top15_mult = fib_sequence[min(top15_multiplier_idx, len(fib_sequence)-1)]
        top4_mult = fib_sequence[min(top4_multiplier_idx, len(fib_sequence)-1)]
        
        top15_bet = 15 * top15_mult
        top4_bet = 16 * top4_mult
        
        top15_reward = 47 * top15_mult if top15_hit else 0
        top4_reward = 47 * top4_mult if top4_hit else 0
        
        top15_total_bet += top15_bet
        top15_total_reward += top15_reward
        top4_total_bet += top4_bet
        top4_total_reward += top4_reward
        
        # 更新独立倍投索引
        if top15_hit:
            top15_multiplier_idx = 0
        else:
            top15_multiplier_idx += 1
            
        if top4_hit:
            top4_multiplier_idx = 0
        else:
            top4_multiplier_idx += 1
        
        # 计算组合策略（只要有一个中就重置倍投）
        combo_mult = fib_sequence[min(combo_multiplier_idx, len(fib_sequence)-1)]
        combo_bet = (15 + 16) * combo_mult  # 两边都投注
        combo_reward = 0
        
        if top15_hit:
            combo_reward += 47 * combo_mult
        if top4_hit:
            combo_reward += 47 * combo_mult
        
        combo_total_bet += combo_bet
        combo_total_reward += combo_reward
        
        # 更新组合倍投索引（只要有一个中就重置）
        if top15_hit or top4_hit:
            combo_multiplier_idx = 0
        else:
            combo_multiplier_idx += 1
        
        # 更新预测器性能
        top15_predictor.update_performance(top15_predictions, actual_number)
        prediction_details = top4_result.get('details', {})
        top4_predictor.update_performance(actual_zodiac, prediction_details)
        
        results.append({
            'date': test_row['date'],
            'period': i - start_idx + 1,
            'actual_number': actual_number,
            'actual_zodiac': actual_zodiac,
            'top15_predictions': top15_predictions,
            'top4_zodiacs': top4_zodiacs,
            'top4_numbers': top4_numbers,
            'top15_hit': top15_hit,
            'top4_hit': top4_hit,
            'hit_type': hit_type,
            'top15_bet': top15_bet,
            'top4_bet': top4_bet,
            'combo_bet': combo_bet,
            'top15_reward': top15_reward,
            'top4_reward': top4_reward,
            'combo_reward': combo_reward
        })
    
    # 输出详细分析
    print("="*80)
    print("互补性统计")
    print("="*80)
    
    total = len(results)
    print(f"\n总测试期数: {total}期")
    print(f"测试周期: {results[0]['date'].strftime('%Y/%m/%d')} ~ {results[-1]['date'].strftime('%Y/%m/%d')}")
    print()
    
    print("命中分布:")
    print(f"  🎯 双方都命中: {both_hits}次 ({both_hits/total*100:.2f}%)")
    print(f"  📊 仅TOP15命中: {top15_only_hits}次 ({top15_only_hits/total*100:.2f}%)")
    print(f"  🐉 仅TOP4命中: {top4_only_hits}次 ({top4_only_hits/total*100:.2f}%)")
    print(f"  ❌ 双方都不中: {both_miss}次 ({both_miss/total*100:.2f}%)")
    print()
    
    # 计算各种命中率
    top15_total_hits = both_hits + top15_only_hits
    top4_total_hits = both_hits + top4_only_hits
    combo_total_hits = both_hits + top15_only_hits + top4_only_hits  # 至少一个命中
    
    top15_hit_rate = top15_total_hits / total * 100
    top4_hit_rate = top4_total_hits / total * 100
    combo_hit_rate = combo_total_hits / total * 100
    
    print("="*80)
    print("方案对比分析")
    print("="*80)
    
    # 方案1：仅TOP15
    top15_profit = top15_total_reward - top15_total_bet
    top15_roi = (top15_profit / top15_total_bet * 100) if top15_total_bet > 0 else 0
    
    print(f"\n【方案1】仅购买精准TOP15")
    print(f"  命中次数: {top15_total_hits}/{total}")
    print(f"  命中率: {top15_hit_rate:.2f}%")
    print(f"  总投入: {top15_total_bet:,}元")
    print(f"  总奖励: {top15_total_reward:,}元")
    print(f"  净收益: {top15_profit:+,}元")
    print(f"  ROI: {top15_roi:+.2f}%")
    
    # 方案2：仅TOP4
    top4_profit = top4_total_reward - top4_total_bet
    top4_roi = (top4_profit / top4_total_bet * 100) if top4_total_bet > 0 else 0
    
    print(f"\n【方案2】仅购买生肖TOP4")
    print(f"  命中次数: {top4_total_hits}/{total}")
    print(f"  命中率: {top4_hit_rate:.2f}%")
    print(f"  总投入: {top4_total_bet:,}元")
    print(f"  总奖励: {top4_total_reward:,}元")
    print(f"  净收益: {top4_profit:+,}元")
    print(f"  ROI: {top4_roi:+.2f}%")
    
    # 方案3：独立购买两者（各自倍投）
    independent_total_bet = top15_total_bet + top4_total_bet
    independent_total_reward = top15_total_reward + top4_total_reward
    independent_profit = independent_total_reward - independent_total_bet
    independent_roi = (independent_profit / independent_total_bet * 100) if independent_total_bet > 0 else 0
    
    print(f"\n【方案3】独立购买两者（各自倍投，互不影响）")
    print(f"  至少一方命中: {combo_total_hits}/{total}")
    print(f"  综合命中率: {combo_hit_rate:.2f}%")
    print(f"  总投入: {independent_total_bet:,}元 (TOP15: {top15_total_bet:,} + TOP4: {top4_total_bet:,})")
    print(f"  总奖励: {independent_total_reward:,}元")
    print(f"  净收益: {independent_profit:+,}元")
    print(f"  ROI: {independent_roi:+.2f}%")
    
    # 方案4：组合购买（共享倍投，一方中即重置）
    combo_profit = combo_total_reward - combo_total_bet
    combo_roi = (combo_profit / combo_total_bet * 100) if combo_total_bet > 0 else 0
    
    print(f"\n【方案4】组合购买（共享倍投，一方中即重置）")
    print(f"  至少一方命中: {combo_total_hits}/{total}")
    print(f"  综合命中率: {combo_hit_rate:.2f}%")
    print(f"  总投入: {combo_total_bet:,}元")
    print(f"  总奖励: {combo_total_reward:,}元")
    print(f"  净收益: {combo_profit:+,}元")
    print(f"  ROI: {combo_roi:+.2f}%")
    
    # 互补性分析
    print("\n" + "="*80)
    print("互补性深度分析")
    print("="*80)
    
    # 计算互补增益
    single_best_hits = max(top15_total_hits, top4_total_hits)
    combo_additional_hits = combo_total_hits - single_best_hits
    hit_rate_increase = combo_hit_rate - max(top15_hit_rate, top4_hit_rate)
    
    print(f"\n✅ 互补性发现:")
    print(f"  单独最佳命中: {single_best_hits}次")
    print(f"  组合总命中: {combo_total_hits}次")
    print(f"  互补新增命中: {combo_additional_hits}次")
    print(f"  命中率提升: {hit_rate_increase:+.2f}个百分点")
    
    # 收益对比
    print(f"\n💰 收益对比:")
    single_best_profit = max(top15_profit, top4_profit)
    single_best_name = "精准TOP15" if top15_profit > top4_profit else "生肖TOP4"
    
    print(f"  单独最佳收益: {single_best_profit:+,}元 ({single_best_name})")
    print(f"  独立组合收益: {independent_profit:+,}元 (差异: {independent_profit - single_best_profit:+,}元)")
    print(f"  共享倍投收益: {combo_profit:+,}元 (差异: {combo_profit - single_best_profit:+,}元)")
    
    # 投资回报率对比
    single_best_roi = max(top15_roi, top4_roi)
    print(f"\n📈 ROI对比:")
    print(f"  单独最佳ROI: {single_best_roi:+.2f}% ({single_best_name})")
    print(f"  独立组合ROI: {independent_roi:+.2f}% (差异: {independent_roi - single_best_roi:+.2f}%)")
    print(f"  共享倍投ROI: {combo_roi:+.2f}% (差异: {combo_roi - single_best_roi:+.2f}%)")
    
    # 最终建议
    print("\n" + "="*80)
    print("💡 策略建议")
    print("="*80)
    
    # 判断是否值得组合
    if combo_additional_hits > 0:
        print(f"\n✅ 两种策略存在互补性!")
        print(f"   互补优势: 额外增加{combo_additional_hits}次命中机会")
        print(f"   命中率从{max(top15_hit_rate, top4_hit_rate):.2f}%提升到{combo_hit_rate:.2f}%")
    else:
        print(f"\n⚠️ 两种策略互补性较弱")
        print(f"   组合后命中率未超过单独最优方案")
    
    # ROI角度分析
    print(f"\n从投资回报率角度:")
    if combo_roi > single_best_roi:
        improvement = combo_roi - single_best_roi
        print(f"  ✅ 组合策略ROI更优")
        print(f"     提升幅度: {improvement:+.2f}个百分点")
        print(f"     推荐: 采用【方案4-组合购买】策略")
    elif independent_roi > single_best_roi:
        improvement = independent_roi - single_best_roi
        print(f"  ✅ 独立组合ROI更优")
        print(f"     提升幅度: {improvement:+.2f}个百分点")
        print(f"     推荐: 采用【方案3-独立购买】策略")
    else:
        print(f"  ⚠️ 单独投注更优")
        print(f"     推荐: 专注【{single_best_name}】单一策略")
        print(f"     原因: 组合投入增加但回报未成比例提升")
    
    # 风险提示
    print(f"\n⚠️ 风险提示:")
    combo_investment_ratio = combo_total_bet / max(top15_total_bet, top4_total_bet)
    print(f"  组合投入是单独投入的 {combo_investment_ratio:.2f} 倍")
    print(f"  需确保资金充足以支持双重倍投策略")
    
    return {
        'total_periods': total,
        'both_hits': both_hits,
        'top15_only_hits': top15_only_hits,
        'top4_only_hits': top4_only_hits,
        'both_miss': both_miss,
        'combo_hit_rate': combo_hit_rate,
        'top15_hit_rate': top15_hit_rate,
        'top4_hit_rate': top4_hit_rate,
        'combo_profit': combo_profit,
        'combo_roi': combo_roi,
        'independent_profit': independent_profit,
        'independent_roi': independent_roi
    }

def main():
    """主函数"""
    print("\n" + "="*80)
    print("精准TOP15 与 生肖TOP4 互补性验证系统")
    print("="*80)
    print("目标: 验证同时购买两种方案能否提高命中概率和收益")
    print("="*80 + "\n")
    
    analyze_complementary(test_periods=200)
    
    print("\n" + "="*80)
    print("验证完成!")
    print("="*80)

if __name__ == "__main__":
    main()
