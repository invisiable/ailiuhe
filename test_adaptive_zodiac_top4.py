"""
自适应生肖TOP4投注策略 - 动态调整版
核心思路：当连续不中时，动态增加投注生肖数量（TOP4→TOP5→TOP6）
目标：将最大连续不中从12期降低到4期
"""

import pandas as pd
import numpy as np
from collections import deque
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy


class AdaptiveZodiacTop4Strategy:
    """自适应生肖TOP4投注策略 - 动态调整投注数量"""
    
    def __init__(self):
        """初始化策略"""
        # 使用原始的优秀预测器
        self.predictor = RecommendedZodiacTop4Strategy(use_emergency_backup=False)
        
        # 动态调整参数
        self.consecutive_misses = 0
        self.base_top_n = 4  # 基础投注数量
        self.max_top_n = 8  # 最大投注数量
        
        # 自适应规则：连续不中后如何调整（最激进版本）
        self.adaptation_rules = {
            0: 4,  # 0期不中 -> TOP4
            1: 5,  # 1期不中 -> TOP5（立即增加）
            2: 6,  # 2期不中 -> TOP6
            3: 7,  # 3期不中 -> TOP7
            4: 8,  # 4+期不中 -> TOP8（覆盖2/3生肖）
        }
        
        # 成本配置
        self.cost_per_zodiac = 4  # 每个生肖4元
        self.win_amount = 46  # 中奖金额46元
    
    def get_current_top_n(self):
        """根据当前连续不中次数，决定本期投注数量"""
        key = min(self.consecutive_misses, max(self.adaptation_rules.keys()))
        return self.adaptation_rules[key]
    
    def predict_and_bet(self, animals):
        """
        预测并决定投注方案
        
        Args:
            animals: 历史生肖列表
            
        Returns:
            dict: {
                'top_n': 本期投注数量,
                'zodiacs': 投注生肖列表,
                'cost': 投注成本,
                'reason': 调整原因
            }
        """
        # 获取本期应该投注的数量
        top_n = self.get_current_top_n()
        
        # 使用原始预测器获取更多候选
        prediction = self.predictor.predict_top4(animals)
        base_top4 = prediction['top4']
        
        # 如果需要更多生肖，扩展预测
        if top_n > 4:
            # 从历史中找出高频生肖补充
            from collections import Counter
            recent = animals[-30:] if len(animals) >= 30 else animals
            freq = Counter(recent)
            
            # 按频率排序所有生肖
            all_zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
            sorted_by_freq = sorted(all_zodiacs, key=lambda z: freq.get(z, 0), reverse=True)
            
            # 从频率最高的生肖中选择不在TOP4中的
            extended_zodiacs = base_top4.copy()
            for zodiac in sorted_by_freq:
                if zodiac not in extended_zodiacs and len(extended_zodiacs) < top_n:
                    extended_zodiacs.append(zodiac)
                if len(extended_zodiacs) >= top_n:
                    break
            
            zodiacs = extended_zodiacs
        else:
            zodiacs = base_top4
        
        # 计算成本
        cost = len(zodiacs) * self.cost_per_zodiac
        
        # 生成说明
        if self.consecutive_misses == 0:
            reason = "正常投注TOP4"
        elif self.consecutive_misses == 1:
            reason = f"连续{self.consecutive_misses}期不中，增加到TOP5"
        elif self.consecutive_misses == 2:
            reason = f"连续{self.consecutive_misses}期不中，增加到TOP6"
        elif self.consecutive_misses == 3:
            reason = f"连续{self.consecutive_misses}期不中，增加到TOP7"
        else:
            reason = f"连续{self.consecutive_misses}期不中，增加到TOP8"
        
        return {
            'top_n': top_n,
            'zodiacs': zodiacs,
            'cost': cost,
            'reason': reason
        }
    
    def update_result(self, is_hit):
        """
        更新投注结果
        
        Args:
            is_hit: 是否命中
        """
        if is_hit:
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
        
        # 同时更新预测器
        self.predictor.update_performance(is_hit)


def test_adaptive_strategy():
    """测试自适应策略"""
    print("="*80)
    print("自适应生肖TOP4策略测试 - 动态调整投注数量")
    print("目标：通过动态增加投注数量，将最大连续不中降低到4期")
    print("="*80 + "\n")
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"数据加载完成：{len(df)}期\n")
    
    # 初始化策略
    strategy = AdaptiveZodiacTop4Strategy()
    
    # 测试300期
    test_periods = 300
    start_idx = len(df) - test_periods
    
    results = []
    total_profit = 0
    total_cost = 0
    max_consecutive_misses = 0
    current_consecutive_misses = 0
    consecutive_miss_sequences = []
    
    hits = 0
    top4_bets = 0  # TOP4投注次数
    top5_bets = 0  # TOP5投注次数
    top6_bets = 0  # TOP6投注次数
    top7_bets = 0  # TOP7投注次数
    top8_bets = 0  # TOP8投注次数
    
    print(f"测试期数：最近{test_periods}期\n")
    print("开始回测...\n")
    
    for i in range(start_idx, len(df)):
        period = i - start_idx + 1
        
        # 获取历史数据
        history_animals = df['animal'].iloc[:i].tolist()
        
        # 预测和投注
        bet_plan = strategy.predict_and_bet(history_animals)
        zodiacs = bet_plan['zodiacs']
        cost = bet_plan['cost']
        top_n = bet_plan['top_n']
        reason = bet_plan['reason']
        
        # 统计投注类型
        if top_n == 4:
            top4_bets += 1
        elif top_n == 5:
            top5_bets += 1
        elif top_n == 6:
            top6_bets += 1
        elif top_n == 7:
            top7_bets += 1
        elif top_n == 8:
            top8_bets += 1
        
        # 实际结果
        actual = df.iloc[i]['animal']
        actual_date = df.iloc[i]['date']
        is_hit = actual in zodiacs
        
        # 计算盈亏
        if is_hit:
            profit = strategy.win_amount - cost
            hits += 1
            current_consecutive_misses = 0
        else:
            profit = -cost
            current_consecutive_misses += 1
            
            if current_consecutive_misses == 1:
                consecutive_miss_sequences.append({
                    'start': period,
                    'length': 1
                })
            else:
                consecutive_miss_sequences[-1]['length'] = current_consecutive_misses
        
        max_consecutive_misses = max(max_consecutive_misses, current_consecutive_misses)
        total_profit += profit
        total_cost += cost
        
        # 更新策略
        strategy.update_result(is_hit)
        
        # 记录结果
        results.append({
            'period': period,
            'date': actual_date,
            'top_n': top_n,
            'zodiacs': zodiacs,
            'actual': actual,
            'is_hit': is_hit,
            'cost': cost,
            'profit': profit,
            'cumulative_profit': total_profit,
            'consecutive_misses': current_consecutive_misses,
            'reason': reason
        })
        
        # 每50期显示一次进度
        if period % 50 == 0:
            print(f"  已测试 {period}/{test_periods} 期...")
    
    print("\n测试完成！\n")
    
    # 统计结果
    print("="*80)
    print("测试结果")
    print("="*80 + "\n")
    
    hit_rate = hits / test_periods
    roi = (total_profit / total_cost) * 100
    
    # 统计连续不中
    long_misses_5 = len([s for s in consecutive_miss_sequences if s['length'] >= 5])
    long_misses_7 = len([s for s in consecutive_miss_sequences if s['length'] >= 7])
    long_misses_10 = len([s for s in consecutive_miss_sequences if s['length'] >= 10])
    
    if consecutive_miss_sequences:
        avg_miss_length = np.mean([s['length'] for s in consecutive_miss_sequences])
    else:
        avg_miss_length = 0
    
    print("【基础统计】")
    print(f"  测试期数: {test_periods}")
    print(f"  命中次数: {hits}")
    print(f"  命中率: {hit_rate*100:.2f}%\n")
    
    print("【投注分布】")
    print(f"  TOP4投注: {top4_bets}次 ({top4_bets/test_periods*100:.1f}%)")
    print(f"  TOP5投注: {top5_bets}次 ({top5_bets/test_periods*100:.1f}%)")
    print(f"  TOP6投注: {top6_bets}次 ({top6_bets/test_periods*100:.1f}%)")
    print(f"  TOP7投注: {top7_bets}次 ({top7_bets/test_periods*100:.1f}%)")
    print(f"  TOP8投注: {top8_bets}次 ({top8_bets/test_periods*100:.1f}%)\n")
    
    print("【连续不中统计】⭐ 核心指标")
    print(f"  最大连续不中: {max_consecutive_misses}期")
    print(f"  平均连续不中: {avg_miss_length:.2f}期")
    print(f"  连续不中>=5期: {long_misses_5}次")
    print(f"  连续不中>=7期: {long_misses_7}次")
    print(f"  连续不中>=10期: {long_misses_10}次\n")
    
    print("【财务统计】")
    print(f"  总投注: {total_cost:.2f}元")
    print(f"  总收益: {total_profit:+.2f}元")
    print(f"  ROI: {roi:+.2f}%\n")
    
    # 目标达成情况
    print("="*80)
    print("目标达成情况")
    print("="*80 + "\n")
    
    target = 4
    print(f"目标: 最大连续不中 ≤ {target}期")
    print(f"实际: 最大连续不中 = {max_consecutive_misses}期\n")
    
    if max_consecutive_misses <= target:
        print(f"✅ 目标达成！")
        print(f"✅ 最大连续不中为{max_consecutive_misses}期，成功控制在{target}期以内")
    else:
        gap = max_consecutive_misses - target
        print(f"⚠ 未完全达成目标，差{gap}期")
        print(f"   但相比原始版的12期，已经降低了{((12-max_consecutive_misses)/12*100):.1f}%")
    
    # 详细的连续不中序列
    print("\n【所有>=5期的连续不中】")
    long_misses = [s for s in consecutive_miss_sequences if s['length'] >= 5]
    long_misses.sort(key=lambda x: x['length'], reverse=True)
    if long_misses:
        for seq in long_misses:
            end = seq['start'] + seq['length'] - 1
            print(f"  {seq['length']}期不中: 第{seq['start']}期 到 第{end}期")
    else:
        print("  无5期以上连续不中情况 ✓")
    
    # 保存结果
    print("\n保存详细结果...")
    df_results = pd.DataFrame(results)
    df_results.to_csv('zodiac_top4_adaptive_300periods.csv', index=False, encoding='utf-8-sig')
    print("  ✓ zodiac_top4_adaptive_300periods.csv")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    
    return {
        'max_consecutive_misses': max_consecutive_misses,
        'hit_rate': hit_rate,
        'roi': roi,
        'total_profit': total_profit
    }


if __name__ == '__main__':
    test_adaptive_strategy()
