"""
动态扩展号码策略 - 最直接有效降低连续不中的方案
核心思路：正常时买15个号，连败时扩展到18-23个号
数学原理：15/49=30.6% → 18/49=36.7% → 20/49=40.8% → 23/49=46.9%
连败概率：(1-0.36)^5=10.7% → (1-0.41)^5=7.1% → (1-0.47)^5=4.2%
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from top15_predictor import Top15Predictor
from precise_top15_predictor import PreciseTop15Predictor
from smart_top15_predictor import SmartTop15Predictor


class DynamicExpansionPredictor:
    """
    动态扩展预测器
    - 连败0期：15个号码（标准）
    - 连败1期：15个号码
    - 连败2期：18个号码 (+3)
    - 连败3期：20个号码 (+5)
    - 连败4期+：23个号码 (+8)
    """
    
    def __init__(self, expand_config=None):
        """
        expand_config: dict映射 连败次数→号码数量
        """
        self.precise_predictor = PreciseTop15Predictor()
        self.smart_predictor = SmartTop15Predictor()
        self.base_predictor = Top15Predictor()
        
        # 默认扩展配置
        self.expand_config = expand_config or {
            0: 15, 1: 15, 2: 18, 3: 20, 4: 23, 5: 23
        }
        self.max_expand = max(self.expand_config.values())
        
        self.consecutive_misses = 0
        self.recent_predictions = []
        self.recent_actuals = []
    
    def update(self, prediction, actual):
        hit = actual in prediction
        if hit:
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
        
        self.recent_predictions.append(prediction)
        self.recent_actuals.append(actual)
        if len(self.recent_predictions) > 20:
            self.recent_predictions.pop(0)
            self.recent_actuals.pop(0)
    
    def get_current_count(self):
        """根据连败次数获取当前应选号码数量"""
        for miss_count in sorted(self.expand_config.keys(), reverse=True):
            if self.consecutive_misses >= miss_count:
                return self.expand_config[miss_count]
        return 15
    
    def predict(self, numbers):
        """动态扩展预测"""
        count = self.get_current_count()
        
        # 使用精准预测器获取扩展数量的号码
        if count <= 15:
            return self.precise_predictor.predict(numbers, k=count)
        
        # 需要扩展：融合多模型
        precise_pred = self.precise_predictor.predict(numbers, k=count + 5)
        smart_pred = self.smart_predictor.predict(numbers)
        base_pred = self.base_predictor.predict(numbers)
        
        # 投票融合
        votes = defaultdict(float)
        for rank, n in enumerate(precise_pred):
            votes[n] += (len(precise_pred) - rank) / len(precise_pred) * 1.5
        for rank, n in enumerate(smart_pred):
            votes[n] += (15 - rank) / 15.0
        for rank, n in enumerate(base_pred):
            votes[n] += (15 - rank) / 15.0 * 0.8
        
        # 加入盲区号码
        if self.recent_actuals:
            for pred, actual in zip(self.recent_predictions[-5:], self.recent_actuals[-5:]):
                if actual not in pred:
                    # miss号码的邻域加分
                    for offset in [-2, -1, 1, 2]:
                        n = actual + offset
                        if 1 <= n <= 49:
                            votes[n] += 0.5
        
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_votes[:count]]


def run_dynamic_expansion_test():
    """测试多种扩展配置"""
    print("=" * 80)
    print("动态扩展号码策略对比 - 400期回测")
    print("=" * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    total = len(df)
    test_periods = 400
    start_idx = total - test_periods
    
    # 不同扩展配置
    configs = {
        'A.基准(固定15)': {0: 15, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15},
        'B.温和扩展(15→18)': {0: 15, 1: 15, 2: 17, 3: 18, 4: 18, 5: 18},
        'C.标准扩展(15→20)': {0: 15, 1: 15, 2: 18, 3: 20, 4: 20, 5: 20},
        'D.激进扩展(15→23)': {0: 15, 1: 15, 2: 18, 3: 20, 4: 23, 5: 23},
        'E.超激进(15→25)': {0: 15, 1: 17, 2: 20, 3: 23, 4: 25, 5: 25},
        'F.渐进扩展(15→21)': {0: 15, 1: 16, 2: 17, 3: 18, 4: 19, 5: 21},
    }
    
    all_results = {}
    
    for name, config in configs.items():
        predictor = DynamicExpansionPredictor(expand_config=config)
        
        hits = []
        expand_counts = []  # 记录每期实际买了多少个号码
        
        for i in range(start_idx, total):
            train_data = numbers[:i]
            actual = int(numbers[i])
            
            current_count = predictor.get_current_count()
            top_n = predictor.predict(train_data)
            is_hit = actual in top_n
            
            predictor.update(top_n, actual)
            hits.append(is_hit)
            expand_counts.append(current_count)
        
        # 统计
        streaks = []
        streak_len = 0
        for h in hits:
            if not h:
                streak_len += 1
            else:
                if streak_len > 0:
                    streaks.append(streak_len)
                streak_len = 0
        if streak_len > 0:
            streaks.append(streak_len)
        
        hit_rate = sum(hits) / len(hits) * 100
        max_streak = max(streaks) if streaks else 0
        avg_streak = sum(streaks) / len(streaks) if streaks else 0
        avg_count = sum(expand_counts) / len(expand_counts)
        
        # 成本估算（每个号码1注=15元基础，号码多则成本线性增加）
        total_cost = sum(c for c in expand_counts)  # 每个号码1元计
        
        all_results[name] = {
            'hit_rate': hit_rate,
            'max_streak': max_streak,
            'avg_streak': avg_streak,
            'streak_ge3': sum(1 for s in streaks if s >= 3),
            'streak_ge5': sum(1 for s in streaks if s >= 5),
            'streak_ge7': sum(1 for s in streaks if s >= 7),
            'avg_count': avg_count,
            'total_cost': total_cost,
            'streaks': streaks,
            'expand_counts': expand_counts,
        }
        
        print(f"\n{name}:")
        print(f"  命中率={hit_rate:.1f}%, 最长连败={max_streak}, ≥5期={sum(1 for s in streaks if s >= 5)}次, "
              f"平均号码数={avg_count:.1f}")
    
    # 对比表
    print(f"\n\n{'=' * 110}")
    print(f"{'策略对比总结':^110}")
    print(f"{'=' * 110}")
    print(f"{'策略':<24} {'命中率':>7} {'最长连败':>8} {'平均连败':>8} "
          f"{'≥3期':>6} {'≥5期':>6} {'≥7期':>6} {'平均号码数':>10} {'总成本(号码)':>12}")
    print("-" * 110)
    
    for name, r in all_results.items():
        print(f"{name:<24} {r['hit_rate']:>6.1f}% {r['max_streak']:>7} "
              f"{r['avg_streak']:>7.2f} {r['streak_ge3']:>6} "
              f"{r['streak_ge5']:>6} {r['streak_ge7']:>6} "
              f"{r['avg_count']:>9.1f} {r['total_cost']:>11}")
    
    # 效率分析：命中率/每号码成本
    print(f"\n{'=' * 80}")
    print(f"效率分析（命中率 vs 成本）")
    print(f"{'=' * 80}")
    print(f"{'策略':<24} {'命中率':>7} {'平均号码数':>10} {'每号码命中效率':>14} {'防连败效果':>10}")
    print("-" * 70)
    
    for name, r in all_results.items():
        efficiency = r['hit_rate'] / r['avg_count']
        anti_streak = (1 - r['streak_ge5'] / 20) * 100
        print(f"{name:<24} {r['hit_rate']:>6.1f}% {r['avg_count']:>9.1f} "
              f"{efficiency:>13.2f}%/号 {anti_streak:>9.0f}分")
    
    # 连败分布对比（关键指标）
    print(f"\n{'=' * 80}")
    print(f"连续不中分布对比")
    print(f"{'=' * 80}")
    print(f"{'长度':<6}", end='')
    for name in all_results:
        short_name = name.split('.')[1][:6]
        print(f" {short_name:>8}", end='')
    print()
    print("-" * 60)
    
    all_lengths = set()
    for r in all_results.values():
        all_lengths.update(r['streaks'])
    
    for length in sorted(all_lengths):
        if length > 15:
            continue
        print(f"{length:>3}期  ", end='')
        for name, r in all_results.items():
            count = Counter(r['streaks']).get(length, 0)
            print(f" {count:>8}", end='')
        print()
    
    # 投注成本模拟（配合最优智能投注策略）
    print(f"\n{'=' * 80}")
    print(f"与最优智能投注策略配合的成本分析")
    print(f"{'=' * 80}")
    print(f"假设：基础每号1元，命中奖金47元/注")
    print(f"{'策略':<24} {'平均每期成本':>12} {'命中收益':>10} {'预期ROI':>10}")
    print("-" * 60)
    
    for name, r in all_results.items():
        avg_cost = r['avg_count']  # 平均每期成本(号码数)
        hit_prob = r['hit_rate'] / 100
        expected_reward = 47 * hit_prob
        expected_profit = expected_reward - avg_cost
        roi = expected_profit / avg_cost * 100
        print(f"{name:<24} {avg_cost:>11.1f}元 {expected_reward:>9.1f}元 {roi:>+9.1f}%")
    
    # 最佳推荐
    print(f"\n{'=' * 80}")
    print(f"推荐方案")
    print(f"{'=' * 80}")
    
    # 找性价比最好的（命中率提升/成本增加比最优且连败改善明显的）
    baseline = all_results['A.基准(固定15)']
    print(f"\n基准: 命中率{baseline['hit_rate']:.1f}%, 最长连败{baseline['max_streak']}, ≥5期{baseline['streak_ge5']}次\n")
    
    for name, r in all_results.items():
        if name == 'A.基准(固定15)':
            continue
        hit_improve = r['hit_rate'] - baseline['hit_rate']
        cost_increase = (r['avg_count'] - baseline['avg_count']) / baseline['avg_count'] * 100
        streak_improve = baseline['max_streak'] - r['max_streak']
        ge5_improve = baseline['streak_ge5'] - r['streak_ge5']
        
        print(f"{name}:")
        print(f"  命中率提升: {hit_improve:+.1f}%  成本增加: {cost_increase:+.1f}%  "
              f"最长连败改善: {streak_improve:+d}期  ≥5期连败减少: {ge5_improve:+d}次")


if __name__ == '__main__':
    run_dynamic_expansion_test()
