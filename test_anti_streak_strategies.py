"""
降低TOP15连续不中的方案探索
测试多种策略组合，对比各方案对连续不中的改善效果
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from top15_predictor import Top15Predictor
from precise_top15_predictor import PreciseTop15Predictor
from smart_top15_predictor import SmartTop15Predictor


class AntiStreakTop15Predictor:
    """
    防连败TOP15预测器
    核心思路：通过多策略融合 + 动态号码多样性注入 来降低连续不中概率
    """
    
    def __init__(self, strategy='ensemble_diverse'):
        self.strategy = strategy
        self.base_predictor = Top15Predictor()
        self.precise_predictor = PreciseTop15Predictor()
        self.smart_predictor = SmartTop15Predictor()
        
        # 追踪
        self.consecutive_misses = 0
        self.recent_predictions = []
        self.recent_actuals = []
        self.miss_pattern = Counter()  # 追踪模型盲区
    
    def update(self, prediction, actual):
        """更新追踪状态"""
        hit = actual in prediction
        if hit:
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
            # 记录模型盲区
            self.miss_pattern[actual] += 1
        
        self.recent_predictions.append(prediction)
        self.recent_actuals.append(actual)
        if len(self.recent_predictions) > 30:
            self.recent_predictions.pop(0)
            self.recent_actuals.pop(0)
    
    def predict_strategy_ensemble_diverse(self, numbers):
        """
        策略1: 集成多样性策略
        - 3个模型各出15个号码，取并集中票数最高的15个
        - 当连续不中>=2时，强制注入"多样性号码"替换低票数号码
        """
        base_pred = self.base_predictor.predict(numbers)
        precise_pred = self.precise_predictor.predict(numbers)
        smart_pred = self.smart_predictor.predict(numbers)
        
        # 投票计分
        votes = defaultdict(float)
        for rank, n in enumerate(base_pred):
            votes[n] += (15 - rank) / 15.0
        for rank, n in enumerate(precise_pred):
            votes[n] += (15 - rank) / 15.0
        for rank, n in enumerate(smart_pred):
            votes[n] += (15 - rank) / 15.0
        
        # 多样性注入：连续不中时加入冷号和补盲区号
        if self.consecutive_misses >= 2:
            diversity_nums = self._get_diversity_numbers(numbers)
            for n in diversity_nums:
                votes[n] += 0.8  # 给予适度加分
        
        sorted_nums = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:15]]
    
    def predict_strategy_adaptive_expand(self, numbers):
        """
        策略2: 自适应扩展策略
        - 正常时选15个号码
        - 连续不中1期后注入2个多样性号码（替换最低分的2个）
        - 连续不中2期后注入4个多样性号码
        - 连续不中3期+后注入5个多样性号码
        """
        base_pred = self.precise_predictor.predict(numbers)
        
        # 根据连续不中次数决定多样性注入程度
        if self.consecutive_misses == 0:
            return base_pred[:15]
        elif self.consecutive_misses == 1:
            inject_count = 2
        elif self.consecutive_misses == 2:
            inject_count = 4
        else:
            inject_count = 5
        
        # 获取多样性号码
        diversity_nums = self._get_diversity_numbers(numbers, count=inject_count)
        
        # 替换最低优先级的号码
        result = base_pred[:15 - inject_count]
        for n in diversity_nums:
            if n not in result:
                result.append(n)
            if len(result) >= 15:
                break
        
        # 补齐到15个
        for n in base_pred:
            if n not in result:
                result.append(n)
            if len(result) >= 15:
                break
        
        return result[:15]
    
    def predict_strategy_model_rotation(self, numbers):
        """
        策略3: 模型轮换策略
        - 连续不中0-1期：用精准模型
        - 连续不中2-3期：切换到智能模型
        - 连续不中4期+：切换到基础模型+多样性注入
        """
        if self.consecutive_misses <= 1:
            return self.precise_predictor.predict(numbers)
        elif self.consecutive_misses <= 3:
            return self.smart_predictor.predict(numbers)
        else:
            base = self.base_predictor.predict(numbers)
            diversity = self._get_diversity_numbers(numbers, count=5)
            result = base[:10]
            for n in diversity:
                if n not in result:
                    result.append(n)
            for n in base[10:]:
                if n not in result:
                    result.append(n)
            return result[:15]
    
    def predict_strategy_union_top(self, numbers):
        """
        策略4: 并集取交策略
        - 3个模型各出20个，取出现>=2次的号码（高共识）
        - 剩余用多样性号码填充
        """
        base_20 = self.base_predictor.predict(numbers)  # 基础只有15，扩展
        precise_20 = self.precise_predictor.predict(numbers, k=20)
        smart_pred = self.smart_predictor.predict(numbers)
        
        # 对base扩展到20
        pattern = self.base_predictor.analyze_pattern(numbers)
        methods = [
            (self.base_predictor.method_frequency_advanced(pattern, 20), 0.25),
            (self.base_predictor.method_zone_dynamic(pattern, 20), 0.25),
            (self.base_predictor.method_cyclic_pattern(pattern, 20), 0.25),
            (self.base_predictor.method_gap_prediction(pattern, 20), 0.25)
        ]
        scores = {}
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                scores[num] = scores.get(num, 0) + score
        base_20 = [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]]
        
        # 计票
        appear_count = Counter()
        for n in base_20:
            appear_count[n] += 1
        for n in precise_20:
            appear_count[n] += 1
        for n in smart_pred:
            appear_count[n] += 1
        
        # 高共识号码（>=2票）
        consensus = [n for n, c in appear_count.most_common() if c >= 2]
        
        result = consensus[:15]
        
        # 不够15个则补充多样性号码
        if len(result) < 15:
            diversity = self._get_diversity_numbers(numbers, count=15 - len(result))
            for n in diversity:
                if n not in result:
                    result.append(n)
                if len(result) >= 15:
                    break
        
        # 还不够则补单票号码
        for n, c in appear_count.most_common():
            if n not in result:
                result.append(n)
            if len(result) >= 15:
                break
        
        return result[:15]
    
    def predict_strategy_anti_pattern(self, numbers):
        """
        策略5: 反模式策略
        - 分析模型的盲区（经常miss的号码类型）
        - 主动覆盖这些盲区
        """
        base_pred = self.precise_predictor.predict(numbers)
        
        if len(self.recent_actuals) < 5:
            return base_pred[:15]
        
        # 分析最近miss的号码特征
        recent_misses = []
        for pred, actual in zip(self.recent_predictions[-10:], self.recent_actuals[-10:]):
            if actual not in pred:
                recent_misses.append(actual)
        
        if not recent_misses:
            return base_pred[:15]
        
        # 分析miss号码的特征
        miss_ranges = Counter()
        for n in recent_misses:
            if n <= 10: miss_ranges['1-10'] += 1
            elif n <= 20: miss_ranges['11-20'] += 1
            elif n <= 30: miss_ranges['21-30'] += 1
            elif n <= 40: miss_ranges['31-40'] += 1
            else: miss_ranges['41-49'] += 1
        
        # 找到模型盲区范围
        blind_spots = [r for r, c in miss_ranges.most_common(2)]
        
        # 从盲区范围选择号码
        blind_candidates = []
        range_map = {'1-10': (1,10), '11-20': (11,20), '21-30': (21,30), 
                     '31-40': (31,40), '41-49': (41,49)}
        
        recent_5 = set(list(numbers)[-5:]) if len(numbers) >= 5 else set()
        
        for r in blind_spots:
            start, end = range_map[r]
            for n in range(start, end+1):
                if n not in recent_5 and n not in base_pred[:10]:
                    blind_candidates.append(n)
        
        # 用盲区号码替换低优先级的基础预测
        inject_count = min(4, len(blind_candidates))
        result = base_pred[:15 - inject_count]
        
        np.random.seed(int(numbers[-1]) + len(numbers))
        if blind_candidates:
            chosen = list(np.random.choice(blind_candidates, 
                                           size=min(inject_count, len(blind_candidates)), 
                                           replace=False))
            for n in chosen:
                if n not in result:
                    result.append(int(n))
        
        # 补齐
        for n in base_pred:
            if n not in result:
                result.append(n)
            if len(result) >= 15:
                break
        
        return result[:15]
    
    def predict_strategy_coverage_max(self, numbers):
        """
        策略6: 最大覆盖策略
        - 保证每个区间(1-10,11-20,21-30,31-40,41-49)至少2个号码
        - 保证奇偶比例平衡（7:8或8:7）
        - 剩余位置用模型打分最高的填充
        """
        # 基础评分
        pattern = self.base_predictor.analyze_pattern(numbers)
        methods = [
            (self.base_predictor.method_frequency_advanced(pattern, 49), 0.30),
            (self.base_predictor.method_zone_dynamic(pattern, 49), 0.25),
            (self.base_predictor.method_cyclic_pattern(pattern, 49), 0.20),
            (self.base_predictor.method_gap_prediction(pattern, 49), 0.25)
        ]
        scores = {}
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                scores[num] = scores.get(num, 0) + score
        
        recent_5 = set(list(numbers)[-5:])
        for n in recent_5:
            if n in scores:
                scores[n] *= 0.3
        
        # 按区间分组
        zones = [(1,10), (11,20), (21,30), (31,40), (41,49)]
        result = []
        
        # 每个区间至少选2个最高分的
        for start, end in zones:
            zone_scores = [(n, scores.get(n, 0)) for n in range(start, end+1)]
            zone_scores.sort(key=lambda x: x[1], reverse=True)
            for n, _ in zone_scores[:2]:
                result.append(n)
        
        # 剩余5个位置用全局最高分填充
        remaining = 15 - len(result)
        all_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for n, _ in all_sorted:
            if n not in result:
                result.append(n)
                remaining -= 1
            if remaining <= 0:
                break
        
        return result[:15]
    
    def _get_diversity_numbers(self, numbers, count=5):
        """获取多样性号码 - 与当前模型预测互补的号码"""
        numbers_list = list(numbers) if not isinstance(numbers, list) else numbers
        recent_30 = numbers_list[-30:]
        recent_5 = set(numbers_list[-5:])
        
        # 找到模型最近经常错过的号码范围
        miss_nums = []
        for pred, actual in zip(self.recent_predictions[-10:], self.recent_actuals[-10:]):
            if actual not in pred:
                miss_nums.append(actual)
        
        candidates = []
        
        # 1. 从miss的号码的邻近号码中选
        for n in miss_nums[-5:]:
            for offset in [-2, -1, 1, 2]:
                neighbor = n + offset
                if 1 <= neighbor <= 49 and neighbor not in recent_5:
                    candidates.append(neighbor)
        
        # 2. 长期未出现但非超冷号码
        freq_30 = Counter(recent_30)
        for n in range(1, 50):
            if freq_30.get(n, 0) == 0 and n not in recent_5:
                candidates.append(n)
        
        # 3. 确保奇偶多样性
        # 去重
        candidates = list(set(candidates))
        
        if len(candidates) < count:
            # 补充随机号
            all_nums = [n for n in range(1, 50) if n not in recent_5]
            np.random.seed(int(numbers_list[-1]) + len(numbers_list))
            extra = list(np.random.choice(all_nums, size=min(10, len(all_nums)), replace=False))
            candidates.extend(extra)
        
        # 按一定规则选取
        candidates = list(set(candidates))
        np.random.seed(int(numbers_list[-1]) * 7 + len(numbers_list))
        if len(candidates) > count:
            chosen = list(np.random.choice(candidates, size=count, replace=False))
            return [int(x) for x in chosen]
        return [int(x) for x in candidates[:count]]
    
    def predict(self, numbers):
        """根据策略选择预测方法"""
        if self.strategy == 'ensemble_diverse':
            return self.predict_strategy_ensemble_diverse(numbers)
        elif self.strategy == 'adaptive_expand':
            return self.predict_strategy_adaptive_expand(numbers)
        elif self.strategy == 'model_rotation':
            return self.predict_strategy_model_rotation(numbers)
        elif self.strategy == 'union_top':
            return self.predict_strategy_union_top(numbers)
        elif self.strategy == 'anti_pattern':
            return self.predict_strategy_anti_pattern(numbers)
        elif self.strategy == 'coverage_max':
            return self.predict_strategy_coverage_max(numbers)
        else:
            return self.precise_predictor.predict(numbers)


def run_comparison(test_periods=400):
    """对比所有策略的400期回测结果"""
    print("=" * 80)
    print("TOP15 防连败策略对比 - 最近400期回测")
    print("=" * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    total = len(df)
    
    if total < test_periods + 30:
        test_periods = total - 30
    
    start_idx = total - test_periods
    
    strategies = {
        '基准(PreciseTop15)': 'baseline',
        '策略1:集成多样性': 'ensemble_diverse',
        '策略2:自适应扩展': 'adaptive_expand',
        '策略3:模型轮换': 'model_rotation',
        '策略4:并集共识': 'union_top',
        '策略5:反模式': 'anti_pattern',
        '策略6:最大覆盖': 'coverage_max',
    }
    
    results = {}
    
    for name, strategy_key in strategies.items():
        print(f"\n正在测试: {name}...", end=' ')
        
        if strategy_key == 'baseline':
            predictor = PreciseTop15Predictor()
        else:
            predictor = AntiStreakTop15Predictor(strategy=strategy_key)
        
        hits = []
        for i in range(start_idx, total):
            train_data = numbers[:i]
            actual = int(numbers[i])
            
            if strategy_key == 'baseline':
                top15 = predictor.predict(train_data)
                predictor.update_performance(top15, actual)
            else:
                top15 = predictor.predict(train_data)
                predictor.update(top15, actual)
            
            hits.append(actual in top15)
        
        # 计算连续不中统计
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
        streak_ge3 = sum(1 for s in streaks if s >= 3)
        streak_ge5 = sum(1 for s in streaks if s >= 5)
        streak_ge7 = sum(1 for s in streaks if s >= 7)
        
        # 连续不中分布
        streak_dist = Counter(streaks)
        
        results[name] = {
            'hit_rate': hit_rate,
            'max_streak': max_streak,
            'avg_streak': avg_streak,
            'streak_ge3': streak_ge3,
            'streak_ge5': streak_ge5,
            'streak_ge7': streak_ge7,
            'total_streaks': len(streaks),
            'streak_dist': streak_dist,
            'hits': hits
        }
        
        print(f"命中率={hit_rate:.1f}%, 最长连败={max_streak}, ≥5期={streak_ge5}次")
    
    # ========== 对比报告 ==========
    print(f"\n\n{'=' * 100}")
    print(f"{'策略对比总结':^100}")
    print(f"{'=' * 100}")
    print(f"{'策略名称':<22} {'命中率':>7} {'最长连败':>8} {'平均连败':>8} "
          f"{'≥3期次数':>8} {'≥5期次数':>8} {'≥7期次数':>8}")
    print("-" * 100)
    
    for name, r in results.items():
        print(f"{name:<22} {r['hit_rate']:>6.1f}% {r['max_streak']:>7} "
              f"{r['avg_streak']:>7.2f} {r['streak_ge3']:>8} "
              f"{r['streak_ge5']:>8} {r['streak_ge7']:>8}")
    
    # ========== 最优策略详细分析 ==========
    # 找到最长连败最短的策略
    best_max_streak = min(results.items(), key=lambda x: x[1]['max_streak'])
    best_ge5 = min(results.items(), key=lambda x: x[1]['streak_ge5'])
    best_hit_rate = max(results.items(), key=lambda x: x[1]['hit_rate'])
    
    print(f"\n{'=' * 80}")
    print(f"最优策略推荐")
    print(f"{'=' * 80}")
    print(f"最短最大连败: {best_max_streak[0]} (最长{best_max_streak[1]['max_streak']}期)")
    print(f"最少≥5期连败: {best_ge5[0]} ({best_ge5[1]['streak_ge5']}次)")
    print(f"最高命中率:   {best_hit_rate[0]} ({best_hit_rate[1]['hit_rate']:.1f}%)")
    
    # 综合评分（命中率×0.3 + 最大连败反向×0.3 + ≥5期次数反向×0.4）
    print(f"\n综合评分（命中率30% + 最大连败控制30% + ≥5期频次控制40%）:")
    scores = {}
    for name, r in results.items():
        # 归一化
        hit_score = r['hit_rate'] / 50  # 命中率归一化
        max_streak_score = 1 - r['max_streak'] / 20  # 越短越好
        ge5_score = 1 - r['streak_ge5'] / 20  # 越少越好
        
        total_score = hit_score * 0.3 + max_streak_score * 0.3 + ge5_score * 0.4
        scores[name] = total_score
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(sorted_scores, 1):
        r = results[name]
        marker = " ⭐" if rank == 1 else ""
        print(f"  {rank}. {name:<22} 综合分={score:.3f} "
              f"(命中{r['hit_rate']:.1f}%, 最长连败{r['max_streak']}, ≥5期{r['streak_ge5']}次){marker}")
    
    # ========== 最优策略连续不中分布 ==========
    best_name = sorted_scores[0][0]
    best_r = results[best_name]
    print(f"\n{'=' * 80}")
    print(f"最优策略 [{best_name}] 连续不中分布")
    print(f"{'=' * 80}")
    for length in sorted(best_r['streak_dist'].keys()):
        count = best_r['streak_dist'][length]
        bar = '█' * count
        print(f"  {length:>2}期: {count:>3}次 {bar}")
    
    # 对比基准改善
    baseline_r = results['基准(PreciseTop15)']
    print(f"\n{'=' * 80}")
    print(f"相比基准模型的改善")
    print(f"{'=' * 80}")
    print(f"{'指标':<16} {'基准':>10} {'最优策略':>10} {'改善':>10}")
    print("-" * 50)
    print(f"{'命中率':<16} {baseline_r['hit_rate']:>9.1f}% {best_r['hit_rate']:>9.1f}% "
          f"{best_r['hit_rate']-baseline_r['hit_rate']:>+9.1f}%")
    print(f"{'最长连败':<16} {baseline_r['max_streak']:>10} {best_r['max_streak']:>10} "
          f"{best_r['max_streak']-baseline_r['max_streak']:>+10}")
    print(f"{'≥5期连败次数':<16} {baseline_r['streak_ge5']:>10} {best_r['streak_ge5']:>10} "
          f"{best_r['streak_ge5']-baseline_r['streak_ge5']:>+10}")
    print(f"{'≥7期连败次数':<16} {baseline_r['streak_ge7']:>10} {best_r['streak_ge7']:>10} "
          f"{best_r['streak_ge7']-baseline_r['streak_ge7']:>+10}")
    print(f"{'平均连败':<16} {baseline_r['avg_streak']:>9.2f} {best_r['avg_streak']:>9.2f} "
          f"{best_r['avg_streak']-baseline_r['avg_streak']:>+9.2f}")


if __name__ == '__main__':
    run_comparison(400)
