"""
尾数预测 - 目标: 最大连续miss不超过4期
测试多种模型和策略组合, 找到能达成目标的最优方案
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tail_digit_predictor import TailDigitPredictor, TAIL_DIGIT_NUMBERS, number_to_tail


class GuaranteedCoveragePredictor:
    """
    保证覆盖模型: 追踪已覆盖的尾数, 确保N期内覆盖足够多的尾数
    核心思路: 如果连续miss, 强制切换到未被覆盖过的尾数
    """
    def __init__(self):
        self.base = TailDigitPredictor()
        self.recent_preds = []  # 最近几期的预测
        self.recent_hits = []
    
    def predict(self, numbers, top_n=4):
        hist_tails = [number_to_tail(n) for n in numbers]
        scores = self.base._calculate_scores(numbers)
        sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        if miss_streak == 0:
            # 正常: TOP4
            predicted = [d for d, _ in sorted_all[:top_n]]
        elif miss_streak == 1:
            # 1 miss: 从未覆盖过的尾数中选
            covered = set(self.recent_preds[-1]) if self.recent_preds else set()
            remaining = [(d, s) for d, s in sorted_all if d not in covered]
            if len(remaining) >= top_n:
                predicted = [d for d, _ in remaining[:top_n]]
            else:
                predicted = [d for d, _ in sorted_all[:top_n]]
        elif miss_streak == 2:
            # 2 miss: 排除最近2期预测
            covered = set()
            for p in self.recent_preds[-2:]:
                covered.update(p)
            remaining = [(d, s) for d, s in sorted_all if d not in covered]
            if len(remaining) >= top_n:
                predicted = [d for d, _ in remaining[:top_n]]
            else:
                # 不够选, 用冷号体系
                cold = self.base._cold_rebound_analysis(hist_tails)
                cold_sorted = sorted(cold.items(), key=lambda x: x[1], reverse=True)
                remaining = [(d, s) for d, s in cold_sorted if d not in covered]
                predicted = [d for d, _ in remaining[:top_n]] if len(remaining) >= top_n else [d for d, _ in cold_sorted[:top_n]]
        else:
            # >=3 miss: 全力覆盖未出现的尾数
            covered = set()
            for p in self.recent_preds[-miss_streak:]:
                covered.update(p)
            # 剩余未覆盖
            uncovered = [d for d in range(10) if d not in covered]
            if len(uncovered) >= top_n:
                # 从未覆盖中选最高分的
                uncovered_scores = [(d, scores[d]) for d in uncovered]
                uncovered_scores.sort(key=lambda x: x[1], reverse=True)
                predicted = [d for d, _ in uncovered_scores[:top_n]]
            else:
                # 未覆盖不够, 全选+补充
                predicted = uncovered[:]
                rest = [(d, s) for d, s in sorted_all if d not in set(predicted)]
                predicted += [d for d, _ in rest[:top_n - len(predicted)]]
        
        return predicted
    
    def record(self, predicted, hit):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)


class MarkovPredictor:
    """
    马尔可夫链预测: 基于转移概率
    """
    def __init__(self):
        self.base = TailDigitPredictor()
    
    def predict(self, numbers, top_n=4):
        hist_tails = [number_to_tail(n) for n in numbers]
        if len(hist_tails) < 10:
            return list(range(top_n))
        
        # 计算转移矩阵
        transitions = defaultdict(lambda: Counter())
        for i in range(len(hist_tails) - 1):
            transitions[hist_tails[i]][hist_tails[i + 1]] += 1
        
        # 当前尾数
        current = hist_tails[-1]
        trans = transitions[current]
        
        if not trans:
            scores = self.base._calculate_scores(numbers)
            sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [d for d, _ in sorted_all[:top_n]]
        
        total = sum(trans.values())
        probs = {d: trans[d] / total for d in range(10)}
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in sorted_probs[:top_n]]


class EnsembleVotePredictor:
    """
    集成投票模型: 多个子模型投票
    """
    def __init__(self):
        self.base = TailDigitPredictor()
    
    def predict(self, numbers, top_n=4):
        hist_tails = [number_to_tail(n) for n in numbers]
        if len(hist_tails) < 20:
            return list(range(top_n))
        
        scores = self.base._calculate_scores(numbers)
        
        # 子模型1: 频率 (最近20期)
        freq = Counter(hist_tails[-20:])
        freq_scores = {d: freq.get(d, 0) / 20 for d in range(10)}
        
        # 子模型2: 冷号
        cold_scores = self.base._cold_rebound_analysis(hist_tails)
        
        # 子模型3: 趋势
        trend_scores = self.base._trend_momentum_analysis(hist_tails)
        
        # 子模型4: 周期
        cycle_scores = self.base._cycle_analysis(hist_tails)
        
        # 子模型5: 间隔
        gap_scores = self.base._gap_pattern_analysis(hist_tails)
        
        # 子模型6: 马尔可夫
        transitions = defaultdict(lambda: Counter())
        for i in range(len(hist_tails) - 1):
            transitions[hist_tails[i]][hist_tails[i + 1]] += 1
        current = hist_tails[-1]
        trans = transitions[current]
        total = sum(trans.values()) if trans else 1
        markov_scores = {d: trans.get(d, 0) / total for d in range(10)}
        
        # 投票: 每个子模型选TOP4, 计算被选中次数
        votes = Counter()
        for model_scores in [freq_scores, cold_scores, trend_scores, cycle_scores, gap_scores, markov_scores]:
            top4 = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:4]
            for d, _ in top4:
                votes[d] += 1
        
        # 按投票数排序, 相同票数按综合得分
        candidates = [(d, votes[d], scores[d]) for d in range(10)]
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [d for d, _, _ in candidates[:top_n]]


class AdaptiveExpansionPredictor:
    """
    自适应扩展模型: 连续miss时逐步增加覆盖
    目标: max_miss ≤ 4
    """
    def __init__(self, expansion_schedule=None):
        self.base = TailDigitPredictor()
        self.recent_hits = []
        self.recent_preds = []
        # 扩展计划: miss_streak → (top_n, strategy)
        if expansion_schedule is None:
            self.schedule = {
                0: (4, 'normal'),
                1: (4, 'rotate'),
                2: (5, 'cold'),
                3: (7, 'rescue'),
            }
        else:
            self.schedule = expansion_schedule
    
    def predict(self, numbers, top_n_override=None):
        hist_tails = [number_to_tail(n) for n in numbers]
        scores = self.base._calculate_scores(numbers)
        sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        # 获取扩展参数
        if miss_streak in self.schedule:
            top_n, strategy = self.schedule[miss_streak]
        else:
            top_n, strategy = self.schedule[max(self.schedule.keys())]
        
        if top_n_override:
            top_n = top_n_override
        
        if strategy == 'normal':
            predicted = [d for d, _ in sorted_all[:top_n]]
        elif strategy == 'rotate':
            excluded = set(self.recent_preds[-1]) if self.recent_preds else set()
            remaining = [(d, s) for d, s in sorted_all if d not in excluded]
            predicted = [d for d, _ in remaining[:top_n]] if len(remaining) >= top_n else [d for d, _ in sorted_all[:top_n]]
        elif strategy == 'cold':
            cold = self.base._cold_rebound_analysis(hist_tails)
            combined = {d: 0.5 * scores[d] + 0.5 * cold[d] for d in range(10)}
            combined_sorted = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            excluded = set()
            for p in self.recent_preds[-2:]:
                excluded.update(p)
            remaining = [(d, s) for d, s in combined_sorted if d not in excluded]
            predicted = [d for d, _ in remaining[:top_n]] if len(remaining) >= top_n else [d for d, _ in combined_sorted[:top_n]]
        elif strategy == 'rescue':
            # 救援: 最大覆盖+排除已覆盖
            covered = set()
            for p in self.recent_preds[-miss_streak:]:
                covered.update(p)
            uncovered = [(d, scores[d]) for d in range(10) if d not in covered]
            uncovered.sort(key=lambda x: x[1], reverse=True)
            if len(uncovered) >= top_n:
                predicted = [d for d, _ in uncovered[:top_n]]
            else:
                predicted = [d for d, _ in uncovered]
                rest = [(d, s) for d, s in sorted_all if d not in set(predicted)]
                predicted += [d for d, _ in rest[:top_n - len(predicted)]]
        else:
            predicted = [d for d, _ in sorted_all[:top_n]]
        
        return predicted
    
    def record(self, predicted, hit):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)


class DiversityCoveragePredictor:
    """
    多样性覆盖模型: 确保3期内覆盖尽可能多的不同尾数
    通过限制连续两期选同一尾数来最大化覆盖面积
    """
    def __init__(self):
        self.base = TailDigitPredictor()
        self.recent_preds = []
        self.recent_hits = []
    
    def predict(self, numbers, top_n=4):
        hist_tails = [number_to_tail(n) for n in numbers]
        scores = self.base._calculate_scores(numbers)
        sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        # 多样性约束: 最多1个尾数可以和上期重复
        if self.recent_preds and miss_streak > 0:
            last_pred = set(self.recent_preds[-1])
            # 策略: 至少选2个新尾数
            new_digits = [(d, s) for d, s in sorted_all if d not in last_pred]
            old_digits = [(d, s) for d, s in sorted_all if d in last_pred]
            
            if miss_streak >= 2 and len(self.recent_preds) >= 2:
                # 排除最近2期全部
                excluded = set()
                for p in self.recent_preds[-2:]:
                    excluded.update(p)
                new_digits = [(d, s) for d, s in sorted_all if d not in excluded]
                if len(new_digits) >= top_n:
                    predicted = [d for d, _ in new_digits[:top_n]]
                else:
                    predicted = [d for d, _ in new_digits]
                    rest = [(d, s) for d, s in sorted_all if d not in set(predicted)]
                    predicted += [d for d, _ in rest[:top_n - len(predicted)]]
            else:
                # 选3个新 + 1个旧(最高分的)
                predicted = [d for d, _ in new_digits[:3]]
                if old_digits:
                    predicted.append(old_digits[0][0])
                else:
                    predicted.append(new_digits[3][0] if len(new_digits) > 3 else sorted_all[3][0])
                predicted = predicted[:top_n]
        else:
            predicted = [d for d, _ in sorted_all[:top_n]]
        
        return predicted
    
    def record(self, predicted, hit):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)


class ComplementPredictor:
    """
    互补预测模型: 分析miss的尾数特征, 下一期针对性补充
    """
    def __init__(self):
        self.base = TailDigitPredictor()
        self.recent_preds = []
        self.recent_hits = []
        self.recent_actuals = []  # 记录实际出现的尾数
    
    def predict(self, numbers, top_n=4):
        hist_tails = [number_to_tail(n) for n in numbers]
        scores = self.base._calculate_scores(numbers)
        sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        if miss_streak == 0:
            predicted = [d for d, _ in sorted_all[:top_n]]
        elif miss_streak >= 1:
            # 分析最近miss期间实际出现的尾数
            missed_actuals = self.recent_actuals[-miss_streak:] if self.recent_actuals else []
            
            # 核心思路: 实际出现的尾数周围有聚集效应
            # 给邻近尾数加分
            bonus = defaultdict(float)
            for actual in missed_actuals:
                bonus[(actual - 1) % 10] += 0.3
                bonus[(actual + 1) % 10] += 0.3
                bonus[actual] += 0.2  # 可能重复出现
            
            # 排除已预测过的
            excluded = set()
            for p in self.recent_preds[-miss_streak:]:
                excluded.update(p)
            
            # 综合打分
            combined = {}
            for d in range(10):
                combined[d] = scores[d] + bonus.get(d, 0)
                if d in excluded:
                    combined[d] *= 0.3  # 降权已预测过的
            
            combined_sorted = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            predicted = [d for d, _ in combined_sorted[:top_n]]
        else:
            predicted = [d for d, _ in sorted_all[:top_n]]
        
        return predicted
    
    def record(self, predicted, hit, actual_tail):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)
        self.recent_actuals.append(actual_tail)


class HybridWindowPredictor:
    """
    混合窗口模型: 结合固定窗口+智能轮换+扩展
    核心: 正常4组固定3期, miss后智能切换; 接近危险时扩展
    """
    def __init__(self, max_allowed_miss=4):
        self.base = TailDigitPredictor()
        self.recent_hits = []
        self.recent_preds = []
        self.current_window_pred = None
        self.window_count = 0
        self.max_miss = max_allowed_miss
    
    def predict(self, numbers, top_n=4):
        hist_tails = [number_to_tail(n) for n in numbers]
        scores = self.base._calculate_scores(numbers)
        sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        # 决定是否需要重新预测
        need_refresh = (self.current_window_pred is None or 
                       self.window_count >= 3 or
                       (self.recent_hits and self.recent_hits[-1]))  # 中了就重新预测
        
        if miss_streak >= self.max_miss - 1:
            # 危险! 下一期必须中 - 大幅扩展
            top_n = 7  # 扩到7组 (~34个号码, 69%覆盖)
            need_refresh = True
        elif miss_streak >= self.max_miss - 2:
            # 警告 - 适度扩展
            top_n = 6
            need_refresh = True
        elif miss_streak >= 2:
            top_n = 5
            need_refresh = True
        
        if need_refresh:
            if miss_streak >= 2:
                # 排除已预测过的尾数
                excluded = set()
                for p in self.recent_preds[-miss_streak:]:
                    excluded.update(p)
                remaining = [(d, s) for d, s in sorted_all if d not in excluded]
                if len(remaining) >= top_n:
                    self.current_window_pred = [d for d, _ in remaining[:top_n]]
                else:
                    self.current_window_pred = [d for d, _ in remaining]
                    rest = [(d, s) for d, s in sorted_all if d not in set(self.current_window_pred)]
                    self.current_window_pred += [d for d, _ in rest[:top_n - len(self.current_window_pred)]]
            else:
                self.current_window_pred = [d for d, _ in sorted_all[:top_n]]
            self.window_count = 0
        
        self.window_count += 1
        return self.current_window_pred
    
    def record(self, predicted, hit):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)


def run_backtest(numbers, predictor_class, name, **kwargs):
    """通用回测框架"""
    test_periods = min(300, len(numbers) - 50)
    start_idx = len(numbers) - test_periods
    
    predictor = predictor_class(**kwargs)
    hits = []
    
    for i in range(start_idx, len(numbers)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)
        
        if hasattr(predictor, 'predict'):
            predicted = predictor.predict(hist)
        else:
            predicted = predictor(hist)
        
        hit = actual_tail in predicted
        hits.append(hit)
        
        if hasattr(predictor, 'record'):
            if 'actual_tail' in predictor.record.__code__.co_varnames:
                predictor.record(predicted, hit, actual_tail)
            else:
                predictor.record(predicted, hit)
    
    # 统计
    hit_count = sum(hits)
    hit_rate = hit_count / len(hits) * 100
    
    windows_3 = [any(hits[i:i + 3]) for i in range(len(hits) - 2)]
    win3_rate = sum(windows_3) / len(windows_3) * 100
    
    windows_4 = [any(hits[i:i + 4]) for i in range(len(hits) - 3)]
    win4_rate = sum(windows_4) / len(windows_4) * 100
    
    max_miss = 0
    cur = 0
    for h in hits:
        if not h:
            cur += 1
            max_miss = max(max_miss, cur)
        else:
            cur = 0
    
    # 平均覆盖数 (检查predictor的recent_preds)
    avg_coverage = 4  # 默认
    if hasattr(predictor, 'recent_preds') and predictor.recent_preds:
        coverages = [sum(len(TAIL_DIGIT_NUMBERS[d]) for d in p) for p in predictor.recent_preds]
        avg_coverage = np.mean(coverages) / 49 * 100  # 覆盖率百分比
    
    return {
        'name': name,
        'hit_rate': hit_rate,
        'win3_rate': win3_rate,
        'win4_rate': win4_rate,
        'max_miss': max_miss,
        'avg_coverage': avg_coverage,
        'hits': hits,
    }


def main():
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values.tolist()
    
    print('=' * 95)
    print('🎯 尾数预测 - 目标: 最大连续miss ≤ 4期')
    print('=' * 95)
    print()
    print('测试多种模型策略, 寻找能达成 max_miss ≤ 4 的方案')
    print()
    
    results = []
    
    # 1. 保证覆盖模型
    results.append(run_backtest(numbers, GuaranteedCoveragePredictor, '保证覆盖(固定4组)'))
    
    # 2. 集成投票
    results.append(run_backtest(numbers, EnsembleVotePredictor, '集成投票(固定4组)'))
    
    # 3. 多样性覆盖
    results.append(run_backtest(numbers, DiversityCoveragePredictor, '多样性覆盖(固定4组)'))
    
    # 4. 互补预测
    results.append(run_backtest(numbers, ComplementPredictor, '互补预测(固定4组)'))
    
    # 5. 自适应扩展 - 温和版 (4-4-5-6)
    results.append(run_backtest(numbers, AdaptiveExpansionPredictor, '自适应(4-4-5-6)',
                                expansion_schedule={0: (4, 'normal'), 1: (4, 'rotate'), 2: (5, 'cold'), 3: (6, 'rescue')}))
    
    # 6. 自适应扩展 - 激进版 (4-5-6-8)
    results.append(run_backtest(numbers, AdaptiveExpansionPredictor, '自适应(4-5-6-8)',
                                expansion_schedule={0: (4, 'normal'), 1: (5, 'rotate'), 2: (6, 'cold'), 3: (8, 'rescue')}))
    
    # 7. 自适应扩展 - 极限版 (4-5-7-9)
    results.append(run_backtest(numbers, AdaptiveExpansionPredictor, '自适应(4-5-7-9)',
                                expansion_schedule={0: (4, 'normal'), 1: (5, 'rotate'), 2: (7, 'cold'), 3: (9, 'rescue')}))
    
    # 8. 混合窗口 + 扩展 (max_miss=4)
    results.append(run_backtest(numbers, HybridWindowPredictor, '混合窗口+扩展(目标4)', max_allowed_miss=4))
    
    # 9. 马尔可夫 + 保证覆盖
    # 特殊: 组合两个模型
    class MarkovGuaranteed:
        def __init__(self):
            self.markov = MarkovPredictor()
            self.guarantee = GuaranteedCoveragePredictor()
            self.recent_hits = []
            self.recent_preds = []
        def predict(self, numbers, top_n=4):
            miss_streak = 0
            for j in range(len(self.recent_hits) - 1, -1, -1):
                if not self.recent_hits[j]:
                    miss_streak += 1
                else:
                    break
            if miss_streak <= 1:
                return self.markov.predict(numbers, top_n)
            else:
                return self.guarantee.predict(numbers, top_n)
        def record(self, predicted, hit):
            self.recent_hits.append(hit)
            self.recent_preds.append(predicted)
            self.guarantee.recent_hits = self.recent_hits
            self.guarantee.recent_preds = self.recent_preds
    
    results.append(run_backtest(numbers, MarkovGuaranteed, '马尔可夫+保证覆盖'))
    
    # 10. 完全互斥轮换 (确保3期内覆盖所有10个尾数)
    class FullRotationGuarantee:
        def __init__(self):
            self.base = TailDigitPredictor()
            self.recent_preds = []
            self.recent_hits = []
            self.rotation_phase = 0
        def predict(self, numbers, top_n=4):
            scores = self.base._calculate_scores(numbers)
            sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            miss_streak = 0
            for j in range(len(self.recent_hits) - 1, -1, -1):
                if not self.recent_hits[j]:
                    miss_streak += 1
                else:
                    break
            
            if miss_streak == 0:
                self.rotation_phase = 0
                predicted = [d for d, _ in sorted_all[:4]]
            elif miss_streak == 1:
                # 选和上期完全不重叠的4个
                excluded = set(self.recent_preds[-1])
                remaining = [(d, s) for d, s in sorted_all if d not in excluded]
                predicted = [d for d, _ in remaining[:4]]
            elif miss_streak == 2:
                # 剩余2个 + 最高分2个
                covered = set()
                for p in self.recent_preds[-2:]:
                    covered.update(p)
                uncovered = [(d, scores[d]) for d in range(10) if d not in covered]
                uncovered.sort(key=lambda x: x[1], reverse=True)
                # 全部未覆盖的 + 补充到4个
                predicted = [d for d, _ in uncovered]
                if len(predicted) < 4:
                    rest = [(d, s) for d, s in sorted_all if d not in set(predicted)]
                    predicted += [d for d, _ in rest[:4 - len(predicted)]]
                predicted = predicted[:4]
            else:
                # 3+ miss - 不可能(理论上3期已覆盖全部), 用得分最高的
                predicted = [d for d, _ in sorted_all[:4]]
            
            return predicted
        def record(self, predicted, hit):
            self.recent_preds.append(predicted)
            self.recent_hits.append(hit)
    
    results.append(run_backtest(numbers, FullRotationGuarantee, '完全互斥轮换(4组)'))
    
    # 输出结果
    print(f"{'模型':>24} {'单期命中':>8} {'3期窗口':>8} {'4期窗口':>8} {'最大miss':>8} {'平均覆盖':>8} {'达标':>4}")
    print('-' * 95)
    
    results.sort(key=lambda x: x['max_miss'])
    
    for r in results:
        target_met = '✅' if r['max_miss'] <= 4 else '❌'
        print(f"{r['name']:>24} {r['hit_rate']:>6.1f}% {r['win3_rate']:>6.1f}% {r['win4_rate']:>6.1f}% {r['max_miss']:>8} {r['avg_coverage']:>6.1f}% {target_met:>4}")
    
    print()
    print('=' * 95)
    
    # 找出达标的方案
    passed = [r for r in results if r['max_miss'] <= 4]
    if passed:
        print(f'\n✅ 达标方案 (max_miss ≤ 4):')
        for r in passed:
            print(f"   {r['name']}: max_miss={r['max_miss']}, 单期{r['hit_rate']:.1f}%, 3期窗口{r['win3_rate']:.1f}%, 覆盖{r['avg_coverage']:.1f}%")
    else:
        print(f'\n❌ 暂无方案达标, 最接近的:')
        best = results[0]
        print(f"   {best['name']}: max_miss={best['max_miss']}")
        print()
        print('分析: 需要更激进的扩展策略或组合方案')
    
    # 详细分析最优方案
    best = results[0]
    print(f'\n{"=" * 95}')
    print(f'📊 最优方案详细分析: {best["name"]}')
    print(f'{"=" * 95}')
    
    # 连续miss分布
    hits = best['hits']
    streaks = []
    c = 0
    for h in hits:
        if not h:
            c += 1
        else:
            if c > 0:
                streaks.append(c)
            c = 0
    if c > 0:
        streaks.append(c)
    
    print(f'\n连续miss分布:')
    streak_counter = Counter(streaks)
    for length in sorted(streak_counter.keys()):
        bar = '█' * streak_counter[length]
        danger = ' ⚠️' if length >= 4 else ''
        print(f'  {length}期: {streak_counter[length]:>3}次 {bar}{danger}')
    
    # 如果没达标, 尝试更极端的方案
    if not passed:
        print(f'\n{"=" * 95}')
        print('🔬 尝试更激进的方案...')
        print('=' * 95)
        
        # 终极方案: 阶梯扩展 4-5-7-10(全覆盖)
        class UltimatePredictor:
            def __init__(self):
                self.base = TailDigitPredictor()
                self.recent_hits = []
                self.recent_preds = []
            def predict(self, numbers, top_n=4):
                scores = self.base._calculate_scores(numbers)
                sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                hist_tails = [number_to_tail(n) for n in numbers]
                
                miss_streak = 0
                for j in range(len(self.recent_hits) - 1, -1, -1):
                    if not self.recent_hits[j]:
                        miss_streak += 1
                    else:
                        break
                
                if miss_streak == 0:
                    top_n = 4
                    predicted = [d for d, _ in sorted_all[:top_n]]
                elif miss_streak == 1:
                    top_n = 5
                    excluded = set(self.recent_preds[-1])
                    remaining = [(d, s) for d, s in sorted_all if d not in excluded]
                    predicted = [d for d, _ in remaining[:top_n]]
                elif miss_streak == 2:
                    top_n = 7
                    excluded = set()
                    for p in self.recent_preds[-2:]:
                        excluded.update(p)
                    remaining = [(d, s) for d, s in sorted_all if d not in excluded]
                    if len(remaining) >= top_n:
                        predicted = [d for d, _ in remaining[:top_n]]
                    else:
                        predicted = [d for d, _ in sorted_all[:top_n]]
                else:  # >=3
                    # 全覆盖! 选全部10个尾数
                    top_n = 10
                    predicted = list(range(10))
                
                return predicted[:top_n]
            def record(self, predicted, hit):
                self.recent_preds.append(predicted)
                self.recent_hits.append(hit)
        
        r_ultimate = run_backtest(numbers, UltimatePredictor, '终极(4-5-7-全覆盖)')
        print(f"\n  终极方案: max_miss={r_ultimate['max_miss']}, 单期{r_ultimate['hit_rate']:.1f}%, 覆盖{r_ultimate['avg_coverage']:.1f}%")
        
        if r_ultimate['max_miss'] <= 4:
            print('  ✅ 达标! 但需要在miss≥3时覆盖全部尾数')
        
        # 尝试 4-6-8-9
        class AggressivePredictor:
            def __init__(self):
                self.base = TailDigitPredictor()
                self.recent_hits = []
                self.recent_preds = []
            def predict(self, numbers, top_n=4):
                scores = self.base._calculate_scores(numbers)
                sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                hist_tails = [number_to_tail(n) for n in numbers]
                
                miss_streak = 0
                for j in range(len(self.recent_hits) - 1, -1, -1):
                    if not self.recent_hits[j]:
                        miss_streak += 1
                    else:
                        break
                
                if miss_streak == 0:
                    top_n = 4
                    predicted = [d for d, _ in sorted_all[:top_n]]
                elif miss_streak == 1:
                    top_n = 6
                    excluded = set(self.recent_preds[-1])
                    remaining = [(d, s) for d, s in sorted_all if d not in excluded]
                    predicted = [d for d, _ in remaining[:top_n]] if len(remaining) >= top_n else [d for d, _ in sorted_all[:top_n]]
                elif miss_streak == 2:
                    top_n = 8
                    excluded = set()
                    for p in self.recent_preds[-2:]:
                        excluded.update(p)
                    remaining = [(d, s) for d, s in sorted_all if d not in excluded]
                    predicted = [d for d, _ in remaining[:top_n]] if len(remaining) >= top_n else [d for d, _ in sorted_all[:top_n]]
                else:
                    top_n = 9
                    excluded = set(self.recent_preds[-1]) if self.recent_preds else set()
                    remaining = [(d, s) for d, s in sorted_all if d not in excluded]
                    predicted = [d for d, _ in remaining[:top_n]] if len(remaining) >= top_n else [d for d, _ in sorted_all[:top_n]]
                
                return predicted
            def record(self, predicted, hit):
                self.recent_preds.append(predicted)
                self.recent_hits.append(hit)
        
        r_aggressive = run_backtest(numbers, AggressivePredictor, '激进(4-6-8-9)')
        print(f"  激进方案: max_miss={r_aggressive['max_miss']}, 单期{r_aggressive['hit_rate']:.1f}%, 覆盖{r_aggressive['avg_coverage']:.1f}%")
        
        if r_aggressive['max_miss'] <= 4:
            print('  ✅ 达标!')


if __name__ == '__main__':
    main()
