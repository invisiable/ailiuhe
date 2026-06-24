"""
尾数预测 - 自适应阶梯扩展模型 (目标: max_miss ≤ 4)
在 tail_digit_predictor.py 的基础上新增 TailDigitAdaptivePredictor
测试不同参数组合找出最稳健的配置
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tail_digit_predictor import TailDigitPredictor, TAIL_DIGIT_NUMBERS, number_to_tail


class TailDigitAdaptivePredictor:
    """
    自适应阶梯扩展尾数预测模型
    目标: 最大连续miss不超过4期
    
    策略:
    - miss_streak=0: 正常预测top4 (得分最高的4组尾数)
    - miss_streak=1: 扩展到5组, 排除上期预测, 从剩余中选
    - miss_streak=2: 扩展到7组, 排除已预测过的, 用冷号+综合评分
    - miss_streak>=3: 扩展到9组, 排除上期, 接近全覆盖确保止损
    """
    
    def __init__(self, schedule=None):
        self.base = TailDigitPredictor()
        self.recent_hits = []
        self.recent_preds = []
        # 阶梯扩展配置: miss_streak → top_n
        if schedule is None:
            self.schedule = {0: 4, 1: 5, 2: 7, 3: 9}
        else:
            self.schedule = schedule
    
    def predict(self, numbers, top_n_override=None):
        """预测下一期尾数"""
        hist_tails = [number_to_tail(n) for n in numbers]
        scores = self.base._calculate_scores(numbers)
        sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 计算连续miss
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        # 确定扩展级别
        if miss_streak in self.schedule:
            top_n = self.schedule[miss_streak]
        else:
            top_n = self.schedule[max(self.schedule.keys())]
        
        if top_n_override:
            top_n = top_n_override
        
        # 根据miss阶段选择策略
        if miss_streak == 0:
            # 正常模式: 取得分最高的
            predicted = [d for d, _ in sorted_all[:top_n]]
        elif miss_streak == 1:
            # 轮换: 排除上期预测
            excluded = set(self.recent_preds[-1]) if self.recent_preds else set()
            remaining = [(d, s) for d, s in sorted_all if d not in excluded]
            if len(remaining) >= top_n:
                predicted = [d for d, _ in remaining[:top_n]]
            else:
                predicted = [d for d, _ in sorted_all[:top_n]]
        elif miss_streak == 2:
            # 冷号+排除: 排除最近2期, 用冷号权重
            excluded = set()
            for p in self.recent_preds[-2:]:
                excluded.update(p)
            cold = self.base._cold_rebound_analysis(hist_tails)
            combined = {d: 0.5 * scores[d] + 0.5 * cold[d] for d in range(10)}
            combined_sorted = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            remaining = [(d, s) for d, s in combined_sorted if d not in excluded]
            if len(remaining) >= top_n:
                predicted = [d for d, _ in remaining[:top_n]]
            else:
                predicted = [d for d, _ in combined_sorted[:top_n]]
        else:
            # 救援模式(≥3 miss): 大幅扩展, 排除上期
            excluded = set(self.recent_preds[-1]) if self.recent_preds else set()
            remaining = [(d, s) for d, s in sorted_all if d not in excluded]
            if len(remaining) >= top_n:
                predicted = [d for d, _ in remaining[:top_n]]
            else:
                predicted = [d for d, _ in sorted_all[:top_n]]
        
        return predicted
    
    def predict_with_details(self, numbers):
        """带详情的预测"""
        scores = self.base._calculate_scores(numbers)
        predicted = self.predict(numbers)
        
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        if miss_streak in self.schedule:
            top_n = self.schedule[miss_streak]
        else:
            top_n = self.schedule[max(self.schedule.keys())]
        
        modes = {0: '正常(4组)', 1: '轮换(5组)', 2: '冷号(7组)', 3: '救援(9组)'}
        mode = modes.get(miss_streak, f'救援({top_n}组)')
        
        return predicted, scores, mode, miss_streak
    
    def record(self, predicted, hit):
        """记录结果"""
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)


def run_variants():
    """测试多种配置变体"""
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values.tolist()
    test_periods = min(300, len(df) - 50)
    start_idx = len(df) - test_periods
    
    print('=' * 95)
    print('🎯 自适应阶梯扩展 - 参数优化 (目标: max_miss ≤ 4, 覆盖尽量小)')
    print('=' * 95)
    print()
    
    # 测试不同配置
    configs = [
        # (name, schedule)
        ('4-5-7-9', {0: 4, 1: 5, 2: 7, 3: 9}),
        ('4-5-7-8', {0: 4, 1: 5, 2: 7, 3: 8}),
        ('4-5-6-9', {0: 4, 1: 5, 2: 6, 3: 9}),
        ('4-5-8-9', {0: 4, 1: 5, 2: 8, 3: 9}),
        ('4-6-7-9', {0: 4, 1: 6, 2: 7, 3: 9}),
        ('4-6-8-9', {0: 4, 1: 6, 2: 8, 3: 9}),
        ('4-4-7-9', {0: 4, 1: 4, 2: 7, 3: 9}),
        ('4-5-7-10', {0: 4, 1: 5, 2: 7, 3: 10}),
        ('4-4-6-9', {0: 4, 1: 4, 2: 6, 3: 9}),
        ('4-4-8-9', {0: 4, 1: 4, 2: 8, 3: 9}),
        ('4-5-6-10', {0: 4, 1: 5, 2: 6, 3: 10}),
        ('4-4-7-10', {0: 4, 1: 4, 2: 7, 3: 10}),
    ]
    
    results = []
    
    for name, schedule in configs:
        predictor = TailDigitAdaptivePredictor(schedule=schedule)
        hits = []
        
        for i in range(start_idx, len(numbers)):
            hist = numbers[:i]
            actual = numbers[i]
            actual_tail = number_to_tail(actual)
            predicted = predictor.predict(hist)
            hit = actual_tail in predicted
            hits.append(hit)
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
        
        # 平均覆盖
        coverages = [sum(len(TAIL_DIGIT_NUMBERS[d]) for d in p) for p in predictor.recent_preds]
        avg_cov = np.mean(coverages)
        avg_cov_pct = avg_cov / 49 * 100
        
        # 各模式使用次数
        mode_counts = Counter()
        ms = 0
        for h in hits:
            if ms in schedule:
                mode_counts[schedule[ms]] += 1
            else:
                mode_counts[schedule[max(schedule.keys())]] += 1
            if not h:
                ms += 1
            else:
                ms = 0
        
        results.append({
            'name': name,
            'hit_rate': hit_rate,
            'win3_rate': win3_rate,
            'win4_rate': win4_rate,
            'max_miss': max_miss,
            'avg_cov': avg_cov,
            'avg_cov_pct': avg_cov_pct,
            'mode_counts': mode_counts,
            'hits': hits,
        })
    
    # 排序: 先按达标(max_miss<=4), 再按平均覆盖从小到大
    results.sort(key=lambda x: (0 if x['max_miss'] <= 4 else 1, x['avg_cov']))
    
    print(f"{'配置':>12} {'单期命中':>8} {'3期窗口':>8} {'4期窗口':>8} {'max_miss':>8} {'平均覆盖':>8} {'达标':>4}")
    print('-' * 75)
    
    for r in results:
        target_met = '✅' if r['max_miss'] <= 4 else '❌'
        print(f"{r['name']:>12} {r['hit_rate']:>6.1f}% {r['win3_rate']:>6.1f}% {r['win4_rate']:>6.1f}% {r['max_miss']:>8} {r['avg_cov']:>5.0f}个 {target_met:>4}")
    
    # 详细分析达标方案
    passed = [r for r in results if r['max_miss'] <= 4]
    if passed:
        print(f'\n{"=" * 95}')
        print(f'✅ 达标方案详细分析')
        print(f'{"=" * 95}')
        
        for r in passed:
            print(f"\n  【{r['name']}】 max_miss={r['max_miss']}, 单期{r['hit_rate']:.1f}%, 平均覆盖{r['avg_cov']:.0f}个号码")
            print(f"    模式使用: ", end='')
            for n_groups, count in sorted(r['mode_counts'].items()):
                print(f"{n_groups}组={count}次 ", end='')
            print()
            
            # miss分布
            streaks = []
            c = 0
            for h in r['hits']:
                if not h:
                    c += 1
                else:
                    if c > 0:
                        streaks.append(c)
                    c = 0
            if c > 0:
                streaks.append(c)
            streak_counter = Counter(streaks)
            print(f"    miss分布: ", end='')
            for length in sorted(streak_counter.keys()):
                print(f"{length}期={streak_counter[length]}次 ", end='')
            print()
        
        # 推荐最优
        best = passed[0]  # 覆盖最小的达标方案
        print(f'\n{"=" * 95}')
        print(f'🏆 推荐方案: {best["name"]}')
        print(f'{"=" * 95}')
        print(f'  • max_miss = {best["max_miss"]} ≤ 4 ✅')
        print(f'  • 单期命中率: {best["hit_rate"]:.1f}%')
        print(f'  • 3期窗口: {best["win3_rate"]:.1f}%')
        print(f'  • 4期窗口: {best["win4_rate"]:.1f}%')
        print(f'  • 平均覆盖: {best["avg_cov"]:.0f}个号码 ({best["avg_cov_pct"]:.0f}%)')
        
        # 完整300期输出
        print(f'\n{"=" * 95}')
        print(f'📋 最优方案300期完整回测')
        print(f'{"=" * 95}')
        
        predictor = TailDigitAdaptivePredictor(schedule=dict(zip(range(4), [int(x) for x in best['name'].split('-')])))
        hits = []
        all_detail = []
        
        for i in range(start_idx, len(numbers)):
            hist = numbers[:i]
            actual = numbers[i]
            actual_tail = number_to_tail(actual)
            predicted = predictor.predict(hist)
            hit = actual_tail in predicted
            hits.append(hit)
            predictor.record(predicted, hit)
            
            ms = 0
            for j in range(len(hits) - 2, -1, -1):
                if not hits[j]:
                    ms += 1
                else:
                    break
            
            coverage = sum(len(TAIL_DIGIT_NUMBERS[d]) for d in predicted)
            all_detail.append({
                'period': i - start_idx + 1,
                'date': df.iloc[i]['date'],
                'actual': actual,
                'tail': actual_tail,
                'predicted': predicted,
                'hit': hit,
                'n_groups': len(predicted),
                'coverage': coverage,
            })
        
        print(f"\n{'期号':>4} {'日期':>12} {'号码':>4} {'尾数':>4} {'预测尾数':>28} {'组':>3} {'覆盖':>4} {'结果':>4} {'4期':>4}")
        print('-' * 85)
        
        for i, r in enumerate(all_detail):
            mark = '✅' if r['hit'] else '❌'
            pred_str = ','.join([str(d) for d in r['predicted']])
            
            if i >= 3:
                w4 = any(hits[i - 3:i + 1])
                w4_str = '✅' if w4 else '💀'
            else:
                w4_str = '--'
            
            print(f"{r['period']:>4} {r['date']:>12} {r['actual']:>4} {r['tail']:>4} {pred_str:>28} {r['n_groups']:>3} {r['coverage']:>4} {mark:>4} {w4_str:>4}")


if __name__ == '__main__':
    run_variants()
