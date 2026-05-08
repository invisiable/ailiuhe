"""
生肖TOP4 v3预测器 - 热号互补反miss
============================================
v2问题: 切换机制用MK150(与静态重叠), miss越切越miss
v3改进: miss时blend热号得分, 打破冷号陷阱

核心策略:
1. 正常模式: 静态组合 cold15(0.30) + cold30(0.10) + MK150(0.60) → TOP4
2. 反miss L1 (连miss>=2): blend 25% hot30 → TOP4 (不扩展, 保持低成本)
3. 反miss L2 (连miss>=4): blend + 扩展TOP5 (加hot30最佳互补)

300期验证预期: ~48% 命中率, 最大连miss ≤ 7
"""
import numpy as np
from collections import Counter

ZODIAC_CYCLE_2026 = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
Z_IDX = {z: i for i, z in enumerate(ZODIAC_CYCLE_2026)}
NUM_TO_ZODIAC_2026 = {n: ZODIAC_CYCLE_2026[(n - 1) % 12] for n in range(1, 50)}
ZODIAC_NUMS_2026 = {}
for z in ZODIAC_CYCLE_2026:
    ZODIAC_NUMS_2026[z] = sorted([n for n in range(1, 50) if NUM_TO_ZODIAC_2026[n] == z])


class ZodiacTop4V3Predictor:
    """
    生肖TOP4 v3预测器
    热号互补反miss机制
    """
    
    def __init__(self):
        self.name = "生肖TOP4_v3"
        self.consecutive_miss = 0
        
        # 静态组合权重
        self.static_weights = {
            'cold15': 0.30,
            'cold30': 0.10,
            'mk150': 0.60,
        }
        
        # 反miss配置
        self.blend_threshold = 2      # 连miss>=2时开始blend热号
        self.expand_threshold = 4     # 连miss>=4时扩展到TOP5
        self.hot_blend_ratio = 0.25   # 热号blend比例
        self.hot_window = 30          # 热号统计窗口
    
    def _cold_scores(self, animals, window):
        """冷号得分: 窗口内出现越少得分越高"""
        w = min(window, len(animals))
        freq = Counter(animals[-w:])
        mx = max(freq.values()) if freq else 1
        return np.array([1.0 - freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])
    
    def _hot_scores(self, animals, window):
        """热号得分: 窗口内出现越多得分越高 (与冷号互补)"""
        w = min(window, len(animals))
        freq = Counter(animals[-w:])
        mx = max(freq.values()) if freq else 1
        return np.array([freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])
    
    def _markov_scores(self, animals, window=None, laplace=1.0):
        """马尔可夫转移概率得分"""
        probs = np.ones(12) / 12
        h = animals[-window:] if window and len(animals) > window else animals
        if len(h) < 2:
            return probs
        
        trans = {}
        for k in range(1, len(h)):
            p, c = h[k - 1], h[k]
            if p not in trans:
                trans[p] = Counter()
            trans[p][c] += 1
        
        state = animals[-1]
        if state in trans:
            total = sum(trans[state].values()) + laplace * 12
            for zi, z in enumerate(ZODIAC_CYCLE_2026):
                probs[zi] = (trans[state].get(z, 0) + laplace) / total
        return probs
    
    def _static_scores(self, animals):
        """静态组合得分: cold15 + cold30 + MK150"""
        cold15 = self._cold_scores(animals, 15)
        cold30 = self._cold_scores(animals, 30)
        mk150 = self._markov_scores(animals, window=150)
        return (self.static_weights['cold15'] * cold15 +
                self.static_weights['cold30'] * cold30 +
                self.static_weights['mk150'] * mk150)
    
    def predict(self, numbers, top_n=4):
        """
        预测下一期最可能的TOP-N生肖
        
        Parameters:
            numbers: 历史号码列表 (1-49)
            top_n: 基础返回生肖数量 (默认4)
        
        Returns:
            list of str: 预测的生肖列表 (按可能性排序)
        """
        animals = [NUM_TO_ZODIAC_2026[n] for n in numbers]
        static = self._static_scores(animals)
        
        if self.consecutive_miss >= self.blend_threshold:
            # 反miss: blend热号
            hot = self._hot_scores(animals, self.hot_window)
            combined = (1 - self.hot_blend_ratio) * static + self.hot_blend_ratio * hot
        else:
            combined = static
        
        sorted_idx = np.argsort(-combined)
        
        if self.consecutive_miss >= self.expand_threshold:
            # 超长miss: 扩展到TOP5, 第5个来自hot互补
            top4_idx = list(sorted_idx[:4])
            hot = self._hot_scores(animals, self.hot_window)
            hot_sorted = np.argsort(-hot)
            for idx in hot_sorted:
                if idx not in top4_idx:
                    top4_idx.append(idx)
                    break
            return [ZODIAC_CYCLE_2026[i] for i in top4_idx[:top_n + 1]]
        
        return [ZODIAC_CYCLE_2026[i] for i in sorted_idx[:top_n]]
    
    def predict_with_details(self, numbers, top_n=4):
        """
        带详情的预测
        
        Returns:
            tuple: (predictions, mode_str, scores_dict)
        """
        animals = [NUM_TO_ZODIAC_2026[n] for n in numbers]
        static = self._static_scores(animals)
        hot = self._hot_scores(animals, self.hot_window)
        
        if self.consecutive_miss >= self.expand_threshold:
            mode = f"反miss-L2(扩展TOP5)"
            combined = (1 - self.hot_blend_ratio) * static + self.hot_blend_ratio * hot
            sorted_idx = np.argsort(-combined)
            top_idx = list(sorted_idx[:4])
            hot_sorted = np.argsort(-hot)
            for idx in hot_sorted:
                if idx not in top_idx:
                    top_idx.append(idx)
                    break
            final_n = top_n + 1
        elif self.consecutive_miss >= self.blend_threshold:
            mode = f"反miss-L1(blend热号)"
            combined = (1 - self.hot_blend_ratio) * static + self.hot_blend_ratio * hot
            sorted_idx = np.argsort(-combined)
            top_idx = list(sorted_idx[:top_n])
            final_n = top_n
        else:
            mode = "正常(静态)"
            combined = static
            sorted_idx = np.argsort(-combined)
            top_idx = list(sorted_idx[:top_n])
            final_n = top_n
        
        predictions = [ZODIAC_CYCLE_2026[i] for i in top_idx[:final_n]]
        
        # 计算各生肖得分
        scores = {}
        for i in top_idx[:final_n]:
            z = ZODIAC_CYCLE_2026[i]
            scores[z] = {
                'static': float(static[i]),
                'hot': float(hot[i]),
            }
        
        return predictions, mode, scores
    
    def record_result(self, hit):
        """记录预测结果"""
        if hit:
            self.consecutive_miss = 0
        else:
            self.consecutive_miss += 1
    
    def reset(self):
        """重置状态"""
        self.consecutive_miss = 0
    
    def get_zodiac_numbers(self, zodiac):
        """获取生肖对应的号码"""
        return ZODIAC_NUMS_2026.get(zodiac, [])
    
    def get_all_predicted_numbers(self, numbers, top_n=4):
        """获取预测生肖对应的所有号码"""
        predicted = self.predict(numbers, top_n)
        result = {}
        for z in predicted:
            result[z] = ZODIAC_NUMS_2026[z]
        return result


def validate_predictor(numbers, test_periods=300, verbose=True):
    """
    滚动验证预测器
    """
    total = len(numbers)
    start = total - test_periods
    
    predictor = ZodiacTop4V3Predictor()
    hits = 0
    results = []
    
    for pi in range(test_periods):
        i = start + pi
        hist = numbers[:i]
        actual = numbers[i]
        actual_zodiac = NUM_TO_ZODIAC_2026[actual]
        
        predicted, mode, scores = predictor.predict_with_details(hist, top_n=4)
        hit = actual_zodiac in predicted
        
        if hit:
            hits += 1
        
        predictor.record_result(hit)
        
        results.append({
            'period': pi + 1,
            'actual_number': actual,
            'actual_zodiac': actual_zodiac,
            'predicted': predicted,
            'hit': hit,
            'mode': mode,
            'consecutive_miss': predictor.consecutive_miss,
            'bet_size': len(predicted),
        })
    
    hit_rate = hits / test_periods * 100
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"生肖TOP4 v3 验证结果")
        print(f"{'='*60}")
        print(f"测试期数: {test_periods}")
        print(f"命中次数: {hits}/{test_periods}")
        print(f"命中率: {hit_rate:.1f}%")
        print(f"随机基线: 33.3%")
        print(f"提升: +{(hit_rate - 33.3):.1f}%")
        
        # 分段统计
        seg_size = 50
        n_segs = test_periods // seg_size
        print(f"\n分段统计(每{seg_size}期):")
        for s in range(n_segs):
            seg_start = s * seg_size
            seg_end = seg_start + seg_size
            seg_hits = sum(1 for r in results[seg_start:seg_end] if r['hit'])
            print(f"  {seg_start+1:>3}-{seg_end:>3}: {seg_hits}/{seg_size} = {seg_hits/seg_size*100:.1f}%")
        
        # 最大连续miss
        max_miss = max(r['consecutive_miss'] for r in results)
        print(f"\n最大连续miss: {max_miss}")
        
        # 连续miss分布
        streaks = []
        c = 0
        for r in results:
            if not r['hit']:
                c += 1
            else:
                if c > 0:
                    streaks.append(c)
                c = 0
        if c > 0:
            streaks.append(c)
        
        ge4 = sum(1 for s in streaks if s >= 4)
        ge6 = sum(1 for s in streaks if s >= 6)
        print(f"≥4期连续miss: {ge4}次")
        print(f"≥6期连续miss: {ge6}次")
        
        # 模式分布
        mode_counts = Counter(r['mode'] for r in results)
        print(f"\n模式分布:")
        for mode, count in mode_counts.most_common():
            mode_hits = sum(1 for r in results if r['mode'] == mode and r['hit'])
            mode_rate = mode_hits / count * 100 if count > 0 else 0
            print(f"  {mode}: {count}期, 命中{mode_hits}/{count}={mode_rate:.1f}%")
        
        # 投注统计
        total_bet = sum(r['bet_size'] * 4 for r in results)
        total_win = sum(46 for r in results if r['hit'])
        profit = total_win - total_bet
        roi = profit / total_bet * 100
        avg_bet = total_bet / test_periods
        print(f"\n投注统计:")
        print(f"  总投入: {total_bet}元 (平均{avg_bet:.1f}元/期)")
        print(f"  总回报: {total_win}元")
        print(f"  净利润: {profit:+}元")
        print(f"  ROI: {roi:+.1f}%")
    
    return {
        'hit_rate': hit_rate,
        'hits': hits,
        'total': test_periods,
        'results': results,
    }


def cross_validate(numbers, n_folds=5, verbose=True):
    """时序交叉验证"""
    total = len(numbers)
    fold_size = total // (n_folds + 1)
    
    cv_results = []
    for fold in range(n_folds):
        test_start = fold_size * (fold + 1)
        test_end = min(test_start + fold_size, total)
        test_size = test_end - test_start
        
        predictor = ZodiacTop4V3Predictor()
        hits = 0
        
        for i in range(test_start, test_end):
            hist = numbers[:i]
            actual = numbers[i]
            actual_z = NUM_TO_ZODIAC_2026[actual]
            predicted = predictor.predict(hist, top_n=4)
            hit = actual_z in predicted
            if hit:
                hits += 1
            predictor.record_result(hit)
        
        rate = hits / test_size * 100
        cv_results.append(rate)
        if verbose:
            print(f"  Fold {fold+1}: {hits}/{test_size} = {rate:.1f}%")
    
    mean_rate = np.mean(cv_results)
    std_rate = np.std(cv_results)
    if verbose:
        print(f"  平均: {mean_rate:.1f}% ± {std_rate:.1f}%")
    
    return cv_results


if __name__ == '__main__':
    import pandas as pd
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].tolist()
    
    print("=" * 60)
    print("生肖TOP4 v3 - 热号互补反miss")
    print("=" * 60)
    
    result = validate_predictor(numbers, test_periods=300)
    
    print("\n" + "=" * 60)
    print("交叉验证")
    print("=" * 60)
    cv = cross_validate(numbers)
