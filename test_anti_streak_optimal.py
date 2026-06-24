"""
进一步优化：组合策略5(反模式)+策略3(模型轮换)+策略6(最大覆盖)的优点
目标：在保持高命中率的同时，进一步压缩最大连败
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from top15_predictor import Top15Predictor
from precise_top15_predictor import PreciseTop15Predictor
from smart_top15_predictor import SmartTop15Predictor


class AntiStreakOptimalPredictor:
    """
    最优防连败预测器
    核心：反模式盲区覆盖 + 模型轮换 + 区域均衡保障
    """
    
    def __init__(self):
        self.base_predictor = Top15Predictor()
        self.precise_predictor = PreciseTop15Predictor()
        self.smart_predictor = SmartTop15Predictor()
        
        self.consecutive_misses = 0
        self.recent_predictions = []
        self.recent_actuals = []
        self.total_predictions = 0
    
    def update(self, prediction, actual):
        hit = actual in prediction
        if hit:
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
        
        self.recent_predictions.append(prediction)
        self.recent_actuals.append(actual)
        self.total_predictions += 1
        if len(self.recent_predictions) > 30:
            self.recent_predictions.pop(0)
            self.recent_actuals.pop(0)
    
    def predict(self, numbers):
        """
        组合策略：
        - 连败0-1期：用精准模型+反模式盲区注入
        - 连败2-3期：切换智能模型+盲区注入+区域保障
        - 连败4期+：三模型投票+强制区域均衡+大量盲区注入
        """
        numbers_list = list(numbers) if not isinstance(numbers, list) else numbers
        
        if self.consecutive_misses <= 1:
            base = self.precise_predictor.predict(numbers)
            # 少量反模式注入
            return self._inject_anti_pattern(base, numbers_list, inject_count=3)
        elif self.consecutive_misses <= 3:
            base = self.smart_predictor.predict(numbers)
            # 中量注入+区域保障
            result = self._inject_anti_pattern(base, numbers_list, inject_count=4)
            return self._ensure_zone_coverage(result, numbers_list)
        else:
            # 三模型投票+强注入+区域保障
            return self._full_defensive_predict(numbers_list)
    
    def _inject_anti_pattern(self, base_pred, numbers_list, inject_count=3):
        """反模式注入：分析盲区，注入互补号码"""
        if len(self.recent_actuals) < 5:
            return base_pred[:15]
        
        # 分析最近miss的号码
        recent_misses = []
        for pred, actual in zip(self.recent_predictions[-10:], self.recent_actuals[-10:]):
            if actual not in pred:
                recent_misses.append(actual)
        
        if not recent_misses:
            return base_pred[:15]
        
        # 盲区号码候选
        blind_candidates = []
        recent_5 = set(numbers_list[-5:])
        
        # 从miss号码的邻域选择
        for n in recent_misses[-5:]:
            for offset in [-3, -2, -1, 1, 2, 3]:
                neighbor = n + offset
                if 1 <= neighbor <= 49 and neighbor not in recent_5 and neighbor not in base_pred[:10]:
                    blind_candidates.append(neighbor)
        
        # 从miss号码所在区间选高概率号
        miss_zones = set()
        for n in recent_misses:
            miss_zones.add((n - 1) // 10)
        
        freq_30 = Counter(numbers_list[-30:])
        for zone_idx in miss_zones:
            start = zone_idx * 10 + 1
            end = min(start + 9, 49)
            for n in range(start, end + 1):
                if n not in recent_5 and n not in base_pred[:10]:
                    if freq_30.get(n, 0) >= 1:
                        blind_candidates.append(n)
        
        blind_candidates = list(set(blind_candidates))
        
        if not blind_candidates:
            return base_pred[:15]
        
        # 确定性选择（基于数据特征而非随机）
        # 按30期频率+间隔综合排序
        def score_candidate(n):
            freq = freq_30.get(n, 0)
            # 计算间隔
            gap = 30
            for i, num in enumerate(reversed(numbers_list[-30:])):
                if num == n:
                    gap = i + 1
                    break
            # 间隔5-15最佳
            gap_score = 2.0 if 5 <= gap <= 15 else (1.0 if gap > 15 else 0.5)
            return freq * 0.5 + gap_score
        
        blind_candidates.sort(key=score_candidate, reverse=True)
        inject = blind_candidates[:inject_count]
        
        # 替换base_pred中最低优先级的号码
        result = base_pred[:15 - inject_count]
        for n in inject:
            if n not in result:
                result.append(n)
        
        # 补齐
        for n in base_pred:
            if n not in result:
                result.append(n)
            if len(result) >= 15:
                break
        
        return result[:15]
    
    def _ensure_zone_coverage(self, pred, numbers_list):
        """确保区域覆盖：每个区间至少1个号码"""
        zones = [(1,10), (11,20), (21,30), (31,40), (41,49)]
        result = list(pred[:15])
        
        for start, end in zones:
            has_zone = any(start <= n <= end for n in result)
            if not has_zone:
                # 从该区间选一个高分号码
                freq_30 = Counter(numbers_list[-30:])
                recent_5 = set(numbers_list[-5:])
                best_n = None
                best_score = -1
                for n in range(start, end + 1):
                    if n not in recent_5:
                        score = freq_30.get(n, 0)
                        if score > best_score:
                            best_score = score
                            best_n = n
                if best_n and best_n not in result:
                    result[-1] = best_n  # 替换最后一个
        
        return result[:15]
    
    def _full_defensive_predict(self, numbers_list):
        """全防御预测：三模型投票+强制多样性"""
        base_pred = self.base_predictor.predict(np.array(numbers_list))
        precise_pred = self.precise_predictor.predict(np.array(numbers_list))
        smart_pred = self.smart_predictor.predict(np.array(numbers_list))
        
        # 投票
        votes = defaultdict(float)
        for rank, n in enumerate(base_pred):
            votes[n] += (15 - rank) / 15.0
        for rank, n in enumerate(precise_pred):
            votes[n] += (15 - rank) / 15.0
        for rank, n in enumerate(smart_pred):
            votes[n] += (15 - rank) / 15.0
        
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        
        # 取前10个高票号码
        result = [n for n, _ in sorted_votes[:10]]
        
        # 剩余5个位置：反模式+区域保障
        blind = self._get_blind_candidates(numbers_list, exclude=set(result))
        for n in blind[:5]:
            if n not in result:
                result.append(n)
            if len(result) >= 15:
                break
        
        # 补齐
        for n, _ in sorted_votes:
            if n not in result:
                result.append(n)
            if len(result) >= 15:
                break
        
        # 区域保障
        result = self._ensure_zone_coverage(result, numbers_list)
        
        return result[:15]
    
    def _get_blind_candidates(self, numbers_list, exclude=None, count=5):
        """获取盲区候选号码"""
        if exclude is None:
            exclude = set()
        
        recent_5 = set(numbers_list[-5:])
        freq_30 = Counter(numbers_list[-30:])
        
        candidates = []
        
        # 从最近miss中学习
        for pred, actual in zip(self.recent_predictions[-8:], self.recent_actuals[-8:]):
            if actual not in pred:
                for offset in [-2, -1, 0, 1, 2]:
                    n = actual + offset
                    if 1 <= n <= 49 and n not in recent_5 and n not in exclude:
                        candidates.append((n, freq_30.get(n, 0)))
        
        # 去重排序
        seen = set()
        unique_candidates = []
        for n, score in candidates:
            if n not in seen:
                seen.add(n)
                unique_candidates.append((n, score))
        
        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in unique_candidates[:count]]


def run_test():
    print("=" * 80)
    print("最优防连败预测器 vs 基准 - 400期对比")
    print("=" * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    total = len(df)
    test_periods = 400
    start_idx = total - test_periods
    
    # 测试两个预测器
    predictors = {
        '基准(PreciseTop15)': PreciseTop15Predictor(),
        '最优防连败': AntiStreakOptimalPredictor()
    }
    
    all_results = {}
    
    for name, predictor in predictors.items():
        print(f"\n测试: {name}")
        hits = []
        
        for i in range(start_idx, total):
            train_data = numbers[:i]
            actual = int(numbers[i])
            
            if name == '基准(PreciseTop15)':
                top15 = predictor.predict(train_data)
                predictor.update_performance(top15, actual)
            else:
                top15 = predictor.predict(train_data)
                predictor.update(top15, actual)
            
            hits.append(actual in top15)
        
        # 统计
        streaks = []
        streak_len = 0
        streak_details = []  # (start_idx, length)
        streak_start = None
        
        for idx, h in enumerate(hits):
            if not h:
                if streak_start is None:
                    streak_start = idx
                streak_len += 1
            else:
                if streak_len > 0:
                    streaks.append(streak_len)
                    streak_details.append((streak_start, streak_len))
                streak_len = 0
                streak_start = None
        if streak_len > 0:
            streaks.append(streak_len)
            streak_details.append((streak_start, streak_len))
        
        hit_rate = sum(hits) / len(hits) * 100
        max_streak = max(streaks) if streaks else 0
        avg_streak = sum(streaks) / len(streaks) if streaks else 0
        
        all_results[name] = {
            'hits': hits,
            'hit_rate': hit_rate,
            'max_streak': max_streak,
            'avg_streak': avg_streak,
            'streaks': streaks,
            'streak_details': streak_details,
            'streak_ge3': sum(1 for s in streaks if s >= 3),
            'streak_ge5': sum(1 for s in streaks if s >= 5),
            'streak_ge7': sum(1 for s in streaks if s >= 7),
        }
        
        print(f"  命中率: {hit_rate:.1f}%")
        print(f"  最长连败: {max_streak}")
        print(f"  ≥3期连败: {sum(1 for s in streaks if s >= 3)}次")
        print(f"  ≥5期连败: {sum(1 for s in streaks if s >= 5)}次")
        print(f"  ≥7期连败: {sum(1 for s in streaks if s >= 7)}次")
    
    # 对比
    print(f"\n{'=' * 80}")
    print(f"对比总结")
    print(f"{'=' * 80}")
    print(f"{'指标':<16} {'基准':>12} {'最优防连败':>12} {'改善':>12}")
    print("-" * 55)
    
    b = all_results['基准(PreciseTop15)']
    o = all_results['最优防连败']
    
    print(f"{'命中率':<16} {b['hit_rate']:>11.1f}% {o['hit_rate']:>11.1f}% {o['hit_rate']-b['hit_rate']:>+11.1f}%")
    print(f"{'最长连败':<16} {b['max_streak']:>12} {o['max_streak']:>12} {o['max_streak']-b['max_streak']:>+12}")
    print(f"{'平均连败':<16} {b['avg_streak']:>11.2f} {o['avg_streak']:>11.2f} {o['avg_streak']-b['avg_streak']:>+11.2f}")
    print(f"{'≥3期连败':<16} {b['streak_ge3']:>12} {o['streak_ge3']:>12} {o['streak_ge3']-b['streak_ge3']:>+12}")
    print(f"{'≥5期连败':<16} {b['streak_ge5']:>12} {o['streak_ge5']:>12} {o['streak_ge5']-b['streak_ge5']:>+12}")
    print(f"{'≥7期连败':<16} {b['streak_ge7']:>12} {o['streak_ge7']:>12} {o['streak_ge7']-b['streak_ge7']:>+12}")
    
    # 连续不中分布对比
    print(f"\n{'=' * 80}")
    print(f"连续不中分布对比")
    print(f"{'=' * 80}")
    print(f"{'长度':<6} {'基准':>8} {'最优防连败':>10}")
    print("-" * 30)
    
    all_lengths = set(b['streaks'] + o['streaks'])
    b_dist = Counter(b['streaks'])
    o_dist = Counter(o['streaks'])
    
    for length in sorted(all_lengths):
        print(f"{length:>3}期   {b_dist.get(length, 0):>6}次  {o_dist.get(length, 0):>8}次")
    
    # ≥5期的详细对比
    print(f"\n{'=' * 80}")
    print(f"最优防连败 - 所有≥5期连续不中段")
    print(f"{'=' * 80}")
    for start, length in sorted(o['streak_details'], key=lambda x: x[1], reverse=True):
        if length >= 5:
            end = start + length - 1
            start_date = df.iloc[start_idx + start]['date']
            end_date = df.iloc[start_idx + end]['date']
            miss_nums = [int(numbers[start_idx + j]) for j in range(start, start + length)]
            print(f"  连续{length}期 ({start_date}~{end_date}): {miss_nums}")


if __name__ == '__main__':
    run_test()
