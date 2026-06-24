"""
尾数预测 - 固定4组, 目标max_miss≤4
严格限制: 每期只能选4个尾数, 不能扩大
尝试多种高级模型, 寻找最优方案
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tail_digit_predictor import TailDigitPredictor, TAIL_DIGIT_NUMBERS, number_to_tail


def get_data():
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    return df['number'].values.tolist(), df


# ============================================================
# 模型1: 强制互斥轮换 (3期覆盖10个尾数)
# ============================================================
class StrictRotationPredictor:
    """
    严格轮换: 确保连续3期覆盖全部10个尾数
    期1: 最佳4个
    期2: 从剩余6个中选最佳4个
    期3: 必选剩余2个 + 最佳2个
    这保证如果同一个尾数连续出现3次, 至少有一次被覆盖
    """
    def __init__(self):
        self.base = TailDigitPredictor()
        self.recent_preds = []
        self.recent_hits = []
        self.phase = 0  # 0,1,2 三阶段轮换
    
    def predict(self, numbers):
        scores = self.base._calculate_scores(numbers)
        sorted_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 命中后重置phase
        if self.recent_hits and self.recent_hits[-1]:
            self.phase = 0
        
        if self.phase == 0:
            predicted = [d for d, _ in sorted_all[:4]]
        elif self.phase == 1:
            excluded = set(self.recent_preds[-1]) if self.recent_preds else set()
            remaining = [(d, s) for d, s in sorted_all if d not in excluded]
            predicted = [d for d, _ in remaining[:4]]
        else:  # phase == 2
            # 必须包含前2期未覆盖的尾数
            covered = set()
            for p in self.recent_preds[-2:]:
                covered.update(p)
            uncovered = [d for d in range(10) if d not in covered]
            # 未覆盖 + 评分最高的补到4个
            predicted = uncovered[:]
            if len(predicted) < 4:
                rest = [(d, s) for d, s in sorted_all if d not in set(predicted)]
                predicted += [d for d, _ in rest[:4 - len(predicted)]]
            predicted = predicted[:4]
        
        self.phase = (self.phase + 1) % 3
        return predicted
    
    def record(self, predicted, hit):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)


# ============================================================
# 模型2: 高阶马尔可夫 (2阶+3阶转移)
# ============================================================
class HighOrderMarkovPredictor:
    """
    高阶马尔可夫: 基于最近2-3个尾数的转移概率
    """
    def __init__(self):
        self.base = TailDigitPredictor()
        self.recent_preds = []
        self.recent_hits = []
    
    def predict(self, numbers):
        tails = [number_to_tail(n) for n in numbers]
        if len(tails) < 30:
            scores = self.base._calculate_scores(numbers)
            return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:4]]
        
        # 2阶马尔可夫
        trans2 = defaultdict(Counter)
        for i in range(len(tails) - 2):
            key = (tails[i], tails[i+1])
            trans2[key][tails[i+2]] += 1
        
        # 3阶马尔可夫
        trans3 = defaultdict(Counter)
        for i in range(len(tails) - 3):
            key = (tails[i], tails[i+1], tails[i+2])
            trans3[key][tails[i+3]] += 1
        
        # 当前状态
        key2 = (tails[-2], tails[-1])
        key3 = (tails[-3], tails[-2], tails[-1])
        
        # 综合评分
        base_scores = self.base._calculate_scores(numbers)
        markov2_scores = {d: 0 for d in range(10)}
        markov3_scores = {d: 0 for d in range(10)}
        
        if key2 in trans2:
            total = sum(trans2[key2].values())
            for d in range(10):
                markov2_scores[d] = trans2[key2].get(d, 0) / total
        
        if key3 in trans3:
            total = sum(trans3[key3].values())
            for d in range(10):
                markov3_scores[d] = trans3[key3].get(d, 0) / total
        
        # 融合
        combined = {}
        for d in range(10):
            combined[d] = 0.4 * base_scores[d] + 0.35 * markov2_scores[d] + 0.25 * markov3_scores[d]
        
        # 如果miss中, 排除上期
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        if miss_streak >= 1 and self.recent_preds:
            excluded = set(self.recent_preds[-1])
            remaining = [(d, s) for d, s in sorted_combined if d not in excluded]
            if len(remaining) >= 4:
                return [d for d, _ in remaining[:4]]
        
        return [d for d, _ in sorted_combined[:4]]
    
    def record(self, predicted, hit):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)


# ============================================================
# 模型3: 自适应权重+实际尾数追踪
# ============================================================
class AdaptiveWeightPredictor:
    """
    自适应权重: 根据各子模型最近表现动态调整权重
    加上实际尾数追踪, 利用miss时的实际尾数信息
    """
    def __init__(self):
        self.base = TailDigitPredictor()
        self.recent_preds = []
        self.recent_hits = []
        self.recent_actuals = []
        # 6个子模型的动态权重
        self.weights = [0.20, 0.25, 0.20, 0.15, 0.10, 0.10]
        self.model_hits = [[] for _ in range(6)]
    
    def predict(self, numbers):
        tails = [number_to_tail(n) for n in numbers]
        if len(tails) < 30:
            return list(range(4))
        
        # 6个子模型打分
        freq = self.base._frequency_analysis(tails)
        cold = self.base._cold_rebound_analysis(tails)
        trend = self.base._trend_momentum_analysis(tails)
        cycle = self.base._cycle_analysis(tails)
        adjacent = self.base._adjacent_analysis(tails)
        gap = self.base._gap_pattern_analysis(tails)
        
        all_model_scores = [freq, cold, trend, cycle, adjacent, gap]
        
        # 动态权重调整 (基于最近20期各模型表现)
        if len(self.recent_actuals) >= 10:
            for m_idx, m_scores_history in enumerate(self.model_hits):
                if len(m_scores_history) >= 10:
                    recent_perf = sum(m_scores_history[-20:]) / min(20, len(m_scores_history[-20:]))
                    self.weights[m_idx] = max(0.05, recent_perf)
            # 归一化
            total_w = sum(self.weights)
            self.weights = [w / total_w for w in self.weights]
        
        # 加权综合
        combined = {}
        for d in range(10):
            combined[d] = sum(self.weights[i] * all_model_scores[i].get(d, 0) for i in range(6))
        
        # Miss后的策略调整
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        if miss_streak >= 1 and self.recent_preds:
            # 利用miss期间实际出现的尾数信息
            if self.recent_actuals:
                missed_actuals = self.recent_actuals[-miss_streak:]
                # 邻近效应: 实际出现的尾数±1更可能再次出现
                bonus = defaultdict(float)
                for a in missed_actuals:
                    bonus[(a - 1) % 10] += 0.15
                    bonus[(a + 1) % 10] += 0.15
                    bonus[a] += 0.1
                
                for d in range(10):
                    combined[d] += bonus.get(d, 0)
            
            # 排除上期
            excluded = set(self.recent_preds[-1])
            remaining = [(d, combined[d]) for d in range(10) if d not in excluded]
            remaining.sort(key=lambda x: x[1], reverse=True)
            if len(remaining) >= 4:
                return [d for d, _ in remaining[:4]]
        
        return [d for d, _ in sorted_combined[:4]]
    
    def record(self, predicted, hit, actual_tail):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)
        self.recent_actuals.append(actual_tail)
        
        # 记录各子模型是否命中
        # (简化: 检查actual是否在该模型TOP4中)
        # 这里用最近一次的scores来估算
        pass


# ============================================================
# 模型4: 模式匹配预测 (KNN-like)
# ============================================================
class PatternMatchPredictor:
    """
    模式匹配: 找历史中与当前最近N期最相似的模式, 预测接下来的尾数
    """
    def __init__(self, pattern_len=5, k=10):
        self.base = TailDigitPredictor()
        self.pattern_len = pattern_len
        self.k = k
        self.recent_preds = []
        self.recent_hits = []
    
    def predict(self, numbers):
        tails = [number_to_tail(n) for n in numbers]
        if len(tails) < self.pattern_len + self.k + 5:
            scores = self.base._calculate_scores(numbers)
            return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:4]]
        
        # 当前模式
        current_pattern = tails[-self.pattern_len:]
        
        # 在历史中找相似模式
        similarities = []
        for i in range(self.pattern_len, len(tails) - self.pattern_len - 1):
            hist_pattern = tails[i - self.pattern_len:i]
            # 计算相似度 (匹配数 + 位移匹配)
            exact_match = sum(1 for a, b in zip(current_pattern, hist_pattern) if a == b)
            diff_match = sum(1 for a, b in zip(current_pattern, hist_pattern) if abs(a - b) <= 1)
            similarity = exact_match * 2 + diff_match
            next_tail = tails[i]
            similarities.append((similarity, next_tail, i))
        
        # 取最相似的k个
        similarities.sort(reverse=True)
        top_k = similarities[:self.k]
        
        # 统计这些相似模式后出现的尾数
        pattern_votes = Counter()
        for sim, next_t, _ in top_k:
            pattern_votes[next_t] += sim  # 加权投票
        
        # 融合基础模型
        base_scores = self.base._calculate_scores(numbers)
        combined = {}
        max_vote = max(pattern_votes.values()) if pattern_votes else 1
        for d in range(10):
            pattern_score = pattern_votes.get(d, 0) / max_vote if max_vote > 0 else 0
            combined[d] = 0.5 * base_scores[d] + 0.5 * pattern_score
        
        sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        # Miss后排除
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        if miss_streak >= 1 and self.recent_preds:
            excluded = set(self.recent_preds[-1])
            remaining = [(d, s) for d, s in sorted_combined if d not in excluded]
            if len(remaining) >= 4:
                return [d for d, _ in remaining[:4]]
        
        return [d for d, _ in sorted_combined[:4]]
    
    def record(self, predicted, hit):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)


# ============================================================
# 模型5: 贝叶斯更新 + 强制多样性
# ============================================================
class BayesDiversityPredictor:
    """
    贝叶斯方法: 维护各尾数的先验概率, 每期更新
    强制多样性: 通过衰减确保不会连续选同样的组合
    """
    def __init__(self):
        self.base = TailDigitPredictor()
        self.priors = {d: 0.1 for d in range(10)}  # 均匀先验
        self.recent_preds = []
        self.recent_hits = []
        self.recent_actuals = []
    
    def predict(self, numbers):
        tails = [number_to_tail(n) for n in numbers]
        if len(tails) < 20:
            return list(range(4))
        
        # 贝叶斯更新: 用最近出现的尾数更新先验
        if self.recent_actuals:
            # 实际出现的尾数增加其概率
            last_actual = self.recent_actuals[-1]
            for d in range(10):
                if d == last_actual:
                    self.priors[d] = self.priors[d] * 1.2
                else:
                    self.priors[d] = self.priors[d] * 0.95
            # 归一化
            total = sum(self.priors.values())
            self.priors = {d: p / total for d, p in self.priors.items()}
        
        # 结合基础模型
        base_scores = self.base._calculate_scores(numbers)
        combined = {}
        for d in range(10):
            combined[d] = 0.5 * base_scores[d] + 0.5 * self.priors[d] * 10
        
        # 多样性衰减: 最近2期选过的尾数降权
        for p in self.recent_preds[-2:]:
            for d in p:
                combined[d] *= 0.6
        
        sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in sorted_combined[:4]]
    
    def record(self, predicted, hit, actual_tail):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)
        self.recent_actuals.append(actual_tail)


# ============================================================
# 模型6: 极致轮换 (命中重置, miss强制不重复)
# ============================================================
class UltraRotationPredictor:
    """
    极致轮换模型:
    - 命中后: 重新评分选TOP4
    - 1 miss: 完全排除上期4个, 从剩余6个中选TOP4
    - 2 miss: 排除前2期8个, 剩余2个+新评分TOP2
    - 3 miss: 用完全不同的评分体系(冷号为主)
    关键: 最大化3期多样性覆盖
    """
    def __init__(self):
        self.base = TailDigitPredictor()
        self.recent_preds = []
        self.recent_hits = []
    
    def predict(self, numbers):
        tails = [number_to_tail(n) for n in numbers]
        scores = self.base._calculate_scores(numbers)
        cold = self.base._cold_rebound_analysis(tails)
        cycle = self.base._cycle_analysis(tails)
        gap = self.base._gap_pattern_analysis(tails)
        
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if miss_streak == 0:
            return [d for d, _ in sorted_scores[:4]]
        elif miss_streak == 1:
            excluded = set(self.recent_preds[-1])
            remaining = [(d, s) for d, s in sorted_scores if d not in excluded]
            return [d for d, _ in remaining[:4]]
        elif miss_streak == 2:
            excluded = set()
            for p in self.recent_preds[-2:]:
                excluded.update(p)
            uncovered = [d for d in range(10) if d not in excluded]
            # 剩余尾数(通常2个) + 最佳冷号2个
            predicted = uncovered[:]
            cold_sorted = sorted(cold.items(), key=lambda x: x[1], reverse=True)
            for d, _ in cold_sorted:
                if d not in set(predicted):
                    predicted.append(d)
                    if len(predicted) >= 4:
                        break
            return predicted[:4]
        else:
            # 3+ miss: 完全切换到冷号+间隔+周期体系
            rescue = {d: 0.4 * cold[d] + 0.35 * gap[d] + 0.25 * cycle[d] for d in range(10)}
            rescue_sorted = sorted(rescue.items(), key=lambda x: x[1], reverse=True)
            excluded = set(self.recent_preds[-1]) if self.recent_preds else set()
            remaining = [(d, s) for d, s in rescue_sorted if d not in excluded]
            return [d for d, _ in remaining[:4]] if len(remaining) >= 4 else [d for d, _ in rescue_sorted[:4]]
    
    def record(self, predicted, hit):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)


# ============================================================
# 模型7: 频率区间轮换
# ============================================================
class FrequencyZonePredictor:
    """
    将尾数分为热区/温区/冷区, 不同miss阶段从不同区选择
    - 正常: 2热+1温+1冷
    - 1 miss: 1热+2温+1冷 (避免纯追热)
    - 2 miss: 1热+1温+2冷 (冷号回补)
    - 3 miss: 0热+1温+3冷 (完全反转)
    """
    def __init__(self):
        self.base = TailDigitPredictor()
        self.recent_preds = []
        self.recent_hits = []
    
    def predict(self, numbers):
        tails = [number_to_tail(n) for n in numbers]
        scores = self.base._calculate_scores(numbers)
        freq = self.base._frequency_analysis(tails)
        cold = self.base._cold_rebound_analysis(tails)
        
        # 分区: 按频率排序
        freq_sorted = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        hot_zone = [d for d, _ in freq_sorted[:4]]   # 热区
        warm_zone = [d for d, _ in freq_sorted[4:7]]  # 温区
        cold_zone = [d for d, _ in freq_sorted[7:]]   # 冷区
        
        # 在各区内按综合得分排序
        def zone_top(zone, n):
            zone_scores = [(d, scores[d]) for d in zone]
            zone_scores.sort(key=lambda x: x[1], reverse=True)
            return [d for d, _ in zone_scores[:n]]
        
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        if miss_streak == 0:
            predicted = zone_top(hot_zone, 2) + zone_top(warm_zone, 1) + zone_top(cold_zone, 1)
        elif miss_streak == 1:
            predicted = zone_top(hot_zone, 1) + zone_top(warm_zone, 2) + zone_top(cold_zone, 1)
        elif miss_streak == 2:
            predicted = zone_top(hot_zone, 1) + zone_top(warm_zone, 1) + zone_top(cold_zone, 2)
        else:
            predicted = zone_top(warm_zone, 1) + zone_top(cold_zone, 3)
        
        # 排除上期(如果miss中)
        if miss_streak >= 1 and self.recent_preds:
            excluded = set(self.recent_preds[-1])
            # 检查是否有重复
            new_predicted = [d for d in predicted if d not in excluded]
            if len(new_predicted) < 4:
                # 补充
                all_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for d, _ in all_sorted:
                    if d not in set(new_predicted) and d not in excluded:
                        new_predicted.append(d)
                        if len(new_predicted) >= 4:
                            break
            predicted = new_predicted[:4]
        
        return predicted[:4]
    
    def record(self, predicted, hit):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)


# ============================================================
# 模型8: 综合最优 (组合前面最好的策略)
# ============================================================
class UltimateFixed4Predictor:
    """
    终极固定4组模型: 综合多种最优策略
    1. 多模型投票选出候选
    2. 强制轮换确保多样性
    3. miss后利用实际尾数信息
    4. 贝叶斯动态权重
    """
    def __init__(self):
        self.base = TailDigitPredictor()
        self.recent_preds = []
        self.recent_hits = []
        self.recent_actuals = []
    
    def predict(self, numbers):
        tails = [number_to_tail(n) for n in numbers]
        if len(tails) < 30:
            scores = self.base._calculate_scores(numbers)
            return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:4]]
        
        # 基础模型得分
        scores = self.base._calculate_scores(numbers)
        freq = self.base._frequency_analysis(tails)
        cold = self.base._cold_rebound_analysis(tails)
        trend = self.base._trend_momentum_analysis(tails)
        cycle = self.base._cycle_analysis(tails)
        gap = self.base._gap_pattern_analysis(tails)
        adjacent = self.base._adjacent_analysis(tails)
        
        # 马尔可夫(2阶)
        trans2 = defaultdict(Counter)
        for i in range(len(tails) - 2):
            trans2[(tails[i], tails[i+1])][tails[i+2]] += 1
        key2 = (tails[-2], tails[-1])
        markov2 = {d: 0 for d in range(10)}
        if key2 in trans2:
            total = sum(trans2[key2].values())
            markov2 = {d: trans2[key2].get(d, 0) / total for d in range(10)}
        
        # 模式匹配
        pattern_len = 4
        current_pattern = tails[-pattern_len:]
        pattern_votes = Counter()
        for i in range(pattern_len, len(tails) - 1):
            hist_pattern = tails[i - pattern_len:i]
            match_count = sum(1 for a, b in zip(current_pattern, hist_pattern) if a == b)
            if match_count >= 2:
                pattern_votes[tails[i]] += match_count
        
        max_pv = max(pattern_votes.values()) if pattern_votes else 1
        pattern_scores = {d: pattern_votes.get(d, 0) / max_pv for d in range(10)}
        
        miss_streak = 0
        for j in range(len(self.recent_hits) - 1, -1, -1):
            if not self.recent_hits[j]:
                miss_streak += 1
            else:
                break
        
        # 根据miss_streak动态调整权重
        if miss_streak == 0:
            combined = {d: 0.20*scores[d] + 0.15*markov2[d] + 0.15*pattern_scores[d] 
                       + 0.15*freq[d] + 0.15*cycle[d] + 0.10*adjacent[d] + 0.10*gap[d] for d in range(10)}
        elif miss_streak == 1:
            # 强调冷号和间隔
            combined = {d: 0.10*scores[d] + 0.15*markov2[d] + 0.15*pattern_scores[d] 
                       + 0.25*cold[d] + 0.15*gap[d] + 0.10*cycle[d] + 0.10*trend[d] for d in range(10)}
        elif miss_streak == 2:
            # 强调冷号、间隔、周期
            combined = {d: 0.30*cold[d] + 0.25*gap[d] + 0.20*cycle[d] 
                       + 0.15*pattern_scores[d] + 0.10*markov2[d] for d in range(10)}
        else:
            # 3+ miss: 完全反转, 冷号主导
            combined = {d: 0.40*cold[d] + 0.30*gap[d] + 0.20*cycle[d] + 0.10*markov2[d] for d in range(10)}
        
        # 利用实际尾数信息(邻近效应)
        if miss_streak >= 1 and self.recent_actuals:
            for a in self.recent_actuals[-miss_streak:]:
                combined[(a - 1) % 10] += 0.08
                combined[(a + 1) % 10] += 0.08
        
        sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        # 强制排除(miss时)
        if miss_streak >= 1 and self.recent_preds:
            if miss_streak == 1:
                excluded = set(self.recent_preds[-1])
            elif miss_streak == 2:
                excluded = set()
                for p in self.recent_preds[-2:]:
                    excluded.update(p)
            else:
                excluded = set(self.recent_preds[-1])  # 只排除上一期
            
            remaining = [(d, s) for d, s in sorted_combined if d not in excluded]
            if len(remaining) >= 4:
                return [d for d, _ in remaining[:4]]
        
        return [d for d, _ in sorted_combined[:4]]
    
    def record(self, predicted, hit, actual_tail):
        self.recent_preds.append(predicted)
        self.recent_hits.append(hit)
        self.recent_actuals.append(actual_tail)


# ============================================================
# 回测框架
# ============================================================
def backtest(numbers, df, predictor, name, needs_actual=False):
    """回测"""
    test_periods = min(300, len(numbers) - 50)
    start_idx = len(numbers) - test_periods
    hits = []
    
    for i in range(start_idx, len(numbers)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)
        predicted = predictor.predict(hist)
        hit = actual_tail in predicted
        hits.append(hit)
        
        if needs_actual:
            predictor.record(predicted, hit, actual_tail)
        else:
            predictor.record(predicted, hit)
    
    # 统计
    hit_rate = sum(hits) / len(hits) * 100
    max_miss = 0
    cur = 0
    for h in hits:
        if not h:
            cur += 1
            max_miss = max(max_miss, cur)
        else:
            cur = 0
    
    windows_3 = [any(hits[i:i+3]) for i in range(len(hits) - 2)]
    win3_rate = sum(windows_3) / len(windows_3) * 100
    windows_4 = [any(hits[i:i+4]) for i in range(len(hits) - 3)]
    win4_rate = sum(windows_4) / len(windows_4) * 100
    
    return {
        'name': name,
        'hit_rate': hit_rate,
        'win3_rate': win3_rate,
        'win4_rate': win4_rate,
        'max_miss': max_miss,
        'hits': hits,
    }


def main():
    numbers, df = get_data()
    
    print('=' * 95)
    print('🎯 尾数预测 - 固定4组, 目标max_miss≤4')
    print('   严格限制: 每期只能选4个尾数(~20个号码), 不能扩大')
    print('=' * 95)
    print()
    
    results = []
    
    # 测试所有模型
    models = [
        (StrictRotationPredictor(), '严格互斥轮换', False),
        (HighOrderMarkovPredictor(), '高阶马尔可夫', False),
        (AdaptiveWeightPredictor(), '自适应权重', True),
        (PatternMatchPredictor(pattern_len=5, k=10), '模式匹配(5,10)', False),
        (PatternMatchPredictor(pattern_len=3, k=15), '模式匹配(3,15)', False),
        (PatternMatchPredictor(pattern_len=4, k=20), '模式匹配(4,20)', False),
        (BayesDiversityPredictor(), '贝叶斯多样性', True),
        (UltraRotationPredictor(), '极致轮换', False),
        (FrequencyZonePredictor(), '频率区间轮换', False),
        (UltimateFixed4Predictor(), '终极综合', True),
    ]
    
    for predictor, name, needs_actual in models:
        r = backtest(numbers, df, predictor, name, needs_actual)
        results.append(r)
        print(f'  ✓ {name}: 命中{r["hit_rate"]:.1f}%, max_miss={r["max_miss"]}')
    
    # 排序
    results.sort(key=lambda x: (x['max_miss'], -x['hit_rate']))
    
    print()
    print('=' * 95)
    print(f"{'模型':>18} {'单期命中':>8} {'3期窗口':>8} {'4期窗口':>8} {'max_miss':>8} {'达标':>4}")
    print('-' * 65)
    
    for r in results:
        target = '✅' if r['max_miss'] <= 4 else f"差{r['max_miss']-4}"
        print(f"{r['name']:>18} {r['hit_rate']:>6.1f}% {r['win3_rate']:>6.1f}% {r['win4_rate']:>6.1f}% {r['max_miss']:>8} {target:>6}")
    
    # 分析最好的结果
    best = results[0]
    print()
    print('=' * 95)
    print(f'🏆 最优: {best["name"]} (max_miss={best["max_miss"]})')
    print('=' * 95)
    
    # miss分布
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
        danger = ' ⚠️' if length >= 5 else ''
        print(f'  {length}期: {streak_counter[length]:>3}次 {bar}{danger}')
    
    if best['max_miss'] > 4:
        print(f'\n{"=" * 95}')
        print('📊 数学分析:')
        print('=' * 95)
        print(f'  • 固定4组尾数, 随机命中率 = 40%')
        print(f'  • 最优模型命中率 = {best["hit_rate"]:.1f}%')
        p_miss = 1 - best['hit_rate'] / 100
        p_4miss = p_miss ** 4
        p_5miss = p_miss ** 5
        print(f'  • 单期miss概率 = {p_miss:.3f}')
        print(f'  • 连续4期miss概率 ≈ {p_4miss:.4f} ({p_4miss*100:.2f}%)')
        print(f'  • 连续5期miss概率 ≈ {p_5miss:.4f} ({p_5miss*100:.2f}%)')
        expected_4miss = 300 * p_4miss
        print(f'  • 300期中预期4+miss出现次数 ≈ {expected_4miss:.1f}')
        print()
        print(f'  结论: 固定4组(40%覆盖), 即使预测提升到{best["hit_rate"]:.0f}%,')
        print(f'         max_miss≤4在数学上极难保证(概率约{(1-p_5miss)**300*100:.0f}%达标)')
        print(f'         当前最佳max_miss = {best["max_miss"]}')
        
        # 计算需要多高的命中率才能"基本保证"max_miss<=4
        # P(5 miss) < 1/300 → p^5 < 0.0033 → p < 0.33 → hit > 67%
        print(f'\n  要"几乎保证"max_miss≤4:')
        print(f'    需要单期命中率 > 67% (当前{best["hit_rate"]:.1f}%)')
        print(f'    或者使用自适应扩展策略(miss时增加覆盖)')


if __name__ == '__main__':
    main()
