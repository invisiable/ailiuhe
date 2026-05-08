"""验证v6.0 马尔可夫动态倍投策略 - 模拟GUI实际行为"""
import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS

PREDICT_K = 15
TRAIN_WINDOW = 25
BASE_COST = 15
WIN_REWARD = 47
MAX_MUL = 10

# 生成命中序列
predictor = PreciseTop15Predictor()
hit_sequence = []
for i in range(start_idx, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train = numbers[lo:i]
    preds = predictor.predict(train, k=PREDICT_K)
    actual = int(numbers[i])
    hit = actual in preds
    predictor.update_performance(preds, actual)
    hit_sequence.append(hit)

hits = sum(hit_sequence)
print(f"命中率: {hits}/{TEST_PERIODS} = {hits/TEST_PERIODS*100:.1f}%")

# Fibonacci
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# 马尔可夫动态策略 (与GUI一致)
class MarkovStrategy:
    def __init__(self):
        self.fib_idx = 0
        self.balance = 0
        self.total_bet = 0
        self.total_win = 0
        self.min_balance = 0
        self.streak_loss = 0
        self.max_streak_loss = 0
        self.recent = []
        self.pattern_stats = {}
        self.hit_max = 0
        self.max_consec = 0
        self.consec = 0
    
    def get_base(self):
        return min(FIB[min(self.fib_idx, len(FIB)-1)], MAX_MUL)
    
    def get_markov_mul(self):
        base = self.get_base()
        if len(self.recent) >= 3:
            pat = tuple(self.recent[-3:])
            stats = self.pattern_stats.get(pat, [0, 0])
            if stats[1] >= 3:
                rate = stats[0] / stats[1]
                if rate >= 0.5:
                    mul = base * 1.5
                elif rate <= 0.25:
                    mul = base * 0.7
                else:
                    mul = base
            else:
                mul = base
        else:
            mul = base
        return min(max(1, round(mul)), MAX_MUL)
    
    def process(self, hit):
        # 更新模式统计(处理前)
        if len(self.recent) >= 3:
            pat = tuple(self.recent[-3:])
            if pat not in self.pattern_stats:
                self.pattern_stats[pat] = [0, 0]
            self.pattern_stats[pat][1] += 1
            if hit:
                self.pattern_stats[pat][0] += 1
        
        mul = self.get_markov_mul()
        bet = BASE_COST * mul
        self.total_bet += bet
        if mul >= MAX_MUL:
            self.hit_max += 1
        
        if hit:
            win = WIN_REWARD * mul
            self.total_win += win
            self.balance += (win - bet)
            self.fib_idx = 0
            self.streak_loss = 0
            self.consec = 0
        else:
            self.balance -= bet
            self.fib_idx += 1
            self.streak_loss += bet
            self.consec += 1
            if self.streak_loss > self.max_streak_loss:
                self.max_streak_loss = self.streak_loss
            if self.consec > self.max_consec:
                self.max_consec = self.consec
        
        if self.balance < self.min_balance:
            self.min_balance = self.balance
        
        self.recent.append(1 if hit else 0)
        if len(self.recent) > 12:
            self.recent.pop(0)
        
        return mul

# 基础策略(不暂停)
s_base = MarkovStrategy()
for h in hit_sequence:
    s_base.process(h)

roi_base = (s_base.total_win - s_base.total_bet) / s_base.total_bet * 100
dd_base = abs(s_base.min_balance)

# 暂停策略(命中1停1)
s_pause = MarkovStrategy()
pause_rem = 0
bet_periods = 0
skip_periods = 0
for h in hit_sequence:
    if pause_rem > 0:
        pause_rem -= 1
        skip_periods += 1
        continue
    bet_periods += 1
    s_pause.process(h)
    if h:
        pause_rem = 1

roi_pause = (s_pause.total_win - s_pause.total_bet) / s_pause.total_bet * 100
dd_pause = abs(s_pause.min_balance)
rr_pause = s_pause.balance / dd_pause if dd_pause > 0 else 0

print(f"\n{'='*60}")
print(f"v6.0 马尔可夫动态倍投(max=10) - 300期验证")
print(f"{'='*60}")

print(f"\n--- 基础策略（不暂停）---")
print(f"  净利润: {s_base.balance:.0f}元, ROI: {roi_base:.1f}%")
print(f"  总投入: {s_base.total_bet:.0f}元, 总奖金: {s_base.total_win:.0f}元")
print(f"  最大回撤: {dd_base:.0f}元, 连挂总额: {s_base.max_streak_loss:.0f}元")
print(f"  最长连败: {s_base.max_consec}期, 触顶({MAX_MUL}倍): {s_base.hit_max}次")
print(f"  风险收益比: {s_base.balance/dd_base:.2f}")

print(f"\n--- 暂停策略（命中1停1）---")
print(f"  投注期: {bet_periods}, 暂停期: {skip_periods}")
print(f"  净利润: {s_pause.balance:.0f}元, ROI: {roi_pause:.1f}%")
print(f"  总投入: {s_pause.total_bet:.0f}元, 总奖金: {s_pause.total_win:.0f}元")
print(f"  最大回撤: {dd_pause:.0f}元, 连挂总额: {s_pause.max_streak_loss:.0f}元")
print(f"  最长连败: {s_pause.max_consec}期, 触顶({MAX_MUL}倍): {s_pause.hit_max}次")
print(f"  风险收益比: {rr_pause:.2f}")

print(f"\n{'='*60}")
print(f"版本对比")
print(f"{'='*60}")
print(f"{'版本':<28} {'命中率':>8} {'ROI':>8} {'净利润':>8} {'回撤':>8} {'连挂额':>8} {'风险比':>8}")
print(f"{'-'*85}")
versions = [
    ("v4.0 纯Fib10 停1",        "35.7%", "21.7%", "1635", "779", "—", "2.10"),
    ("v5.0 TOP23 Fib10 停1",    "50.7%", "8.5%",  "537",  "883", "—", "0.61"),
    ("v5.1 纯Fib20 停1",        "36.0%", "29.9%", "2447", "450", "795", "5.44"),
    (f"v6.0 马尔可夫max10 基础",  f"{hits/TEST_PERIODS*100:.1f}%", f"{roi_base:.1f}%", f"{s_base.balance:.0f}", f"{dd_base:.0f}", f"{s_base.max_streak_loss:.0f}", f"{s_base.balance/dd_base:.2f}"),
    (f"v6.0 马尔可夫max10 停1",  f"{hits/TEST_PERIODS*100:.1f}%", f"{roi_pause:.1f}%", f"{s_pause.balance:.0f}", f"{dd_pause:.0f}", f"{s_pause.max_streak_loss:.0f}", f"{rr_pause:.2f}"),
]
for v in versions:
    print(f"  {v[0]:<26} {v[1]:>8} {v[2]:>8} {v[3]:>8} {v[4]:>8} {v[5]:>8} {v[6]:>8}")

# 马尔可夫模式统计
print(f"\n{'='*60}")
print(f"马尔可夫模式统计（暂停策略）")
print(f"{'='*60}")
patterns = sorted(s_pause.pattern_stats.items(), key=lambda x: x[1][1], reverse=True)
print(f"{'模式':<20} {'样本':>6} {'命中':>6} {'命中率':>8}")
for pat, (h, t) in patterns:
    pat_str = ''.join(['✓' if p else '✗' for p in pat])
    rate = h/t*100 if t > 0 else 0
    marker = ' ← 加倍' if rate >= 50 and t >= 3 else (' ← 减倍' if rate <= 25 and t >= 3 else '')
    print(f"  {pat_str:<18} {t:>6} {h:>6} {rate:>7.1f}%{marker}")
