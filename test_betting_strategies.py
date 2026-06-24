"""投注策略全面对比分析"""
import pandas as pd
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values.tolist()
test_periods = min(400, len(df) - 50)
start_idx = len(df) - test_periods

# 生成命中序列
predictor = PreciseTop15Predictor()
hit_sequence = []
for i in range(start_idx, len(df)):
    hist = numbers[:i]
    actual = numbers[i]
    data = hist[max(0, len(hist)-25):]
    pred = predictor.predict(data, k=15)
    hit_sequence.append(1 if actual in pred else 0)

total_hits = sum(hit_sequence)
print(f'总期数: {len(hit_sequence)}, 命中: {total_hits}, 命中率: {total_hits/len(hit_sequence)*100:.2f}%')
print()

# 统计连败分布
streaks = []
current = 0
for h in hit_sequence:
    if h == 0:
        current += 1
    else:
        if current > 0:
            streaks.append(current)
        current = 0
if current > 0:
    streaks.append(current)
streak_dist = Counter(streaks)
print('连败分布:')
for k in sorted(streak_dist.keys()):
    print(f'  连败{k}期: {streak_dist[k]}次')
print(f'  最大连败: {max(streaks)}')
print(f'  平均连败: {np.mean(streaks):.2f}')
print()

# ====== 投注策略对比 ======
BASE_BET = 15
WIN_REWARD = 47
FIB = [1,1,2,3,5,8,13,21,34,55,89]

def simulate(hit_seq, get_multiplier, pause_after_hit=0, name=''):
    balance = 0
    min_balance = 0
    max_drawdown = 0
    total_bet = 0
    total_win = 0
    fib_idx = 0
    pause_remaining = 0
    results = []
    active_periods = 0
    
    for h in hit_seq:
        if pause_remaining > 0:
            pause_remaining -= 1
            continue
        
        active_periods += 1
        mul = get_multiplier(fib_idx, results)
        bet = BASE_BET * mul
        total_bet += bet
        
        if h:
            win = WIN_REWARD * mul
            total_win += win
            profit = win - bet
            balance += profit
            fib_idx = 0
            if pause_after_hit > 0:
                pause_remaining = pause_after_hit
        else:
            balance -= bet
            fib_idx += 1
        
        if balance < min_balance:
            min_balance = balance
            max_drawdown = abs(min_balance)
        
        results.append(h)
    
    roi = (total_win - total_bet) / total_bet * 100 if total_bet > 0 else 0
    risk_ratio = (total_win - total_bet) / max_drawdown if max_drawdown > 0 else float('inf')
    
    return {
        'name': name,
        'balance': balance,
        'max_drawdown': max_drawdown,
        'total_bet': total_bet,
        'total_win': total_win,
        'profit': total_win - total_bet,
        'roi': roi,
        'risk_ratio': risk_ratio,
        'active': active_periods
    }

# 策略定义
def fixed_mul(fib_idx, results):
    return 1

def fibonacci_mul(fib_idx, results):
    return min(FIB[min(fib_idx, len(FIB)-1)], 10)

def dalembert_mul(fib_idx, results):
    return min(1 + fib_idx, 10)

def martingale_mul(fib_idx, results):
    return min(2**fib_idx, 10)

def make_delayed_fib(delay):
    def fn(fib_idx, results):
        if fib_idx <= delay - 1:
            return 1
        return min(FIB[min(fib_idx - delay + 1, len(FIB)-1)], 10)
    return fn

# 马尔可夫策略（模拟GUI中的v6.1）
class MarkovSimulator:
    def __init__(self, pattern_len=2, boost=2.5, reduce=0.4, high_thresh=0.35, low_thresh=0.25):
        self.pattern_len = pattern_len
        self.boost = boost
        self.reduce = reduce
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.pattern_stats = {}
        self.recent = []
        self.fib_idx = 0
    
    def get_multiplier(self):
        base = min(FIB[min(self.fib_idx, len(FIB)-1)], 10)
        if len(self.recent) >= self.pattern_len:
            pattern = tuple(self.recent[-self.pattern_len:])
            stats = self.pattern_stats.get(pattern, [0, 0])
            if stats[1] >= 1:
                pred_rate = stats[0] / stats[1]
                if pred_rate >= self.high_thresh:
                    base = round(base * self.boost)
                elif pred_rate <= self.low_thresh:
                    base = round(base * self.reduce)
        return min(max(1, base), 10)
    
    def update(self, hit):
        pl = self.pattern_len
        if len(self.recent) >= pl:
            pattern = tuple(self.recent[-pl:])
            if pattern not in self.pattern_stats:
                self.pattern_stats[pattern] = [0, 0]
            self.pattern_stats[pattern][1] += 1
            if hit:
                self.pattern_stats[pattern][0] += 1
        self.recent.append(1 if hit else 0)
        if len(self.recent) > 12:
            self.recent.pop(0)
        if hit:
            self.fib_idx = 0
        else:
            self.fib_idx += 1

def simulate_markov(hit_seq, pause_after_hit=1, name=''):
    mk = MarkovSimulator()
    balance = 0
    min_balance = 0
    max_drawdown = 0
    total_bet = 0
    total_win = 0
    pause_remaining = 0
    active = 0
    
    for h in hit_seq:
        if pause_remaining > 0:
            pause_remaining -= 1
            continue
        active += 1
        mul = mk.get_multiplier()
        bet = BASE_BET * mul
        total_bet += bet
        if h:
            win = WIN_REWARD * mul
            total_win += win
            balance += (win - bet)
            if pause_after_hit > 0:
                pause_remaining = pause_after_hit
        else:
            balance -= bet
        if balance < min_balance:
            min_balance = balance
            max_drawdown = abs(min_balance)
        mk.update(h)
    
    roi = (total_win - total_bet) / total_bet * 100 if total_bet > 0 else 0
    risk_ratio = (total_win - total_bet) / max_drawdown if max_drawdown > 0 else float('inf')
    return {
        'name': name, 'balance': balance, 'max_drawdown': max_drawdown,
        'total_bet': total_bet, 'profit': total_win - total_bet,
        'roi': roi, 'risk_ratio': risk_ratio, 'active': active
    }

# ====== 新增策略: 区间倍投(根据连败分布的智能倍投) ======
def make_zone_fib(zones):
    """zones: [(连败范围, 倍数)] e.g. [(0,1), (1,1), (2,1), (3,2), (4,3), (5,5)]"""
    def fn(fib_idx, results):
        for max_idx, mul in reversed(zones):
            if fib_idx >= max_idx:
                return min(mul, 10)
        return 1
    return fn

# 自定义区间倍投: 根据连败分布，70%连败<=2期内
zone_conservative = make_zone_fib([(0,1),(1,1),(2,1),(3,2),(4,3),(5,5),(6,8),(7,10)])
zone_aggressive = make_zone_fib([(0,1),(1,1),(2,2),(3,3),(4,5),(5,8),(6,10)])
zone_balanced = make_zone_fib([(0,1),(1,1),(2,1),(3,1),(4,2),(5,3),(6,5),(7,8),(8,10)])

# ====== 新策略: 连胜后减注/停注 ======
def simulate_win_reduce(hit_seq, base_fn, reduce_after_wins=2, reduce_factor=0.5, name=''):
    """连胜N次后减注"""
    balance = 0
    min_balance = 0
    max_drawdown = 0
    total_bet = 0
    total_win = 0
    fib_idx = 0
    consecutive_wins = 0
    active = 0
    
    for h in hit_seq:
        active += 1
        mul = base_fn(fib_idx, [])
        
        # 连胜后减注
        if consecutive_wins >= reduce_after_wins:
            mul = max(1, round(mul * reduce_factor))
        
        bet = BASE_BET * mul
        total_bet += bet
        
        if h:
            win = WIN_REWARD * mul
            total_win += win
            balance += (win - bet)
            fib_idx = 0
            consecutive_wins += 1
        else:
            balance -= bet
            fib_idx += 1
            consecutive_wins = 0
        
        if balance < min_balance:
            min_balance = balance
            max_drawdown = abs(min_balance)
    
    roi = (total_win - total_bet) / total_bet * 100 if total_bet > 0 else 0
    risk_ratio = (total_win - total_bet) / max_drawdown if max_drawdown > 0 else float('inf')
    return {
        'name': name, 'balance': balance, 'max_drawdown': max_drawdown,
        'total_bet': total_bet, 'profit': total_win - total_bet,
        'roi': roi, 'risk_ratio': risk_ratio, 'active': active
    }

# ====== 执行所有策略 ======
strategies = [
    ('固定1倍', fixed_mul, 0),
    ('Fibonacci(max10)', fibonacci_mul, 0),
    ('达朗贝尔(max10)', dalembert_mul, 0),
    ('马丁格尔(max10)', martingale_mul, 0),
    ('延迟Fib-2(max10)', make_delayed_fib(2), 0),
    ('延迟Fib-3(max10)', make_delayed_fib(3), 0),
    ('延迟Fib-4(max10)', make_delayed_fib(4), 0),
    ('固定1倍+停1', fixed_mul, 1),
    ('Fib+停1', fibonacci_mul, 1),
    ('延迟Fib-2+停1', make_delayed_fib(2), 1),
    ('延迟Fib-3+停1', make_delayed_fib(3), 1),
    ('区间保守', zone_conservative, 0),
    ('区间激进', zone_aggressive, 0),
    ('区间平衡', zone_balanced, 0),
    ('区间保守+停1', zone_conservative, 1),
    ('区间平衡+停1', zone_balanced, 1),
]

results_all = []
print('='*95)
print(f'{"策略":<22} {"利润":>8} {"回撤":>8} {"ROI%":>8} {"风险比":>8} {"投入":>8} {"活跃期":>6}')
print('='*95)

for name, mul_fn, pause in strategies:
    r = simulate(hit_sequence, mul_fn, pause, name)
    results_all.append(r)
    print(f'{r["name"]:<22} {r["profit"]:>8.0f} {r["max_drawdown"]:>8.0f} {r["roi"]:>7.1f}% {r["risk_ratio"]:>8.2f} {r["total_bet"]:>8.0f} {r["active"]:>6}')

# 马尔可夫策略
for pause in [0, 1]:
    name = f'马尔可夫v6.1+停{pause}'
    r = simulate_markov(hit_sequence, pause, name)
    results_all.append(r)
    print(f'{r["name"]:<22} {r["profit"]:>8.0f} {r["max_drawdown"]:>8.0f} {r["roi"]:>7.1f}% {r["risk_ratio"]:>8.2f} {r["total_bet"]:>8.0f} {r["active"]:>6}')

# 连胜减注
for wins, factor in [(2, 0.5), (1, 0.5), (2, 0.3)]:
    name = f'Fib+胜{wins}减{factor}'
    r = simulate_win_reduce(hit_sequence, fibonacci_mul, wins, factor, name)
    results_all.append(r)
    print(f'{r["name"]:<22} {r["profit"]:>8.0f} {r["max_drawdown"]:>8.0f} {r["roi"]:>7.1f}% {r["risk_ratio"]:>8.2f} {r["total_bet"]:>8.0f} {r["active"]:>6}')

print('='*95)

# 排名
print()
print('>>> 按风险比排名 TOP5（利润/回撤）:')
by_risk = sorted(results_all, key=lambda x: -x['risk_ratio'])
for i, r in enumerate(by_risk[:5]):
    print(f'  #{i+1} {r["name"]}: 风险比={r["risk_ratio"]:.2f}, 利润={r["profit"]:.0f}, 回撤={r["max_drawdown"]:.0f}, ROI={r["roi"]:.1f}%')

print()
print('>>> 按ROI排名 TOP5:')
by_roi = sorted(results_all, key=lambda x: -x['roi'])
for i, r in enumerate(by_roi[:5]):
    print(f'  #{i+1} {r["name"]}: ROI={r["roi"]:.1f}%, 利润={r["profit"]:.0f}, 回撤={r["max_drawdown"]:.0f}')

print()
print('>>> 按绝对利润排名 TOP5:')
by_profit = sorted(results_all, key=lambda x: -x['profit'])
for i, r in enumerate(by_profit[:5]):
    print(f'  #{i+1} {r["name"]}: 利润={r["profit"]:.0f}, 回撤={r["max_drawdown"]:.0f}, ROI={r["roi"]:.1f}%')

# ====== 关键分析: 暂停策略的影响 ======
print()
print('='*70)
print('关键分析: 暂停跳过的那些期命中情况')
print('='*70)

pause_remaining = 0
skipped_hits = 0
skipped_misses = 0
for h in hit_sequence:
    if pause_remaining > 0:
        pause_remaining -= 1
        if h:
            skipped_hits += 1
        else:
            skipped_misses += 1
        continue
    if h:
        pause_remaining = 1

print(f'暂停期总数: {skipped_hits + skipped_misses}')
print(f'暂停期命中: {skipped_hits} ({skipped_hits/(skipped_hits+skipped_misses)*100:.1f}%)')
print(f'暂停期未中: {skipped_misses} ({skipped_misses/(skipped_hits+skipped_misses)*100:.1f}%)')
print(f'非暂停期数: {len(hit_sequence) - skipped_hits - skipped_misses}')
print()
print('分析: 暂停跳过的期中，如果命中率低于总体36.25%，则暂停策略有效')
pause_hr = skipped_hits/(skipped_hits+skipped_misses)*100
print(f'暂停期命中率: {pause_hr:.1f}% vs 总体: 36.25%')
if pause_hr < 36.25:
    print(f'  ✅ 暂停策略有效! 跳过了低命中期，节省了成本')
else:
    print(f'  ⚠️ 暂停策略跳过的期命中率偏高，可能不划算')

# ====== 新思路: 变长暂停 ======
print()
print('='*70)
print('变长暂停对比: 命中后停1/2/3期')
print('='*70)
for pause_len in [0, 1, 2, 3]:
    r = simulate(hit_sequence, fibonacci_mul, pause_len, f'Fib+停{pause_len}')
    print(f'  Fib+停{pause_len}: 利润={r["profit"]:.0f}, 回撤={r["max_drawdown"]:.0f}, ROI={r["roi"]:.1f}%, 风险比={r["risk_ratio"]:.2f}')

# ====== max倍数灵敏度 ======
print()
print('='*70)
print('最大倍数灵敏度: Fib+停1 (max=5/8/10/13/15)')
print('='*70)
for max_mul in [5, 8, 10, 13, 15, 20]:
    def make_fib_max(m):
        def fn(fib_idx, results):
            return min(FIB[min(fib_idx, len(FIB)-1)], m)
        return fn
    r = simulate(hit_sequence, make_fib_max(max_mul), 1, f'Fib(max{max_mul})+停1')
    print(f'  max={max_mul:>2}: 利润={r["profit"]:>6.0f}, 回撤={r["max_drawdown"]:>6.0f}, ROI={r["roi"]:>5.1f}%, 风险比={r["risk_ratio"]:.2f}')
