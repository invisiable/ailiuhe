"""投注策略深度优化 - 探索最优组合"""
import pandas as pd
import numpy as np
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

BASE_BET = 15
WIN_REWARD = 47
FIB = [1,1,2,3,5,8,13,21,34,55,89]

# ====== 核心发现: max=10→13/15 大幅提升, 延迟Fib极低回撤 ======
# 现在组合: 延迟Fib + Markov调整 + 提高max + 停1

class OptimizedStrategy:
    """延迟Fib + 马尔可夫动态调整 + 可调max"""
    def __init__(self, delay=3, max_mul=13, pattern_len=2, 
                 boost=2.5, reduce=0.4, high_thresh=0.35, low_thresh=0.25,
                 pause_after_hit=1):
        self.delay = delay
        self.max_mul = max_mul
        self.pattern_len = pattern_len
        self.boost = boost
        self.reduce = reduce
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.pause_after_hit = pause_after_hit
        
    def simulate(self, hit_seq):
        pattern_stats = {}
        recent = []
        fib_idx = 0
        balance = 0
        min_balance = 0
        max_drawdown = 0
        total_bet = 0
        total_win = 0
        pause_remaining = 0
        active = 0
        max_streak_loss = 0
        streak_loss = 0
        
        for h in hit_seq:
            if pause_remaining > 0:
                pause_remaining -= 1
                continue
            
            active += 1
            
            # 延迟Fib基础倍数
            if fib_idx <= self.delay - 1:
                base = 1
            else:
                fi = fib_idx - self.delay + 1
                base = min(FIB[min(fi, len(FIB)-1)], self.max_mul)
            
            # 马尔可夫调整
            mul = base
            if len(recent) >= self.pattern_len:
                pattern = tuple(recent[-self.pattern_len:])
                stats = pattern_stats.get(pattern, [0, 0])
                if stats[1] >= 1:
                    pred_rate = stats[0] / stats[1]
                    if pred_rate >= self.high_thresh:
                        mul = round(base * self.boost)
                    elif pred_rate <= self.low_thresh:
                        mul = round(base * self.reduce)
            mul = min(max(1, mul), self.max_mul)
            
            bet = BASE_BET * mul
            total_bet += bet
            
            if h:
                win = WIN_REWARD * mul
                total_win += win
                balance += (win - bet)
                fib_idx = 0
                streak_loss = 0
                if self.pause_after_hit > 0:
                    pause_remaining = self.pause_after_hit
            else:
                balance -= bet
                fib_idx += 1
                streak_loss += bet
                max_streak_loss = max(max_streak_loss, streak_loss)
            
            if balance < min_balance:
                min_balance = balance
                max_drawdown = abs(min_balance)
            
            # 更新马尔可夫统计
            if len(recent) >= self.pattern_len:
                pattern = tuple(recent[-self.pattern_len:])
                if pattern not in pattern_stats:
                    pattern_stats[pattern] = [0, 0]
                pattern_stats[pattern][1] += 1
                if h:
                    pattern_stats[pattern][0] += 1
            recent.append(1 if h else 0)
            if len(recent) > 12:
                recent.pop(0)
        
        profit = total_win - total_bet
        roi = profit / total_bet * 100 if total_bet > 0 else 0
        risk_ratio = profit / max_drawdown if max_drawdown > 0 else float('inf')
        
        return {
            'profit': profit, 'max_drawdown': max_drawdown,
            'roi': roi, 'risk_ratio': risk_ratio,
            'total_bet': total_bet, 'active': active,
            'max_streak_loss': max_streak_loss
        }

# ====== 测试1: 延迟期数 × max倍数 × 暂停期 (不含Markov) ======
print('='*90)
print('测试1: 延迟Fib + max倍数 + 暂停期 (无Markov)')
print('='*90)
print(f'{"延迟":>4} {"max":>4} {"停":>2} | {"利润":>7} {"回撤":>7} {"ROI%":>7} {"风险比":>8} {"连败最大损":>10}')
print('-'*70)

best_risk = None
best_roi = None
best_profit = None

for delay in [2, 3, 4, 5]:
    for max_mul in [8, 10, 13, 15, 20]:
        for pause in [0, 1]:
            s = OptimizedStrategy(delay=delay, max_mul=max_mul, 
                                  boost=1.0, reduce=1.0,  # 禁用Markov
                                  pause_after_hit=pause)
            r = s.simulate(hit_sequence)
            label = f'{delay:>4} {max_mul:>4} {pause:>2}'
            print(f'{label} | {r["profit"]:>7.0f} {r["max_drawdown"]:>7.0f} {r["roi"]:>6.1f}% {r["risk_ratio"]:>8.2f} {r["max_streak_loss"]:>10.0f}')
            
            if best_risk is None or r['risk_ratio'] > best_risk['risk_ratio']:
                best_risk = {**r, 'delay': delay, 'max': max_mul, 'pause': pause}
            if best_roi is None or r['roi'] > best_roi['roi']:
                best_roi = {**r, 'delay': delay, 'max': max_mul, 'pause': pause}
            if best_profit is None or r['profit'] > best_profit['profit']:
                best_profit = {**r, 'delay': delay, 'max': max_mul, 'pause': pause}

print()
print(f'最佳风险比: delay={best_risk["delay"]}, max={best_risk["max"]}, pause={best_risk["pause"]}')
print(f'  风险比={best_risk["risk_ratio"]:.2f}, 利润={best_risk["profit"]:.0f}, 回撤={best_risk["max_drawdown"]:.0f}')
print(f'最佳ROI: delay={best_roi["delay"]}, max={best_roi["max"]}, pause={best_roi["pause"]}')
print(f'  ROI={best_roi["roi"]:.1f}%, 利润={best_roi["profit"]:.0f}')
print(f'最佳利润: delay={best_profit["delay"]}, max={best_profit["max"]}, pause={best_profit["pause"]}')
print(f'  利润={best_profit["profit"]:.0f}, 回撤={best_profit["max_drawdown"]:.0f}')

# ====== 测试2: 在最优延迟基础上加Markov ======
print()
print('='*90)
print('测试2: 延迟Fib + Markov调整 (在最佳配置上叠加)')
print('='*90)

# 先用delay=3, max=13-20的配置测试Markov效果
for delay in [3, 4]:
    for max_mul in [13, 15, 20]:
        for pause in [1]:
            # 无Markov基准
            s_base = OptimizedStrategy(delay=delay, max_mul=max_mul, 
                                        boost=1.0, reduce=1.0, pause_after_hit=pause)
            r_base = s_base.simulate(hit_sequence)
            
            # 有Markov
            s_mk = OptimizedStrategy(delay=delay, max_mul=max_mul,
                                      boost=2.5, reduce=0.4, 
                                      high_thresh=0.35, low_thresh=0.25,
                                      pause_after_hit=pause)
            r_mk = s_mk.simulate(hit_sequence)
            
            print(f'\ndelay={delay}, max={max_mul}, pause={pause}:')
            print(f'  无Markov: 利润={r_base["profit"]:.0f}, 回撤={r_base["max_drawdown"]:.0f}, ROI={r_base["roi"]:.1f}%, 风险比={r_base["risk_ratio"]:.2f}')
            print(f'  有Markov: 利润={r_mk["profit"]:.0f}, 回撤={r_mk["max_drawdown"]:.0f}, ROI={r_mk["roi"]:.1f}%, 风险比={r_mk["risk_ratio"]:.2f}')
            diff_profit = r_mk["profit"] - r_base["profit"]
            diff_dd = r_mk["max_drawdown"] - r_base["max_drawdown"]
            print(f'  差异: 利润{diff_profit:+.0f}, 回撤{diff_dd:+.0f}')

# ====== 测试3: Markov参数微调 ======
print()
print('='*90)
print('测试3: Markov参数网格搜索 (delay=3, max=15, 停1)')
print('='*90)
print(f'{"boost":>5} {"reduce":>6} {"high":>5} {"low":>5} | {"利润":>7} {"回撤":>7} {"ROI%":>7} {"风险比":>8}')
print('-'*65)

best_combo = None
for boost in [1.5, 2.0, 2.5, 3.0]:
    for reduce in [0.3, 0.4, 0.5, 0.6]:
        for high_t in [0.30, 0.35, 0.40]:
            for low_t in [0.20, 0.25, 0.30]:
                if low_t >= high_t:
                    continue
                s = OptimizedStrategy(delay=3, max_mul=15,
                                      boost=boost, reduce=reduce,
                                      high_thresh=high_t, low_thresh=low_t,
                                      pause_after_hit=1)
                r = s.simulate(hit_sequence)
                
                if best_combo is None or r['risk_ratio'] > best_combo['risk_ratio']:
                    best_combo = {**r, 'boost': boost, 'reduce': reduce, 'high': high_t, 'low': low_t}

print(f'最佳Markov参数: boost={best_combo["boost"]}, reduce={best_combo["reduce"]}, high={best_combo["high"]}, low={best_combo["low"]}')
print(f'  利润={best_combo["profit"]:.0f}, 回撤={best_combo["max_drawdown"]:.0f}, ROI={best_combo["roi"]:.1f}%, 风险比={best_combo["risk_ratio"]:.2f}')

# 也找ROI最优
best_roi_mk = None
for boost in [1.5, 2.0, 2.5, 3.0]:
    for reduce in [0.3, 0.4, 0.5, 0.6]:
        for high_t in [0.30, 0.35, 0.40]:
            for low_t in [0.20, 0.25, 0.30]:
                if low_t >= high_t:
                    continue
                s = OptimizedStrategy(delay=3, max_mul=15,
                                      boost=boost, reduce=reduce,
                                      high_thresh=high_t, low_thresh=low_t,
                                      pause_after_hit=1)
                r = s.simulate(hit_sequence)
                if best_roi_mk is None or r['roi'] > best_roi_mk['roi']:
                    best_roi_mk = {**r, 'boost': boost, 'reduce': reduce, 'high': high_t, 'low': low_t}

print(f'最佳ROI参数: boost={best_roi_mk["boost"]}, reduce={best_roi_mk["reduce"]}, high={best_roi_mk["high"]}, low={best_roi_mk["low"]}')
print(f'  利润={best_roi_mk["profit"]:.0f}, 回撤={best_roi_mk["max_drawdown"]:.0f}, ROI={best_roi_mk["roi"]:.1f}%, 风险比={best_roi_mk["risk_ratio"]:.2f}')

# ====== 最终对比: 当前v6.1 vs 优化方案 ======
print()
print('='*90)
print('最终对比: 当前v6.1 vs 优化方案')
print('='*90)

# 当前v6.1 (Fib + Markov, delay=0, max=10, 停1)
current = OptimizedStrategy(delay=1, max_mul=10, boost=2.5, reduce=0.4,
                             high_thresh=0.35, low_thresh=0.25, pause_after_hit=1)
r_current = current.simulate(hit_sequence)

# 优化方案A: 延迟Fib-3 + max15 + 停1 (无Markov, 极低回撤)
opt_a = OptimizedStrategy(delay=3, max_mul=15, boost=1.0, reduce=1.0, pause_after_hit=1)
r_a = opt_a.simulate(hit_sequence)

# 优化方案B: 延迟Fib-3 + max15 + Markov + 停1
opt_b = OptimizedStrategy(delay=3, max_mul=15, 
                           boost=best_combo['boost'], reduce=best_combo['reduce'],
                           high_thresh=best_combo['high'], low_thresh=best_combo['low'],
                           pause_after_hit=1)
r_b = opt_b.simulate(hit_sequence)

# 优化方案C: 延迟Fib-4 + max20 + 停1 (超低回撤)
opt_c = OptimizedStrategy(delay=4, max_mul=20, boost=1.0, reduce=1.0, pause_after_hit=1)
r_c = opt_c.simulate(hit_sequence)

# 优化方案D: 延迟Fib-3 + max20 + 停1 (利润优先)
opt_d = OptimizedStrategy(delay=3, max_mul=20, boost=1.0, reduce=1.0, pause_after_hit=1)
r_d = opt_d.simulate(hit_sequence)

# 当前v6.1无停1
current_no_pause = OptimizedStrategy(delay=1, max_mul=10, boost=2.5, reduce=0.4,
                                      high_thresh=0.35, low_thresh=0.25, pause_after_hit=0)
r_cnp = current_no_pause.simulate(hit_sequence)

configs = [
    ('当前v6.1(无停)', r_cnp),
    ('当前v6.1(停1)', r_current),
    ('A:延迟3+max15+停1', r_a),
    ('B:延迟3+max15+MK+停1', r_b),
    ('C:延迟4+max20+停1', r_c),
    ('D:延迟3+max20+停1', r_d),
]

print(f'{"方案":<26} {"利润":>7} {"回撤":>7} {"ROI%":>7} {"风险比":>8} {"连败最大损":>10} {"投入":>7}')
print('-'*85)
for name, r in configs:
    print(f'{name:<26} {r["profit"]:>7.0f} {r["max_drawdown"]:>7.0f} {r["roi"]:>6.1f}% {r["risk_ratio"]:>8.2f} {r["max_streak_loss"]:>10.0f} {r["total_bet"]:>7.0f}')

print()
print('='*90)
print('结论与建议')
print('='*90)
