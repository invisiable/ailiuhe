"""
动态倍投策略优化 - 最大10倍限制
目标：在max=10的约束下，找到最优的动态倍投策略，兼顾高收益和低回撤

策略对比：
1. 纯斐波那契 Fib(max=10)
2. 激进斐波那契 AggressiveFib(连败>=4时跳级)
3. 自适应倍投 AdaptiveFib(根据近期命中率动态调整)
4. 连败加速 StreakBoost(连败越多增长越快，但封顶10)
5. 马尔可夫倍投 MarkovBet(根据命中模式预测下期，调整倍数)
6. 回撤感知 DrawdownAware(回撤大时降倍，回撤小时加倍)
7. 分段倍投 SegmentBet(前3败温和，4-6败中等，7+败激进)
"""
import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor
from collections import Counter

# === 加载数据 ===
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS
PREDICT_K = 15
TRAIN_WINDOW = 25
BASE_COST = 15
WIN_REWARD = 47
MAX_MUL = 10

# === 生成命中序列 ===
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

hits_total = sum(hit_sequence)
print(f"命中序列: {hits_total}/{TEST_PERIODS} = {hits_total/TEST_PERIODS*100:.1f}%")

# === 分析命中模式 ===
# 连败分布
streaks = []
current_streak = 0
for h in hit_sequence:
    if not h:
        current_streak += 1
    else:
        if current_streak > 0:
            streaks.append(current_streak)
        current_streak = 0
if current_streak > 0:
    streaks.append(current_streak)

streak_dist = Counter(streaks)
print(f"\n连败分布:")
for k in sorted(streak_dist.keys()):
    print(f"  连败{k}期: {streak_dist[k]}次 ({streak_dist[k]/len(streaks)*100:.1f}%)")

# 连败后的命中概率
print(f"\n连败N期后下期命中概率:")
for n in range(1, 10):
    count = 0
    hit_after = 0
    curr = 0
    for i, h in enumerate(hit_sequence):
        if not h:
            curr += 1
        else:
            if curr == n:
                count += 1
                # 不需要看下一期，当前就是命中
            curr = 0
    # 重新计算: 在连败恰好N期后(第N+1期)命中的情况
    curr = 0
    count2 = 0
    hit_after2 = 0
    for i in range(len(hit_sequence)):
        if not hit_sequence[i]:
            curr += 1
        else:
            if curr >= n:
                hit_after2 += 1
            if curr > 0:
                count2 += 1
            curr = 0
    # 更精确: 连败>=N期后，在第N+1期及之后命中
    # 计算条件概率: P(命中|已连败>=N期)
    states = []
    curr = 0
    for i in range(len(hit_sequence)):
        if not hit_sequence[i]:
            curr += 1
            states.append(('miss', curr))
        else:
            states.append(('hit', curr))
            curr = 0
    
    # 已连败>=N期时，下期命中的概率
    after_n_misses = 0
    after_n_total = 0
    cm = 0
    for i in range(len(hit_sequence)-1):
        if not hit_sequence[i]:
            cm += 1
        else:
            cm = 0
        if cm >= n:
            after_n_total += 1
            if hit_sequence[i+1]:
                after_n_misses += 1
    if after_n_total > 0:
        print(f"  连败>={n}期后: 命中{after_n_misses}/{after_n_total} = {after_n_misses/after_n_total*100:.1f}%")

# === 策略类定义 ===
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

class BaseStrategy:
    """基类"""
    def __init__(self, name, max_mul=MAX_MUL):
        self.name = name
        self.max_mul = max_mul
        self.reset()
    
    def reset(self):
        self.fib_idx = 0
        self.balance = 0
        self.total_bet = 0
        self.total_win = 0
        self.min_balance = 0
        self.streak_loss = 0
        self.max_streak_loss = 0
        self.consecutive_miss = 0
        self.max_consecutive_miss = 0
        self.hit_max_count = 0
        self.recent_hits = []
    
    def get_multiplier(self):
        """子类重写此方法"""
        return min(FIB[min(self.fib_idx, len(FIB)-1)], self.max_mul)
    
    def process(self, hit):
        mul = self.get_multiplier()
        bet = BASE_COST * mul
        self.total_bet += bet
        if mul >= self.max_mul:
            self.hit_max_count += 1
        
        if hit:
            win = WIN_REWARD * mul
            self.total_win += win
            self.balance += (win - bet)
            self.on_hit(mul)
            self.streak_loss = 0
            self.consecutive_miss = 0
        else:
            self.balance -= bet
            self.on_miss(mul)
            self.streak_loss += bet
            self.consecutive_miss += 1
            if self.streak_loss > self.max_streak_loss:
                self.max_streak_loss = self.streak_loss
            if self.consecutive_miss > self.max_consecutive_miss:
                self.max_consecutive_miss = self.consecutive_miss
        
        if self.balance < self.min_balance:
            self.min_balance = self.balance
        
        self.recent_hits.append(1 if hit else 0)
        if len(self.recent_hits) > 20:
            self.recent_hits.pop(0)
        
        return mul, bet
    
    def on_hit(self, mul):
        self.fib_idx = 0
    
    def on_miss(self, mul):
        self.fib_idx += 1
    
    def get_recent_rate(self, n=12):
        if not self.recent_hits:
            return 0.36
        window = self.recent_hits[-n:]
        return sum(window) / len(window) if window else 0.36


class Strategy1_PureFib(BaseStrategy):
    """1. 纯斐波那契(max=10) - 基准"""
    def __init__(self):
        super().__init__("纯Fib(max=10)")


class Strategy2_AggressiveFib(BaseStrategy):
    """2. 激进斐波那契 - 连败>=4时跳级加速"""
    def __init__(self):
        super().__init__("激进Fib(连败4跳级)")
    
    def get_multiplier(self):
        if self.consecutive_miss >= 4:
            # 连败4期后跳级: idx+1
            idx = min(self.fib_idx + 1, len(FIB)-1)
        else:
            idx = min(self.fib_idx, len(FIB)-1)
        return min(FIB[idx], self.max_mul)


class Strategy3_AdaptiveFib(BaseStrategy):
    """3. 自适应倍投 - 热区加倍，冷区保守"""
    def __init__(self):
        super().__init__("自适应(热区1.5x)")
    
    def get_multiplier(self):
        base = FIB[min(self.fib_idx, len(FIB)-1)]
        rate = self.get_recent_rate(12)
        if rate >= 0.42:  # 热区(命中率>42%)
            mul = base * 1.5
        elif rate <= 0.25:  # 冷区(命中率<25%)
            mul = base * 0.7
        else:
            mul = base
        return min(max(1, round(mul)), self.max_mul)


class Strategy4_StreakBoost(BaseStrategy):
    """4. 连败加速 - 连败序列映射到更高倍数"""
    def __init__(self):
        super().__init__("连败加速(阶梯)")
        # 连败映射: 连败N期→倍数
        # 设计思路: 连败越多命中概率越高(赌徒心理有统计支撑)
        self.streak_map = {
            0: 1, 1: 1, 2: 2, 3: 3, 
            4: 5, 5: 7, 6: 8, 7: 10, 8: 10, 9: 10
        }
    
    def get_multiplier(self):
        n = min(self.consecutive_miss, 9)
        return min(self.streak_map[n], self.max_mul)


class Strategy5_MarkovBet(BaseStrategy):
    """5. 马尔可夫倍投 - 根据最近3期模式预测"""
    def __init__(self):
        super().__init__("马尔可夫(模式识别)")
        self.pattern_stats = {}  # {(h,h,m): [hit_count, total]}
    
    def get_multiplier(self):
        base = FIB[min(self.fib_idx, len(FIB)-1)]
        # 获取最近3期模式
        if len(self.recent_hits) >= 3:
            pattern = tuple(self.recent_hits[-3:])
            stats = self.pattern_stats.get(pattern, [0, 0])
            if stats[1] >= 3:  # 有足够样本
                pred_rate = stats[0] / stats[1]
                if pred_rate >= 0.5:  # 高概率命中
                    mul = base * 1.5
                elif pred_rate <= 0.25:  # 低概率命中
                    mul = base * 0.7
                else:
                    mul = base
            else:
                mul = base
        else:
            mul = base
        return min(max(1, round(mul)), self.max_mul)
    
    def process(self, hit):
        # 更新模式统计(在处理前)
        if len(self.recent_hits) >= 3:
            pattern = tuple(self.recent_hits[-3:])
            if pattern not in self.pattern_stats:
                self.pattern_stats[pattern] = [0, 0]
            self.pattern_stats[pattern][1] += 1
            if hit:
                self.pattern_stats[pattern][0] += 1
        return super().process(hit)


class Strategy6_DrawdownAware(BaseStrategy):
    """6. 回撤感知 - 回撤大时降倍保本，回撤小时正常"""
    def __init__(self):
        super().__init__("回撤感知(自保)")
    
    def get_multiplier(self):
        base = FIB[min(self.fib_idx, len(FIB)-1)]
        # 当前回撤深度
        current_drawdown = self.balance - self.min_balance if self.balance > self.min_balance else 0
        # 如果当前余额接近历史最低(回撤深度<50元)，降低倍数
        if self.balance < 0 and abs(self.balance) > 300:
            mul = max(1, base // 2)  # 深度回撤，倍数减半
        elif self.balance < 0 and abs(self.balance) > 150:
            mul = max(1, int(base * 0.7))  # 中度回撤
        else:
            mul = base  # 正常
        return min(mul, self.max_mul)


class Strategy7_SegmentBet(BaseStrategy):
    """7. 分段倍投 - 1-3败温和, 4-6败中等, 7+败激进"""
    def __init__(self):
        super().__init__("分段(3段加速)")
    
    def get_multiplier(self):
        n = self.consecutive_miss
        if n <= 2:
            # 温和期: 1, 1, 2
            mul = [1, 1, 2][n]
        elif n <= 5:
            # 中等期: 3, 5, 7
            mul = [3, 5, 7][n-3]
        else:
            # 激进期: 直接10
            mul = 10
        return min(mul, self.max_mul)


class Strategy8_ConservativeStart(BaseStrategy):
    """8. 保守启动 - 命中后从0.5倍开始(即1倍)，连败后正常Fib"""
    def __init__(self):
        super().__init__("保守启动(低起)")
        self._just_hit = False
    
    def get_multiplier(self):
        if self._just_hit and self.consecutive_miss == 0:
            return 1  # 命中后第一期用1倍
        base = FIB[min(self.fib_idx, len(FIB)-1)]
        return min(base, self.max_mul)
    
    def on_hit(self, mul):
        self.fib_idx = 0
        self._just_hit = True
    
    def on_miss(self, mul):
        self.fib_idx += 1
        self._just_hit = False


class Strategy9_WeightedFib(BaseStrategy):
    """9. 加权Fib - 连败6+期时权重增加(基于SKILL规则)"""
    def __init__(self):
        super().__init__("加权Fib(6期+加权)")
    
    def get_multiplier(self):
        base = FIB[min(self.fib_idx, len(FIB)-1)]
        if self.consecutive_miss >= 6:
            # SKILL规则: 连续6期没命中通常到8期才命中，加权
            weight = 1.3 + (self.consecutive_miss - 6) * 0.15
            mul = round(base * weight)
        elif self.consecutive_miss >= 4:
            mul = round(base * 1.15)
        else:
            mul = base
        return min(max(1, mul), self.max_mul)


class Strategy10_HybridDynamic(BaseStrategy):
    """10. 混合动态 - 结合连败加速+自适应+回撤感知"""
    def __init__(self):
        super().__init__("混合动态(三合一)")
    
    def get_multiplier(self):
        base = FIB[min(self.fib_idx, len(FIB)-1)]
        
        # 因子1: 连败加速 (连败越多越激进)
        streak_factor = 1.0
        if self.consecutive_miss >= 6:
            streak_factor = 1.4
        elif self.consecutive_miss >= 4:
            streak_factor = 1.2
        
        # 因子2: 自适应 (热区加倍)
        rate = self.get_recent_rate(12)
        adapt_factor = 1.0
        if rate >= 0.42:
            adapt_factor = 1.2
        elif rate <= 0.22:
            adapt_factor = 0.8
        
        # 因子3: 回撤保护
        dd_factor = 1.0
        if self.balance < -300:
            dd_factor = 0.7
        elif self.balance < -150:
            dd_factor = 0.85
        
        mul = round(base * streak_factor * adapt_factor * dd_factor)
        return min(max(1, mul), self.max_mul)


# === 模拟函数 ===
def simulate(strategy, hit_seq, pause_mode=0):
    """
    模拟策略表现
    pause_mode: 0=不暂停, 1=命中1停1
    """
    strategy.reset()
    pause_remaining = 0
    bet_periods = 0
    skip_periods = 0
    
    for hit in hit_seq:
        if pause_remaining > 0:
            pause_remaining -= 1
            skip_periods += 1
            continue
        
        bet_periods += 1
        mul, bet = strategy.process(hit)
        
        if hit and pause_mode >= 1:
            pause_remaining = pause_mode
    
    roi = (strategy.total_win - strategy.total_bet) / strategy.total_bet * 100 if strategy.total_bet > 0 else 0
    profit = strategy.balance
    drawdown = abs(strategy.min_balance)
    risk_reward = profit / drawdown if drawdown > 0 else 0
    
    return {
        'name': strategy.name,
        'bet_periods': bet_periods,
        'skip_periods': skip_periods,
        'total_bet': strategy.total_bet,
        'total_win': strategy.total_win,
        'profit': profit,
        'roi': roi,
        'drawdown': drawdown,
        'max_streak_loss': strategy.max_streak_loss,
        'max_consecutive_miss': strategy.max_consecutive_miss,
        'hit_max_count': strategy.hit_max_count,
        'risk_reward': risk_reward
    }


# === 运行所有策略 ===
strategies = [
    Strategy1_PureFib,
    Strategy2_AggressiveFib,
    Strategy3_AdaptiveFib,
    Strategy4_StreakBoost,
    Strategy5_MarkovBet,
    Strategy6_DrawdownAware,
    Strategy7_SegmentBet,
    Strategy8_ConservativeStart,
    Strategy9_WeightedFib,
    Strategy10_HybridDynamic,
]

pause_modes = [0, 1]
pause_names = ['不暂停', '命中1停1']

print(f"\n{'='*120}")
print(f"动态倍投策略对比（MAX={MAX_MUL}倍, TOP{PREDICT_K}, 300期回测）")
print(f"{'='*120}")

all_results = []

for pause_mode in pause_modes:
    print(f"\n{'─'*120}")
    print(f"暂停模式: {pause_names[pause_mode]}")
    print(f"{'─'*120}")
    print(f"{'策略':<25} {'投注期':>6} {'净利润':>10} {'ROI':>8} {'回撤':>8} {'连挂额':>8} {'最长连败':>8} {'触顶次':>6} {'风险比':>8}")
    print(f"{'-'*100}")
    
    for StratClass in strategies:
        s = StratClass()
        r = simulate(s, hit_sequence, pause_mode)
        all_results.append({**r, 'pause': pause_names[pause_mode]})
        
        print(f"  {r['name']:<23} {r['bet_periods']:>5}期 {r['profit']:>+9.0f}元 {r['roi']:>7.1f}% "
              f"{r['drawdown']:>7.0f}元 {r['max_streak_loss']:>7.0f}元 {r['max_consecutive_miss']:>6}期 "
              f"{r['hit_max_count']:>5}次 {r['risk_reward']:>7.2f}")

# === 综合排名 ===
print(f"\n{'='*120}")
print(f"综合排名（按 风险收益比 排序）")
print(f"{'='*120}")
print(f"{'排名':<4} {'暂停模式':<10} {'策略':<25} {'净利润':>10} {'ROI':>8} {'回撤':>8} {'风险比':>8}")
print(f"{'-'*80}")

sorted_results = sorted(all_results, key=lambda x: x['risk_reward'], reverse=True)
for i, r in enumerate(sorted_results[:15], 1):
    marker = " ⭐" if i <= 3 else ""
    print(f"  {i:<3} {r['pause']:<10} {r['name']:<23} {r['profit']:>+9.0f}元 {r['roi']:>7.1f}% "
          f"{r['drawdown']:>7.0f}元 {r['risk_reward']:>7.2f}{marker}")

# === 按ROI排名 ===
print(f"\n{'='*120}")
print(f"综合排名（按 ROI 排序）")
print(f"{'='*120}")
print(f"{'排名':<4} {'暂停模式':<10} {'策略':<25} {'净利润':>10} {'ROI':>8} {'回撤':>8} {'风险比':>8}")
print(f"{'-'*80}")

sorted_by_roi = sorted(all_results, key=lambda x: x['roi'], reverse=True)
for i, r in enumerate(sorted_by_roi[:15], 1):
    marker = " ⭐" if i <= 3 else ""
    print(f"  {i:<3} {r['pause']:<10} {r['name']:<23} {r['profit']:>+9.0f}元 {r['roi']:>7.1f}% "
          f"{r['drawdown']:>7.0f}元 {r['risk_reward']:>7.2f}{marker}")

# === 对比 v5.1(max=20) vs 最优(max=10) ===
print(f"\n{'='*120}")
print(f"关键对比: v5.1(Fib max=20) vs 最优动态(max=10)")
print(f"{'='*120}")

# v5.1基准 (纯Fib max=20)
fib20 = BaseStrategy("Fib(max=20)", max_mul=20)
r20_base = simulate(fib20, hit_sequence, pause_mode=0)
fib20_p = BaseStrategy("Fib(max=20)+停1", max_mul=20)
r20_pause = simulate(fib20_p, hit_sequence, pause_mode=1)

# 最优max=10 (取风险比最高的)
best_rr = sorted_results[0]
best_roi_r = sorted_by_roi[0]

print(f"\n{'策略':<30} {'净利润':>10} {'ROI':>8} {'回撤':>8} {'连挂额':>8} {'触顶':>6} {'风险比':>8}")
print(f"{'-'*90}")
print(f"  {'v5.1 Fib20 基础':<28} {r20_base['profit']:>+9.0f}元 {r20_base['roi']:>7.1f}% "
      f"{r20_base['drawdown']:>7.0f}元 {r20_base['max_streak_loss']:>7.0f}元 {r20_base['hit_max_count']:>5}次 {r20_base['risk_reward']:>7.2f}")
print(f"  {'v5.1 Fib20 停1':<28} {r20_pause['profit']:>+9.0f}元 {r20_pause['roi']:>7.1f}% "
      f"{r20_pause['drawdown']:>7.0f}元 {r20_pause['max_streak_loss']:>7.0f}元 {r20_pause['hit_max_count']:>5}次 {r20_pause['risk_reward']:>7.2f}")
print(f"  {'最优风险比(max=10)':} {best_rr['pause']:<6} {best_rr['name']:<17} {best_rr['profit']:>+9.0f}元 {best_rr['roi']:>7.1f}% "
      f"{best_rr['drawdown']:>7.0f}元 {best_rr['max_streak_loss']:>7.0f}元 {best_rr['hit_max_count']:>5}次 {best_rr['risk_reward']:>7.2f}")
print(f"  {'最优ROI(max=10)':} {best_roi_r['pause']:<6} {best_roi_r['name']:<17} {best_roi_r['profit']:>+9.0f}元 {best_roi_r['roi']:>7.1f}% "
      f"{best_roi_r['drawdown']:>7.0f}元 {best_roi_r['max_streak_loss']:>7.0f}元 {best_roi_r['hit_max_count']:>5}次 {best_roi_r['risk_reward']:>7.2f}")

# === 详细分析最优策略的倍投分布 ===
print(f"\n{'='*120}")
print(f"最优策略倍投分布详情")
print(f"{'='*120}")

for idx in [0, 1, 2]:
    best = sorted_results[idx]
    # 重新模拟获取详细数据
    for StratClass in strategies:
        s = StratClass()
        if s.name == best['name']:
            s.reset()
            multipliers = []
            pause_remaining = 0
            pause_mode = 1 if best['pause'] == '命中1停1' else 0
            for hit in hit_sequence:
                if pause_remaining > 0:
                    pause_remaining -= 1
                    continue
                mul = s.get_multiplier()
                multipliers.append(mul)
                s.process(hit)
                if hit and pause_mode >= 1:
                    pause_remaining = 1
            
            mul_dist = Counter(multipliers)
            print(f"\n#{idx+1} {best['name']} ({best['pause']})")
            print(f"  倍数分布:")
            for m in sorted(mul_dist.keys()):
                bar = '█' * (mul_dist[m] // 2)
                print(f"    {m:>3}倍: {mul_dist[m]:>4}次 ({mul_dist[m]/len(multipliers)*100:>5.1f}%) {bar}")
            break

print(f"\n{'='*120}")
print("完成!")
