# -*- coding: utf-8 -*-
"""
TOP15 马尔可夫链命中预测成功率验证脚本
滚动回测：每期只用前缀数据建模，预测该期→对比实际命中
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

# ─────────────────────────── 复制自 lucky_number_gui.py ───────────────────────
def _calculate_markov2_transition(hit_records):
    trans = {(a, b): [0, 0] for a in [False, True] for b in [False, True]}
    for i in range(2, len(hit_records)):
        st = (hit_records[i - 2], hit_records[i - 1])
        trans[st][1] += 1
        if hit_records[i]:
            trans[st][0] += 1
    return {st: (h, t, h / t if t > 0 else 0.0) for st, (h, t) in trans.items()}

def _analyze_losing_streak_recovery(hit_records, max_streak=10):
    streak_stats = {}
    for streak_len in range(1, max_streak + 1):
        recovery_hits = total_occurrences = 0
        for i in range(len(hit_records) - streak_len):
            if all(not hit_records[i + j] for j in range(streak_len)):
                if i + streak_len < len(hit_records):
                    total_occurrences += 1
                    if hit_records[i + streak_len]:
                        recovery_hits += 1
        if total_occurrences > 0:
            streak_stats[streak_len] = (recovery_hits, total_occurrences, recovery_hits / total_occurrences)
        else:
            streak_stats[streak_len] = (0, 0, 0.0)
    return streak_stats

def _calculate_winning_streak_continuation(hit_records):
    continuation_stats = {}
    i = 0
    while i < len(hit_records):
        if hit_records[i]:
            current_streak = 0
            while i < len(hit_records) and hit_records[i]:
                current_streak += 1
                i += 1
            for n in range(1, current_streak):
                if n not in continuation_stats:
                    continuation_stats[n] = [0, 0]
                continuation_stats[n][1] += 1
                continuation_stats[n][0] += 1
            if i < len(hit_records):
                n = current_streak
                if n not in continuation_stats:
                    continuation_stats[n] = [0, 0]
                continuation_stats[n][1] += 1
        else:
            i += 1
    return {n: (c, t, c / t) for n, (c, t) in continuation_stats.items() if t > 0}

def _predict_current_hit_probability(hit_records, position, losing_streak_stats, winning_continuation=None):
    base_n = position if position > 0 else len(hit_records)
    base_rate = sum(hit_records[:base_n]) / base_n if base_n > 0 else 0.5
    if position == 0:
        return base_rate
    cur_ls = cur_ws = 0
    for k in range(position - 1, -1, -1):
        if not hit_records[k]: cur_ls += 1
        else: break
    if cur_ls == 0:
        for k in range(position - 1, -1, -1):
            if hit_records[k]: cur_ws += 1
            else: break
    if cur_ls > 0 and losing_streak_stats:
        key = min(cur_ls, max(losing_streak_stats.keys()))
        if key in losing_streak_stats:
            _, sample, streak_prob = losing_streak_stats[key]
            weight = min(0.8, sample / 40.0)
            return weight * streak_prob + (1 - weight) * base_rate
    elif cur_ws > 0 and winning_continuation:
        key = min(cur_ws, max(winning_continuation.keys()))
        if key in winning_continuation:
            _, sample, cont_prob = winning_continuation[key]
            weight = min(0.8, sample / 40.0)
            return weight * cont_prob + (1 - weight) * base_rate
    return base_rate

def _predict_markov_hit_probability(hit_records, position, mk2_table, losing_streak_stats, winning_continuation=None):
    base_n = position if position > 0 else len(hit_records)
    base_rate = sum(hit_records[:base_n]) / base_n if base_n > 0 else 0.5
    if position < 2:
        return _predict_current_hit_probability(hit_records, position, losing_streak_stats, winning_continuation)
    st = (hit_records[position - 2], hit_records[position - 1])
    _h, t, mk_prob = mk2_table.get(st, (0, 0, base_rate))
    last8 = sum(hit_records[max(0, position - 8):position]) / min(8, position) if position > 0 else base_rate
    w_mk = min(0.60, t / 50.0)
    w_win = 0.20
    return w_mk * mk_prob + w_win * last8 + (1.0 - w_mk - w_win) * base_rate

# ─────────────────────────── 滚动回测 ────────────────────────────────────────
def rolling_backtest(df, test_periods=300, min_train=50):
    predictor = PreciseTop15Predictor()
    n = len(df)
    start = max(min_train, n - test_periods)

    hit_records = []   # 真实命中序列（全量，滚动追加）
    records = []       # 每期详情

    for i in range(start, n):
        train_nums = df.iloc[:i]['number'].values
        preds = predictor.predict(train_nums)
        actual = df.iloc[i]['number']
        hit = bool(actual in preds)

        # 用 i-start 作为 position（只看已知历史）
        pos = len(hit_records)   # 0-based，等于已积累的 hit_records 长度

        # 全量统计辅助（基于全部已知命中序列）
        mk2   = _calculate_markov2_transition(hit_records) if len(hit_records) >= 2 else {}
        ls    = _analyze_losing_streak_recovery(hit_records) if hit_records else {}
        wc    = _calculate_winning_streak_continuation(hit_records) if hit_records else {}

        prob_markov  = _predict_markov_hit_probability(hit_records, pos, mk2, ls, wc)
        prob_legacy  = _predict_current_hit_probability(hit_records, pos, ls, wc)

        records.append({
            'period': i - start + 1,
            'date':   df.iloc[i]['date'],
            'actual': actual,
            'hit':    hit,
            'prob_markov': prob_markov,
            'prob_legacy': prob_legacy,
        })
        hit_records.append(hit)

    return records

# ─────────────────────────── 统计分析 ────────────────────────────────────────
def eval_strategy(records, cond_fn, label, base_bet=15, win_reward=47):
    bets = hits = 0
    total = peak = maxdd = 0
    for r in records:
        if cond_fn(r):
            bets += 1
            if r['hit']:
                total += win_reward - base_bet; hits += 1
            else:
                total -= base_bet
            if total > peak: peak = total
            dd = peak - total
            if dd > maxdd: maxdd = dd
    if bets == 0:
        print(f"  {label:<36} 无触发")
        return
    hr  = hits / bets * 100
    roi = total / (bets * base_bet) * 100
    rr  = peak / maxdd if maxdd > 0 else float('inf')
    print(f"  {label:<36} {bets:>4}期  命中率{hr:>5.1f}%  盈亏{total:>+7.0f}元  ROI{roi:>+7.2f}%  回撤{maxdd:>5.0f}元  风险收益{rr:>5.2f}")

def markov_calibration(records):
    """分阈值校准：每个概率区间内，预测率 vs 实际命中率"""
    buckets = {}
    for r in records:
        p = r['prob_markov']
        b = round(p * 10) / 10   # 0.0 / 0.1 / … / 1.0
        if b not in buckets:
            buckets[b] = [0, 0]
        buckets[b][1] += 1
        if r['hit']:
            buckets[b][0] += 1
    print(f"\n  {'预测概率区间':<12} {'样本数':>6} {'实际命中率':>10} {'校准误差':>10}")
    print("  " + "-" * 45)
    for b in sorted(buckets):
        h, t = buckets[b]
        actual_hr = h / t * 100 if t > 0 else 0
        err = actual_hr - b * 100
        marker = " ←" if abs(err) > 10 else ""
        print(f"  {b*100:>5.0f}%附近        {t:>6}  {actual_hr:>9.1f}%  {err:>+8.1f}%{marker}")

def state_transition_accuracy(records):
    """按4个马尔可夫状态（全量建模）统计实际命中率"""
    # 重新用全量 hit_records 统计状态转移
    hit_records = [r['hit'] for r in records]
    mk2 = _calculate_markov2_transition(hit_records)
    names = {(False, False): '败-败', (True, False): '胜-败',
             (False, True):  '败-胜', (True, True):  '胜-胜'}

    # 按状态分桶，统计实际命中
    buckets = {k: [0, 0] for k in names}
    for i in range(2, len(records)):
        st = (records[i - 2]['hit'], records[i - 1]['hit'])
        buckets[st][1] += 1
        if records[i]['hit']:
            buckets[st][0] += 1

    print(f"\n  {'状态':<6} | {'马尔可夫预测':>10} | {'样本数':>5} | {'实际命中率':>10} | {'误差':>7}")
    print("  " + "-" * 52)
    for st in [(False, False), (True, False), (False, True), (True, True)]:
        h, t = buckets[st]
        _mh, _mt, mk_p = mk2.get(st, (0, 0, 0.0))
        actual_hr = h / t * 100 if t > 0 else 0
        err = actual_hr - mk_p * 100
        print(f"  {names[st]:<6} | {mk_p*100:>9.1f}% | {t:>5} | {actual_hr:>9.1f}% | {err:>+6.1f}%")

# ─────────────────────────── 主程序 ──────────────────────────────────────────
if __name__ == '__main__':
    DATA = 'data/lucky_numbers.csv'
    df   = pd.read_csv(DATA, encoding='utf-8-sig')
    print(f"\n数据加载：{len(df)} 期  ({df.iloc[0]['date']} ~ {df.iloc[-1]['date']})")

    TEST = min(300, len(df) - 50)
    print(f"滚动回测：最近 {TEST} 期（每期只用前缀数据建模）\n")
    print("  正在逐期预测，请稍候…")
    records = rolling_backtest(df, test_periods=TEST)
    total_hits = sum(r['hit'] for r in records)
    print(f"  实际命中：{total_hits}/{len(records)}  基准命中率：{total_hits/len(records)*100:.1f}%")

    # ── 策略过滤验证
    print(f"\n{'='*76}")
    print("三策略过滤验证（固投 15 元 / 命中返 47 元）")
    print(f"{'='*76}")
    eval_strategy(records, lambda r: True,                                    "基准：全部投注")
    eval_strategy(records, lambda r: r['prob_markov'] > 0.50,                 "策略B: 马尔可夫 >50%")
    eval_strategy(records, lambda r: r['prob_markov'] > 0.48,                 "策略B2: 马尔可夫 >48%")
    eval_strategy(records, lambda r: r['prob_legacy']  > 0.50,                "策略A: 原版综合加权 >50%")
    eval_strategy(records, lambda r: r['prob_markov'] > 0.50 and
                                     r['prob_legacy']  > 0.48,                "策略C: 双重确认")
    eval_strategy(records, lambda r: r['prob_markov'] > 0.45,                 "策略B3: 马尔可夫 >45%")

    # ── 马尔可夫校准分析
    print(f"\n{'='*76}")
    print("马尔可夫预测概率校准分析（预测 vs 实际）")
    print(f"{'='*76}")
    markov_calibration(records)

    # ── 4状态精细分析
    print(f"\n{'='*76}")
    print("4种马尔可夫状态实际命中率（全量建模）")
    print(f"{'='*76}")
    state_transition_accuracy(records)

    # ── 概率分布统计
    probs = [r['prob_markov'] for r in records]
    print(f"\n{'='*76}")
    print("马尔可夫预测概率分布统计")
    print(f"{'='*76}")
    print(f"  均值: {np.mean(probs)*100:.1f}%  中位数: {np.median(probs)*100:.1f}%  "
          f"标准差: {np.std(probs)*100:.1f}%")
    print(f"  >50%期数: {sum(p > 0.50 for p in probs)}期  "
          f">48%期数: {sum(p > 0.48 for p in probs)}期  "
          f">45%期数: {sum(p > 0.45 for p in probs)}期")

    # ── 连续正确/错误预测分析
    correct_calls = []
    for r in records:
        predicted_hit = r['prob_markov'] > 0.50
        actually_hit  = r['hit']
        correct_calls.append(predicted_hit == actually_hit)
    tp = sum(1 for r in records if r['prob_markov'] > 0.50 and r['hit'])
    fp = sum(1 for r in records if r['prob_markov'] > 0.50 and not r['hit'])
    tn = sum(1 for r in records if r['prob_markov'] <= 0.50 and not r['hit'])
    fn = sum(1 for r in records if r['prob_markov'] <= 0.50 and r['hit'])
    print(f"\n  阈值 0.50 混淆矩阵:")
    print(f"    TP(正确预测命中) ={tp:>4}  FP(错误预测命中) ={fp:>4}")
    print(f"    TN(正确预测未中) ={tn:>4}  FN(错误预测未中) ={fn:>4}")
    total_c = tp + fp + tn + fn
    acc = (tp + tn) / total_c * 100 if total_c > 0 else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    print(f"    准确率: {acc:.1f}%  精确率: {prec:.1f}%  召回率: {recall:.1f}%")

    print(f"\n✅ 验证完成！\n")
