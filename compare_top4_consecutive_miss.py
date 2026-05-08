"""
对比分析: 原始TOP4 vs v2 TOP4 的连续失败分布
重点关注: 最大连续miss、miss段分布、切换策略效果
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from collections import Counter

# ===== 2026年生肖映射 =====
ZODIAC_CYCLE_2026 = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
NUM_TO_ZODIAC_2026 = {n: ZODIAC_CYCLE_2026[(n - 1) % 12] for n in range(1, 50)}

# ===== 加载数据 =====
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
animals = [NUM_TO_ZODIAC_2026[n] for n in numbers]
total = len(numbers)
test_periods = 300
start = total - test_periods

print(f"数据总期数: {total}, 测试区间: 第{start+1}~{total}期 (最近{test_periods}期)")
print("=" * 80)


def analyze_miss_streaks(hit_list):
    """分析连续miss段"""
    streaks = []
    current_miss = 0
    miss_start = -1
    for i, hit in enumerate(hit_list):
        if not hit:
            if current_miss == 0:
                miss_start = i
            current_miss += 1
        else:
            if current_miss > 0:
                streaks.append((miss_start, current_miss))
            current_miss = 0
    if current_miss > 0:
        streaks.append((miss_start, current_miss))
    return streaks


# ===== 模型1: v2 (静态组合 + 连miss切换) =====
from zodiac_top4_v2_predictor import ZodiacTop4V2Predictor

pred_v2 = ZodiacTop4V2Predictor()
v2_hits = []
v2_modes = []
for pi in range(test_periods):
    i = start + pi
    hist = numbers[:i]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    predicted = pred_v2.predict(hist, top_n=4)
    hit = actual_z in predicted
    v2_hits.append(hit)
    v2_modes.append("切换" if pred_v2.consecutive_miss >= pred_v2.miss_switch_threshold else "正常")
    pred_v2.record_result(hit)

# ===== 模型2: 原始推荐策略 =====
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

orig = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
orig_hits = []
orig_modes = []
for pi in range(test_periods):
    i = start + pi
    hist_animals = animals[:i]
    actual_z = animals[i]
    result = orig.predict_top4(hist_animals)
    hit = actual_z in result['top4']
    orig_hits.append(hit)
    orig_modes.append(result['model'])
    orig.update_performance(hit)
    orig.check_and_switch_model()

# ===== 对比分析 =====
def print_model_stats(name, hits_list, modes):
    total_hits = sum(hits_list)
    hit_rate = total_hits / len(hits_list) * 100
    streaks = analyze_miss_streaks(hits_list)
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  总命中: {total_hits}/{len(hits_list)} = {hit_rate:.1f}%")
    
    # 连续miss分布
    miss_lengths = [s[1] for s in streaks]
    if miss_lengths:
        print(f"  最大连续miss: {max(miss_lengths)}期")
        print(f"  连miss段数: {len(streaks)}")
        
        # miss段长度分布
        dist = Counter(miss_lengths)
        print(f"\n  连续miss长度分布:")
        for length in sorted(dist.keys()):
            bar = "█" * dist[length]
            print(f"    {length:>2}期miss: {dist[length]:>3}次 {bar}")
        
        # 超长miss (>=4)
        long = [s for s in streaks if s[1] >= 4]
        if long:
            print(f"\n  ≥4期连续miss ({len(long)}次):")
            for start_idx, length in sorted(long, key=lambda x: -x[1]):
                # 计算日期
                period = start_idx + 1
                end_period = start_idx + length
                print(f"    第{period:>3}-{end_period:>3}期: {length}期连miss")
        
        # >=6期的miss
        very_long = [s for s in streaks if s[1] >= 6]
        print(f"\n  ≥6期连续miss: {len(very_long)}次")
        if very_long:
            for start_idx, length in sorted(very_long, key=lambda x: -x[1]):
                print(f"    第{start_idx+1:>3}-{start_idx+length:>3}期: {length}期")
    
    # 分段命中率
    print(f"\n  分段统计(每50期):")
    for s in range(6):
        seg = hits_list[s*50:(s+1)*50]
        seg_hits = sum(seg)
        seg_rate = seg_hits / 50 * 100
        # miss streaks in this segment
        seg_streaks = analyze_miss_streaks(seg)
        seg_max_miss = max([s2[1] for s2 in seg_streaks]) if seg_streaks else 0
        bar = "█" * (seg_hits // 2)
        print(f"    {s*50+1:>3}-{(s+1)*50:>3}: {seg_hits}/50 = {seg_rate:>5.1f}% 最大miss={seg_max_miss:>2} {bar}")
    
    # 模式分布
    if modes:
        mode_counts = Counter(modes)
        print(f"\n  模式分布: {dict(mode_counts)}")


print_model_stats("v2 (静态组合+连miss切换)", v2_hits, v2_modes)
print_model_stats("原始 (重训练v2.0+应急备份)", orig_hits, orig_modes)


# ===== 关键对比 =====
print(f"\n{'='*60}")
print(f"  关键对比")
print(f"{'='*60}")

v2_streaks = analyze_miss_streaks(v2_hits)
orig_streaks = analyze_miss_streaks(orig_hits)

v2_miss_len = [s[1] for s in v2_streaks]
orig_miss_len = [s[1] for s in orig_streaks]

v2_long = sum(1 for l in v2_miss_len if l >= 4)
orig_long = sum(1 for l in orig_miss_len if l >= 4)

print(f"  {'指标':<20} {'v2':>10} {'原始':>10}")
print(f"  {'命中率':<20} {sum(v2_hits)/300*100:>9.1f}% {sum(orig_hits)/300*100:>9.1f}%")
print(f"  {'最大连miss':<20} {max(v2_miss_len):>9}期 {max(orig_miss_len):>9}期")
print(f"  {'≥4期miss次数':<20} {v2_long:>10} {orig_long:>10}")
print(f"  {'≥6期miss次数':<20} {sum(1 for l in v2_miss_len if l>=6):>10} {sum(1 for l in orig_miss_len if l>=6):>10}")
print(f"  {'平均miss段长':<20} {np.mean(v2_miss_len):>9.1f}期 {np.mean(orig_miss_len):>9.1f}期")

# ===== v2的切换模式是否真有效 =====
print(f"\n{'='*60}")
print(f"  v2切换模式效果分析")
print(f"{'='*60}")
normal_hits = sum(1 for h, m in zip(v2_hits, v2_modes) if m == "正常" and h)
normal_total = sum(1 for m in v2_modes if m == "正常")
switch_hits = sum(1 for h, m in zip(v2_hits, v2_modes) if m == "切换" and h)
switch_total = sum(1 for m in v2_modes if m == "切换")

print(f"  正常模式: {normal_hits}/{normal_total} = {normal_hits/normal_total*100:.1f}%")
print(f"  切换模式: {switch_hits}/{switch_total} = {switch_hits/switch_total*100:.1f}%")
print(f"  切换模式是否更好: {'是' if switch_total and switch_hits/switch_total > normal_hits/normal_total else '否'}")

# ===== 分析切换模式进入后的恢复情况 =====
print(f"\n  切换模式进入后几期内命中:")
switch_entries = []
for i, m in enumerate(v2_modes):
    if m == "切换" and (i == 0 or v2_modes[i-1] == "正常"):
        switch_entries.append(i)

for entry in switch_entries[:15]:  # 最多显示15次
    # 从entry开始，找到切换模式结束(或命中)的位置
    for j in range(entry, min(entry + 15, len(v2_hits))):
        if v2_hits[j]:
            recovery = j - entry
            print(f"    第{entry+1}期进入切换 → {recovery}期后命中(第{j+1}期)")
            break
    else:
        print(f"    第{entry+1}期进入切换 → 15期内未命中!")

# ===== 如果不使用切换机制呢？(纯静态) =====
print(f"\n{'='*60}")
print(f"  假设分析: 不用切换, 纯静态组合")
print(f"{'='*60}")

pred_static = ZodiacTop4V2Predictor()
static_hits = []
for pi in range(test_periods):
    i = start + pi
    hist = numbers[:i]
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in hist]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    # 强制用静态模式
    static = pred_static._static_predict(hist_animals)
    top4_idx = np.argsort(-static)[:4]
    top4 = [ZODIAC_CYCLE_2026[idx] for idx in top4_idx]
    hit = actual_z in top4
    static_hits.append(hit)

static_total = sum(static_hits)
static_streaks = analyze_miss_streaks(static_hits)
static_miss_len = [s[1] for s in static_streaks]

print(f"  纯静态命中率: {static_total}/{test_periods} = {static_total/test_periods*100:.1f}%")
print(f"  纯静态最大连miss: {max(static_miss_len)}期")
print(f"  纯静态≥6期miss: {sum(1 for l in static_miss_len if l >= 6)}次")

# ===== 纯MK150 =====
pred_mk = ZodiacTop4V2Predictor()
mk_hits = []
for pi in range(test_periods):
    i = start + pi
    hist = numbers[:i]
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in hist]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    mk = pred_mk._markov_scores(hist_animals, window=150)
    top4_idx = np.argsort(-mk)[:4]
    top4 = [ZODIAC_CYCLE_2026[idx] for idx in top4_idx]
    hit = actual_z in top4
    mk_hits.append(hit)

mk_total = sum(mk_hits)
mk_streaks = analyze_miss_streaks(mk_hits)
mk_miss_len = [s[1] for s in mk_streaks]

print(f"\n  纯MK150命中率: {mk_total}/{test_periods} = {mk_total/test_periods*100:.1f}%")
print(f"  纯MK150最大连miss: {max(mk_miss_len)}期")
print(f"  纯MK150≥6期miss: {sum(1 for l in mk_miss_len if l >= 6)}次")

# ===== 如果用TOP5会怎样 =====
print(f"\n{'='*60}")
print(f"  如果扩展到TOP5 / TOP4自适应扩展")
print(f"{'='*60}")

pred_expand = ZodiacTop4V2Predictor()
expand_hits = []
expand_sizes = []
for pi in range(test_periods):
    i = start + pi
    hist = numbers[:i]
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in hist]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    
    # 正常用TOP4，连miss>=3时扩展到TOP5
    static = pred_expand._static_predict(hist_animals)
    sorted_idx = np.argsort(-static)
    
    if pred_expand.consecutive_miss >= 3:
        top_n = 5  # 扩展到TOP5
    else:
        top_n = 4
    
    top = [ZODIAC_CYCLE_2026[idx] for idx in sorted_idx[:top_n]]
    hit = actual_z in top
    expand_hits.append(hit)
    expand_sizes.append(top_n)
    pred_expand.record_result(hit)

expand_total = sum(expand_hits)
expand_streaks = analyze_miss_streaks(expand_hits)
expand_miss_len = [s[1] for s in expand_streaks]
avg_size = np.mean(expand_sizes)

print(f"  自适应扩展: {expand_total}/{test_periods} = {expand_total/test_periods*100:.1f}%")
print(f"  最大连miss: {max(expand_miss_len)}期")
print(f"  ≥6期miss: {sum(1 for l in expand_miss_len if l >= 6)}次")
print(f"  平均投注生肖数: {avg_size:.1f}")

# ===== 融合策略: v2 + 原始模型投票 =====
print(f"\n{'='*60}")
print(f"  融合策略: v2静态 + 原始模型 联合投票")
print(f"{'='*60}")

pred_fusion = ZodiacTop4V2Predictor()
orig_fusion = RecommendedZodiacTop4Strategy(use_emergency_backup=False)
fusion_hits = []
for pi in range(test_periods):
    i = start + pi
    hist = numbers[:i]
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in hist]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    
    # v2静态得分
    static = pred_fusion._static_predict(hist_animals)
    
    # 原始模型预测
    orig_result = orig_fusion.predict_top4(hist_animals)
    orig_top4 = orig_result['top4']
    
    # 融合: v2得分 + 原始模型投票加分
    combined = static.copy()
    for z in orig_top4:
        zi = ZODIAC_CYCLE_2026.index(z)
        combined[zi] += 0.1  # 原始模型推荐的加分
    
    top4_idx = np.argsort(-combined)[:4]
    top4 = [ZODIAC_CYCLE_2026[idx] for idx in top4_idx]
    hit = actual_z in top4
    fusion_hits.append(hit)
    orig_fusion.update_performance(hit)

fusion_total = sum(fusion_hits)
fusion_streaks = analyze_miss_streaks(fusion_hits)
fusion_miss_len = [s[1] for s in fusion_streaks]

print(f"  融合命中率: {fusion_total}/{test_periods} = {fusion_total/test_periods*100:.1f}%")
print(f"  最大连miss: {max(fusion_miss_len)}期")
print(f"  ≥6期miss: {sum(1 for l in fusion_miss_len if l >= 6)}次")
print(f"  ≥4期miss: {sum(1 for l in fusion_miss_len if l >= 4)}次")
