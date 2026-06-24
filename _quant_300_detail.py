# 量化预测器 - 最近300期预测详情
from quantitative_predictor import HistoryData, compute_statistics, score_all_numbers, auto_tune_rules, WEIGHT

hd = HistoryData('data/lucky_numbers.csv')
draws = hd.draws
test_periods = 300
start_idx = len(draws) - test_periods

hits5 = hits10 = hits15 = 0
rows = []

for i in range(start_idx, len(draws)):
    hist = draws[:i]
    if len(hist) < 30:
        continue
    stats = compute_statistics(hist)
    rules = auto_tune_rules(hist, stats)
    scores = score_all_numbers(stats, rules)

    actual = draws[i].number
    top15 = sorted(scores, key=scores.get, reverse=True)[:15]
    top10 = top15[:10]
    top5  = top15[:5]

    h5  = actual in top5
    h10 = actual in top10
    h15 = actual in top15
    if h5:  hits5  += 1
    if h10: hits10 += 1
    if h15: hits15 += 1

    rank = top15.index(actual) + 1 if h15 else '-'
    mark = 'HIT' if h15 else '---'
    rows.append((i - start_idx + 1, draws[i].date, actual, draws[i].animal,
                 top5, rank, mark))

print(f"{'期号':>4}  {'日期':<12} {'实际':>4} {'生肖':<4} {'TOP5预测':<30} {'排名':>4}  {'命中'}")
print('-' * 75)
for seq, date, actual, animal, top5, rank, mark in rows:
    top5_str = str(top5)
    print(f"{seq:>4}  {str(date):<12} {actual:>4}  {animal:<4} {top5_str:<30} {str(rank):>4}  {mark}")

total = len(rows)
print('\n' + '=' * 75)
print(f"共 {total} 期  TOP5: {hits5}/{total} = {hits5/total*100:.1f}%  "
      f"TOP10: {hits10}/{total} = {hits10/total*100:.1f}%  "
      f"TOP15: {hits15}/{total} = {hits15/total*100:.1f}%")
print(f"随机基准 TOP5: 10.2%  TOP10: 20.4%  TOP15: 30.6%")
