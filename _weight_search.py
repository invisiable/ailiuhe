from quantitative_predictor import HistoryData, compute_statistics, score_all_numbers, auto_tune_rules
import quantitative_predictor as qp

hd = HistoryData('data/lucky_numbers.csv')
draws = hd.draws
test_periods = 200
start_idx = len(draws) - test_periods

def test_weights(wm, wg, war, wf):
    hits15 = hits10 = 0
    for i in range(start_idx, len(draws)):
        hist = draws[:i]
        if len(hist) < 30: continue
        stats = compute_statistics(hist)
        rules = auto_tune_rules(hist, stats)
        actual = draws[i].number
        orig = dict(qp.WEIGHT)
        qp.WEIGHT.update({'miss':wm,'recency':-war,'freq':wf,'gap_cycle':wg,'tail':0,'zone':0})
        scores = score_all_numbers(stats, rules)
        qp.WEIGHT.update(orig)
        top15 = sorted(scores, key=scores.get, reverse=True)[:15]
        if actual in top15: hits15 += 1
        if actual in top15[:10]: hits10 += 1
    return hits15/test_periods*100, hits10/test_periods*100

results = []
for wm in [0.3,0.4,0.5,0.6]:
    for wg in [0.2,0.3,0.4,0.5]:
        for war in [0.0,0.1,0.2]:
            for wf in [0.0,0.1]:
                r15,r10 = test_weights(wm,wg,war,wf)
                results.append((r15,r10,wm,wg,war,wf))

results.sort(reverse=True)
print('TOP10结果:')
print('  TOP15%   TOP10%   miss   gap  anti_rec  freq')
for r in results[:10]:
    print(f'  {r[0]:5.1f}%  {r[1]:5.1f}%   {r[2]:.1f}  {r[3]:.1f}   {r[4]:.1f}     {r[5]:.1f}')
print('基准: 30.6% / 20.4%')
