import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor

PREDICT_K = 23
TRAIN_WINDOW = 25

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
test_periods = min(300, len(df) - TRAIN_WINDOW)
start_idx = len(df) - test_periods

predictor = PreciseTop15Predictor()
results = []
for i in range(start_idx, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train_data = numbers[lo:i]
    predictions = predictor.predict(train_data, k=PREDICT_K)
    actual = int(numbers[i])
    hit = actual in predictions
    results.append({'period': i-start_idx+1, 'date': df.iloc[i]['date'], 'actual': actual, 'predictions': predictions, 'hit': hit})

hits = sum(1 for r in results if r['hit'])
print(f"Total: {hits}/{len(results)} = {hits/len(results)*100:.1f}%")

for r in results[:5] + results[-5:]:
    hit_mark = 'Y' if r['hit'] else 'N'
    pred_sorted = sorted(r['predictions'])
    pred_display = []
    for n in pred_sorted:
        if n == r['actual']:
            pred_display.append(f'[{n}]')
        else:
            pred_display.append(str(n))
    print(f"#{r['period']:3d} {r['date']} actual={r['actual']:2d} {hit_mark} preds={','.join(pred_display)}")
