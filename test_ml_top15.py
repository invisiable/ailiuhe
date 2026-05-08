"""ML方法: 为每个号码计算出现概率, 选Top15"""
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS

def build_features(numbers_seq, target_num, window=25):
    features = []
    labels = []
    for i in range(window, len(numbers_seq)):
        recent = numbers_seq[i-window:i]
        freq = Counter(recent)
        f = []
        f.append(freq.get(target_num, 0))
        f.append(freq.get(target_num, 0) / window)
        gap = window + 1
        for j in range(len(recent)-1, -1, -1):
            if recent[j] == target_num:
                gap = len(recent) - j
                break
        f.append(gap)
        f.append(1 if gap <= 10 else 0)
        f.append(1 if gap <= 5 else 0)
        zone = (target_num - 1) // 10
        zone_count = sum(1 for x in recent if (x-1)//10 == zone)
        f.append(zone_count)
        f.append(zone_count / window)
        recent5 = numbers_seq[i-5:i] if i >= 5 else numbers_seq[:i]
        f.append(1 if target_num in recent5 else 0)
        neighbor_active = sum(1 for x in recent if abs(x - target_num) <= 3 and x != target_num)
        f.append(neighbor_active)
        f.append(target_num % 2)
        f.append(recent[-1])
        f.append(abs(recent[-1] - target_num))
        features.append(f)
        labels.append(1 if numbers_seq[i] == target_num else 0)
    return np.array(features), np.array(labels)

def build_all_features(numbers_seq, i, window=25):
    recent = numbers_seq[max(0,i-window):i]
    freq = Counter(recent)
    recent5 = set(numbers_seq[max(0,i-5):i])
    recent3 = set(numbers_seq[max(0,i-3):i])
    features_all = []
    for target_num in range(1, 50):
        f = []
        f.append(freq.get(target_num, 0))
        f.append(freq.get(target_num, 0) / max(len(recent), 1))
        gap = len(recent) + 1
        for j in range(len(recent)-1, -1, -1):
            if recent[j] == target_num:
                gap = len(recent) - j
                break
        f.append(gap)
        f.append(1 if gap <= 10 else 0)
        f.append(1 if gap <= 5 else 0)
        zone = (target_num - 1) // 10
        zone_count = sum(1 for x in recent if (x-1)//10 == zone)
        f.append(zone_count)
        f.append(zone_count / max(len(recent), 1))
        f.append(1 if target_num in recent5 else 0)
        neighbor_active = sum(1 for x in recent if abs(x - target_num) <= 3 and x != target_num)
        f.append(neighbor_active)
        f.append(target_num % 2)
        f.append(recent[-1] if len(recent) > 0 else 25)
        f.append(abs((recent[-1] if len(recent) > 0 else 25) - target_num))
        features_all.append(f)
    return np.array(features_all)

print("=== 方法1: 统一二分类器 ===")
train_end = start_idx
WINDOW = 25
X_train = []
y_train = []

for i in range(WINDOW, train_end):
    recent = numbers[max(0,i-WINDOW):i]
    actual = numbers[i]
    freq = Counter(recent)
    recent5 = set(numbers[max(0,i-5):i])
    for target_num in range(1, 50):
        f = []
        f.append(freq.get(target_num, 0))
        f.append(freq.get(target_num, 0) / WINDOW)
        gap = WINDOW + 1
        for j in range(len(recent)-1, -1, -1):
            if recent[j] == target_num:
                gap = len(recent) - j
                break
        f.append(gap)
        f.append(1 if gap <= 10 else 0)
        f.append(1 if gap <= 5 else 0)
        zone = (target_num - 1) // 10
        zone_count = sum(1 for x in recent if (x-1)//10 == zone)
        f.append(zone_count)
        f.append(zone_count / WINDOW)
        f.append(1 if target_num in recent5 else 0)
        neighbor_active = sum(1 for x in recent if abs(x - target_num) <= 3 and x != target_num)
        f.append(neighbor_active)
        f.append(target_num % 2)
        f.append(recent[-1])
        f.append(abs(recent[-1] - target_num))
        X_train.append(f)
        y_train.append(1 if actual == target_num else 0)

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"训练样本: {len(X_train)} (正样本: {y_train.sum()}, 比例: {y_train.mean():.4f})")

models = {
    'GBM': GradientBoostingClassifier(n_estimators=100, max_depth=4, subsample=0.8, random_state=42),
    'RF': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced'),
    'LR': LogisticRegression(max_iter=1000, C=0.1),
}

for model_name, model in models.items():
    print(f"\n训练 {model_name}...")
    model.fit(X_train, y_train)
    hits = {15: 0, 18: 0, 20: 0, 22: 0}
    for i in range(start_idx, len(df)):
        features_all = build_all_features(numbers, i, WINDOW)
        probs = model.predict_proba(features_all)[:, 1]
        ranked = np.argsort(probs)[::-1]
        for k in hits.keys():
            top_k = [ranked[j] + 1 for j in range(k)]
            actual = numbers[i]
            if actual in top_k:
                hits[k] += 1
    for k, h in hits.items():
        rate = h / TEST_PERIODS * 100
        marker = " ⭐" if rate >= 50 else ""
        print(f"  {model_name} Top{k}: {h}/{TEST_PERIODS} = {rate:.1f}%{marker}")

print(f"\n=== 方法2: GBM滚动训练 (每50期重训) ===")
retrain_interval = 50
model = None
hits = {15: 0, 18: 0, 20: 0, 22: 0}

for i in range(start_idx, len(df)):
    period_num = i - start_idx
    if model is None or period_num % retrain_interval == 0:
        X_t = []
        y_t = []
        for t in range(WINDOW, i):
            recent = numbers[max(0,t-WINDOW):t]
            actual = numbers[t]
            freq = Counter(recent)
            recent5 = set(numbers[max(0,t-5):t])
            for target_num in range(1, 50):
                f = []
                f.append(freq.get(target_num, 0))
                f.append(freq.get(target_num, 0) / WINDOW)
                gap = WINDOW + 1
                for j in range(len(recent)-1, -1, -1):
                    if recent[j] == target_num:
                        gap = len(recent) - j
                        break
                f.append(gap)
                f.append(1 if gap <= 10 else 0)
                f.append(1 if gap <= 5 else 0)
                zone = (target_num - 1) // 10
                zone_count = sum(1 for x in recent if (x-1)//10 == zone)
                f.append(zone_count)
                f.append(zone_count / WINDOW)
                f.append(1 if target_num in recent5 else 0)
                neighbor_active = sum(1 for x in recent if abs(x - target_num) <= 3 and x != target_num)
                f.append(neighbor_active)
                f.append(target_num % 2)
                f.append(recent[-1])
                f.append(abs(recent[-1] - target_num))
                X_t.append(f)
                y_t.append(1 if actual == target_num else 0)
        X_t = np.array(X_t)
        y_t = np.array(y_t)
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, subsample=0.8, random_state=42)
        model.fit(X_t, y_t)
    features_all = build_all_features(numbers, i, WINDOW)
    probs = model.predict_proba(features_all)[:, 1]
    ranked = np.argsort(probs)[::-1]
    actual = numbers[i]
    for k in hits.keys():
        top_k = [ranked[j] + 1 for j in range(k)]
        if actual in top_k:
            hits[k] += 1

for k, h in hits.items():
    rate = h / TEST_PERIODS * 100
    marker = " ⭐" if rate >= 50 else ""
    print(f"  GBM滚动 Top{k}: {h}/{TEST_PERIODS} = {rate:.1f}%{marker}")

print(f"\n=== 方法3: GBM概率 + PreciseTop15融合 ===")
from precise_top15_predictor import PreciseTop15Predictor

static_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, subsample=0.8, random_state=42)
static_model.fit(X_train, y_train)

predictor = PreciseTop15Predictor()
for k in [15, 18, 20, 22]:
    hits = 0
    for i in range(start_idx, len(df)):
        features_all = build_all_features(numbers, i, WINDOW)
        probs = static_model.predict_proba(features_all)[:, 1]
        window25 = numbers[max(0, i-25):i]
        p15_preds = predictor.predict(window25)
        ml_scores = {}
        for n in range(1, 50):
            ml_scores[n] = probs[n-1]
        p15_scores = {}
        for rank, n in enumerate(p15_preds):
            p15_scores[n] = 1.0 - rank / 15
        ml_max = max(ml_scores.values())
        ml_min = min(ml_scores.values())
        ml_range = ml_max - ml_min if ml_max > ml_min else 1
        combined = {}
        for n in range(1, 50):
            ml_norm = (ml_scores[n] - ml_min) / ml_range
            p15 = p15_scores.get(n, 0)
            combined[n] = ml_norm * 0.5 + p15 * 0.5
        sorted_nums = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        preds = [n for n, _ in sorted_nums[:k]]
        actual = numbers[i]
        if actual in preds:
            hits += 1
    rate = hits / TEST_PERIODS * 100
    marker = " ⭐" if rate >= 50 else ""
    print(f"  融合 Top{k}: {hits}/{TEST_PERIODS} = {rate:.1f}%{marker}")
