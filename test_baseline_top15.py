import pandas as pd
import numpy as np
from collections import Counter

# Load data
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values

# Import all TOP15 predictors
from precise_top15_predictor import PreciseTop15Predictor
from top15_statistical_predictor import Top15StatisticalPredictor
from top15_zodiac_enhanced_v2 import Top15ZodiacEnhancedV2
from ensemble_top15_predictor import EnsembleTop15Predictor
from top15_predictor import Top15Predictor

predictors = {
    'PreciseTop15': PreciseTop15Predictor(),
    'Statistical': Top15StatisticalPredictor(),
    'ZodiacEnhV2': Top15ZodiacEnhancedV2(),
    'Ensemble': EnsembleTop15Predictor(),
    'BaseTop15': Top15Predictor(),
}

TRAIN_WINDOW = 25
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS

print(f"总数据: {len(df)}期, 测试: {TEST_PERIODS}期, 窗口: {TRAIN_WINDOW}期")
print(f"测试范围: 第{start_idx+1}期 ~ 第{len(df)}期\n")

# Test each predictor
for name, predictor in predictors.items():
    hits = 0
    for i in range(start_idx, len(df)):
        lo = max(0, i - TRAIN_WINDOW)
        train_data = numbers[lo:i]
        try:
            preds = predictor.predict(train_data)
        except:
            try:
                preds = predictor.predict_top15(train_data)
            except:
                preds = predictor.predict_top20(train_data)[:15]
        actual = numbers[i]
        if actual in preds:
            hits += 1
    rate = hits / TEST_PERIODS * 100
    print(f"{name:20s}: 命中 {hits}/{TEST_PERIODS} = {rate:.1f}%")

# Also test with different window sizes for PreciseTop15
print("\n--- PreciseTop15 不同窗口期 ---")
for window in [15, 20, 25, 30, 40, 50, 75, 100]:
    predictor = PreciseTop15Predictor()
    hits = 0
    for i in range(start_idx, len(df)):
        lo = max(0, i - window)
        train_data = numbers[lo:i]
        preds = predictor.predict(train_data)
        actual = numbers[i]
        if actual in preds:
            hits += 1
            predictor.update_performance(preds, actual)
        else:
            predictor.update_performance(preds, actual)
    rate = hits / TEST_PERIODS * 100
    print(f"窗口{window:3d}期: 命中 {hits}/{TEST_PERIODS} = {rate:.1f}%")

# Test ensemble with different combos using top20
print("\n--- 各预测器 TOP20 命中率 ---")
for name in ['Statistical', 'ZodiacEnhV2', 'Ensemble']:
    pred = predictors[name]
    hits = 0
    for i in range(start_idx, len(df)):
        lo = max(0, i - TRAIN_WINDOW)
        train_data = numbers[lo:i]
        try:
            preds = pred.predict_top20(train_data)
        except:
            preds = pred.predict(train_data)
        actual = numbers[i]
        if actual in preds:
            hits += 1
    rate = hits / TEST_PERIODS * 100
    print(f"{name:20s} Top20: 命中 {hits}/{TEST_PERIODS} = {rate:.1f}%")
