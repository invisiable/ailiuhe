"""快速测试v12.0在50期上的性能"""
import pandas as pd
from zodiac_balanced_smart import ZodiacBalancedSmart

predictor = ZodiacBalancedSmart()
df = pd.read_csv('data/lucky_numbers.csv')
test_data = df.tail(50)
total = len(test_data)
hits = 0

for i in range(total):
    train_end = len(df) - total + i
    predictor_data = df.iloc[:train_end]
    
    if len(predictor_data) < 50:
        continue
    
    # 临时保存
    original_df = pd.read_csv('data/lucky_numbers.csv')
    predictor_data.to_csv('data/lucky_numbers.csv', index=False)
    
    # 预测 (静默模式)
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        top5 = predictor.predict_top5()
    finally:
        sys.stdout = old_stdout
    
    # 恢复
    original_df.to_csv('data/lucky_numbers.csv', index=False)
    
    # 检查
    actual = test_data.iloc[i]['animal']
    if actual in top5:
        hits += 1
    
    print(f'\r进度: {i+1}/50', end='')

print(f'\n\n50期TOP5命中: {hits}/50 = {hits/50*100:.1f}%')
if hits/50 >= 0.50:
    print("✓ 目标达成!")
else:
    print(f"✗ 距离目标还差 {(0.50 - hits/50)*100:.1f}%")
