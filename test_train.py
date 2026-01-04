"""测试训练流程"""
from lucky_number_predictor import LuckyNumberPredictor
import traceback

p = LuckyNumberPredictor()
print("1. 加载数据...")
p.load_data('data/lucky_numbers.csv')
print(f"   数据加载成功: X shape={p.X.shape}, y shape={p.y.shape}")

print("\n2. 开始训练...")
try:
    result = p.train_model('gradient_boosting')
    print("   训练成功!")
    print(f"   测试MAE: {result['test_mae']:.4f}")
    print(f"   测试R²: {result['test_r2']:.4f}")
except Exception as e:
    print(f"   训练失败: {e}")
    traceback.print_exc()
