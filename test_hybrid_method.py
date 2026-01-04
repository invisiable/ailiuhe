"""
测试hybrid_predict方法是否能正常运行
"""

import sys
import os

# 测试导入和执行
try:
    print("="*70)
    print("🧪 测试 hybrid_predict 方法")
    print("="*70)
    
    from final_hybrid_predictor import FinalHybridPredictor
    from datetime import datetime
    
    print("\n✅ 成功导入 FinalHybridPredictor")
    
    # 创建预测器
    predictor = FinalHybridPredictor()
    print("✅ 成功创建预测器实例")
    
    # 获取预测信息
    info = predictor.get_prediction_info()
    print("✅ 成功获取预测信息")
    
    print(f"\n📊 模型信息:")
    print(f"   版本: {info['version']}")
    print(f"   总记录数: {info['total_records']}")
    print(f"   最新期数: {info['latest_period']['date']}")
    print(f"   最新号码: {info['latest_period']['number']}")
    
    # 执行预测
    top15 = predictor.predict()
    print(f"\n✅ 成功执行预测")
    print(f"   TOP15: {top15}")
    
    # 获取分析
    import pandas as pd
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    analysis = predictor._analyze_full_history(numbers)
    print(f"\n📈 趋势分析:")
    print(f"   极端值趋势: {analysis['is_extreme']}")
    
    print("\n" + "="*70)
    print("✅ 所有功能测试通过！")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ 测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
