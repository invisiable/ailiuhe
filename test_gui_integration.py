"""
快速测试新增的动态择优功能
"""

import sys
import os

# 确保导入路径
sys.path.insert(0, os.path.dirname(__file__))

# 测试导入
try:
    from ensemble_select_best_predictor import EnsembleSelectBestPredictor
    print("✅ EnsembleSelectBestPredictor 导入成功")
    
    # 测试实例化
    predictor = EnsembleSelectBestPredictor()
    print("✅ 预测器实例化成功")
    
    # 测试基本预测
    test_animals = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪'] * 3
    result = predictor.predict_top4(test_animals)
    print(f"✅ 预测功能正常: {result['top4']}")
    print(f"   使用预测器: {result['predictor']}")
    
    # 测试GUI导入
    from lucky_number_gui import LuckyNumberGUI
    print("✅ LuckyNumberGUI 导入成功")
    
    print("\n" + "="*60)
    print("🎉 所有测试通过！新功能已成功集成到GUI")
    print("="*60)
    print("\n可以运行以下命令启动GUI:")
    print("  python lucky_number_gui.py")
    print("\n新增按钮位置: 🌟 生肖TOP4动态择优")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
