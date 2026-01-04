"""
测试UI集成 - 验证固化混合策略模型是否正确集成
"""

import sys
import os

def test_ui_integration():
    """测试UI集成"""
    
    print("="*70)
    print("🧪 测试固化混合策略模型的UI集成")
    print("="*70)
    
    # 1. 检查必要文件
    print("\n1️⃣ 检查必要文件...")
    
    files_to_check = [
        'lucky_number_gui.py',
        'final_hybrid_predictor.py',
        'data/lucky_numbers.csv'
    ]
    
    all_exist = True
    for file_path in files_to_check:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n❌ 缺少必要文件，无法继续测试")
        return False
    
    # 2. 检查GUI代码中是否包含hybrid_predict方法
    print("\n2️⃣ 检查GUI代码集成...")
    
    with open('lucky_number_gui.py', 'r', encoding='utf-8') as f:
        gui_code = f.read()
    
    checks = [
        ('hybrid_predict方法', 'def hybrid_predict(self):'),
        ('混合策略按钮', 'hybrid_button'),
        ('FinalHybridPredictor导入', 'from final_hybrid_predictor import FinalHybridPredictor'),
        ('固化混合策略标题', '固化混合策略'),
    ]
    
    all_integrated = True
    for check_name, check_pattern in checks:
        exists = check_pattern in gui_code
        status = "✅" if exists else "❌"
        print(f"   {status} {check_name}")
        if not exists:
            all_integrated = False
    
    if not all_integrated:
        print("\n❌ GUI代码集成不完整")
        return False
    
    # 3. 测试FinalHybridPredictor是否可以正常导入和运行
    print("\n3️⃣ 测试FinalHybridPredictor功能...")
    
    try:
        from final_hybrid_predictor import FinalHybridPredictor
        print("   ✅ 成功导入 FinalHybridPredictor")
        
        predictor = FinalHybridPredictor()
        print("   ✅ 成功创建预测器实例")
        
        top15 = predictor.predict()
        print(f"   ✅ 成功执行预测: TOP15 = {top15}")
        
        info = predictor.get_prediction_info()
        print(f"   ✅ 成功获取预测信息")
        print(f"      - 版本: {info['version']}")
        print(f"      - 数据周期: {info['total_records']}")
        print(f"      - 最新日期: {info['latest_period']['date']}")
        print(f"      - 成功率: TOP15={info['success_rate']['top15']}")
        
    except Exception as e:
        print(f"   ❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 检查UI集成文档
    print("\n4️⃣ 检查集成文档...")
    
    doc_files = [
        'UI集成说明.md',
        '固化模型使用说明.md',
        '固化完成报告.md'
    ]
    
    for doc_file in doc_files:
        exists = os.path.exists(doc_file)
        status = "✅" if exists else "⚠️"
        print(f"   {status} {doc_file}")
    
    # 总结
    print("\n" + "="*70)
    print("✅ 所有测试通过！固化混合策略模型已成功集成到UI")
    print("="*70)
    
    print("\n🚀 使用方式:")
    print("   python lucky_number_gui.py")
    print("\n   然后点击 '🚀 固化混合策略 v1.0' 按钮")
    
    return True

if __name__ == '__main__':
    success = test_ui_integration()
    sys.exit(0 if success else 1)
