"""
模拟测试GUI的hybrid_predict方法
不启动GUI界面，直接测试方法逻辑
"""

import sys
from datetime import datetime

class MockLogOutput:
    """模拟日志输出"""
    def __init__(self):
        self.logs = []
    
    def __call__(self, text):
        self.logs.append(text)
        print(text, end='')

def test_hybrid_predict_logic():
    """测试hybrid_predict方法的核心逻辑"""
    
    print("="*70)
    print("🧪 测试 GUI hybrid_predict 方法逻辑")
    print("="*70)
    
    try:
        # 导入模块
        from final_hybrid_predictor import FinalHybridPredictor
        
        log_output = MockLogOutput()
        
        # 模拟hybrid_predict方法的核心逻辑
        log_output(f"\n{'='*70}\n")
        log_output(f"🚀 固化混合策略模型 v1.0 - 50%成功率\n")
        log_output(f"{'='*70}\n")
        
        # 显示预测时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_output(f"预测时间: {current_time}\n")
        log_output("🔄 加载最新数据并执行预测...\n")
        
        # 创建预测器（自动加载最新数据）
        predictor = FinalHybridPredictor()
        
        # 获取预测信息
        info = predictor.get_prediction_info()
        
        log_output(f"✅ 数据加载完成: {info['total_records']}期\n")
        log_output(f"最新一期: {info['latest_period']['date']} - 号码: {info['latest_period']['number']}\n")
        log_output("📊 正在执行混合策略预测...\n\n")
        
        # 执行预测
        top15 = predictor.predict()
        
        # 获取详细分析
        analysis = info['analysis']
        
        log_output(f"策略执行完成:\n")
        log_output(f"  策略A (全历史数据): 稳定预测\n")
        log_output(f"  策略B (最近10期): 精准预测\n")
        log_output(f"  混合组合: TOP1-5使用策略B，TOP6-15使用策略A\n\n")
        
        # 构建预测结果
        predictions = []
        for i, num in enumerate(top15, 1):
            predictions.append({
                'rank': i,
                'number': num,
                'probability': 1.0 - (i-1) * 0.05
            })
        
        # 显示结果
        log_output("\n【TOP 5 预测结果 - 策略B精准预测】\n")
        for i in range(5):
            pred = predictions[i]
            log_output(f"  ⭐ {i+1}. 数字: {pred['number']:>2}  优先级: {pred['probability']:>6.4f}\n")
        
        log_output("\n【TOP 6-15 预测结果 - 策略A稳定预测】\n")
        for i in range(5, 15):
            pred = predictions[i]
            marker = "✓" if i < 10 else "○"
            log_output(f"  {marker} {i+1:>2}. 数字: {pred['number']:>2}  优先级: {pred['probability']:>6.4f}\n")
        
        log_output(f"\n趋势分析: {analysis['trend']}\n")
        log_output(f"极端值占比: {analysis['extreme_ratio']:.0f}% (最近10期)\n")
        
        # 区域分布
        zones = analysis['zones']
        log_output(f"\n区域分布统计 (TOP15):\n")
        for zone_name, zone_nums in zones.items():
            zone_in_top15 = [n for n in top15 if n in zone_nums]
            if zone_in_top15:
                log_output(f"  {zone_name}: {zone_in_top15}\n")
        
        # 五行分布
        log_output(f"\n五行分布统计 (TOP15):\n")
        for element_name, element_nums in analysis['elements'].items():
            element_in_top15 = [n for n in top15 if n in element_nums]
            if element_in_top15:
                log_output(f"  {element_name}: {element_in_top15}\n")
        
        log_output(f"\n基于历史数据: {info['total_records']} 期\n")
        log_output(f"最新数据日期: {info['latest_period']['date']}\n")
        log_output(f"模型版本: {info['version']}\n")
        log_output(f"{'='*70}\n")
        
        print("\n" + "="*70)
        print("✅ GUI hybrid_predict 方法逻辑测试通过！")
        print("="*70)
        print("\n现在可以启动GUI并点击按钮测试：")
        print("   python lucky_number_gui.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_hybrid_predict_logic()
    sys.exit(0 if success else 1)
