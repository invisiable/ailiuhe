"""
测试GUI投注策略按钮
"""

print("=" * 80)
print("测试GUI投注策略按钮是否能正常调用")
print("=" * 80)
print()

# 模拟GUI环境
class MockVar:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value

class MockGUI:
    def __init__(self):
        self.output_log = []
        self.file_path_var = MockVar('data/lucky_numbers.csv')
    
    def log_output(self, message):
        self.output_log.append(message)
        print(message, end='')
    
    def analyze_betting_strategy(self):
        """复制GUI中的方法进行测试"""
        import pandas as pd
        from datetime import datetime
        from betting_strategy import BettingStrategy
        from top15_predictor import Top15Predictor
        
        try:
            self.log_output(f"\n{'='*70}\n")
            self.log_output(f"💰 智能投注策略分析 - 收益最大化系统\n")
            self.log_output(f"{'='*70}\n")
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_output(f"分析时间: {current_time}\n\n")
            
            # 读取数据
            file_path = self.file_path_var.get()
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            if len(df) < 50:
                self.log_output("数据不足50期\n")
                return False
            
            self.log_output(f"✅ 数据加载完成: {len(df)}期\n")
            
            # 使用少量数据快速测试
            test_periods = 20
            start_idx = len(df) - test_periods
            
            self.log_output(f"测试期数: {test_periods}期\n\n")
            
            predictor = Top15Predictor()
            predictions_top5 = []
            actuals = []
            
            # 生成预测（使用与综合预测相同方法）
            for i in range(start_idx, len(df)):
                train_data = df.iloc[:i]['number'].values
                analysis = predictor.get_analysis(train_data)
                top15 = analysis['top15']
                top5 = top15[:5]
                predictions_top5.append(top5)
                
                actual = df.iloc[i]['number']
                actuals.append(actual)
                
                if (i - start_idx + 1) % 10 == 0:
                    self.log_output(f"  已处理 {i - start_idx + 1}/{test_periods} 期...\n")
            
            self.log_output(f"\n✅ 预测生成完成！共 {len(predictions_top5)} 期\n\n")
            
            # 创建投注策略
            betting = BettingStrategy()
            
            # 测试一个策略
            self.log_output("运行策略分析...\n")
            result = betting.simulate_strategy(predictions_top5, actuals, 'martingale')
            
            self.log_output(f"\n✅ 策略分析完成\n")
            self.log_output(f"  命中率: {result['hit_rate']*100:.1f}%\n")
            self.log_output(f"  总收益: {result['total_profit']:+.2f}元\n")
            self.log_output(f"  ROI: {result['roi']:+.1f}%\n")
            
            return True
            
        except Exception as e:
            self.log_output(f"\n❌ 错误: {str(e)}\n")
            import traceback
            traceback.print_exc()
            return False

# 测试
gui = MockGUI()
print("开始测试按钮点击功能...\n")

success = gui.analyze_betting_strategy()

print("\n" + "=" * 80)
if success:
    print("✅ GUI投注策略按钮功能正常！")
else:
    print("❌ GUI投注策略按钮功能失败")
print("=" * 80)
