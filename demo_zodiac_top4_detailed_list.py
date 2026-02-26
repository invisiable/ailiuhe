"""
演示生肖TOP4投注策略的最近100期详细列表功能
注意：这个脚本模拟GUI中的输出格式
"""
import sys
import io
import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

# 设置UTF-8编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def demo_detailed_list():
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    # 创建策略
    strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
    
    # 测试最近100期
    test_periods = 100
    start_idx = len(df) - test_periods
    
    predictions_top4 = []
    actuals = []
    hit_records = []
    
    print("正在生成预测...")
    for i in range(start_idx, len(df)):
        train_animals = df['animal'].iloc[:i].tolist()
        
        # 预测
        prediction = strategy.predict_top4(train_animals)
        top4 = prediction['top4']
        
        # 实际结果
        actual = df.iloc[i]['animal']
        hit = actual in top4
        
        predictions_top4.append(top4)
        actuals.append(actual)
        hit_records.append(hit)
        
        # 更新策略
        strategy.update_performance(hit)
        if (i - start_idx + 1) % 10 == 0:
            strategy.check_and_switch_model()
    
    # 计算统计
    hits = sum(hit_records)
    hit_rate = hits / len(hit_records)
    
    print("\n" + "="*80)
    print("📋 最近100期详细命中记录")
    print("="*80)
    print(f"{'期号':<6} {'日期':<12} {'实际生肖':<8} {'预测TOP4':<30} {'结果':<6}")
    print("-"*80)
    
    # 输出详细列表
    for i in range(100):
        record_idx = i
        period_idx = start_idx + record_idx
        
        date_str = df.iloc[period_idx]['date']
        actual_animal = actuals[record_idx]
        predicted_top4 = predictions_top4[record_idx]
        hit = hit_records[record_idx]
        
        # 格式化预测TOP4
        top4_str = ', '.join(predicted_top4)
        
        # 结果标记
        result_mark = "✓" if hit else "✗"
        
        # 期号
        period_number = i + 1
        
        print(f"{period_number:<6} {date_str:<12} {actual_animal:<8} {top4_str:<30} {result_mark:<6}")
    
    print("="*80)
    print(f"\n汇总: 命中{hits}/100期 = {hit_rate*100:.2f}%")
    print("\n✅ 详细列表演示完成！")
    print("\n💡 这个列表将在GUI的【生肖TOP4投注】按钮中自动显示")

if __name__ == '__main__':
    demo_detailed_list()
