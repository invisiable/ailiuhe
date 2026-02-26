"""
验证下期预测是基于最新的完整历史数据
"""
import sys
import io
import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def verify_prediction_data():
    print("="*80)
    print("验证下期预测使用的数据范围")
    print("="*80)
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"\n✅ 数据加载完成: 共{len(df)}期")
    print(f"数据范围: {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")
    
    # 获取最后一期的信息
    last_period = df.iloc[-1]
    print(f"\n最新一期数据:")
    print(f"  日期: {last_period['date']}")
    print(f"  号码: {last_period['number']}")
    print(f"  生肖: {last_period['animal']}")
    print(f"  五行: {last_period['element']}")
    
    # 创建策略
    strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
    
    # 使用全部历史数据进行预测（包括最新一期）
    all_animals = df['animal'].tolist()
    print(f"\n📊 预测输入数据:")
    print(f"  使用期数: {len(all_animals)}期")
    print(f"  数据范围: {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")
    print(f"  最近5期生肖: {', '.join(all_animals[-5:])}")
    
    # 生成下期预测
    prediction_result = strategy.predict_top4(all_animals)
    next_top4 = prediction_result['top4']
    predictor_name = prediction_result['predictor']
    model_name = strategy.get_current_model_name()
    
    print(f"\n🎯 下期预测结果:")
    print(f"  预测TOP4: {', '.join(next_top4)}")
    print(f"  使用模型: {model_name}")
    print(f"  预测器: {predictor_name}")
    
    # 验证：如果移除最后一期数据，预测结果应该不同
    print(f"\n🔍 验证预测是否使用最新数据:")
    animals_without_last = df['animal'].iloc[:-1].tolist()
    prediction_without_last = strategy.predict_top4(animals_without_last)
    top4_without_last = prediction_without_last['top4']
    
    print(f"  不含最新期的预测: {', '.join(top4_without_last)}")
    print(f"  包含最新期的预测: {', '.join(next_top4)}")
    
    if top4_without_last != next_top4:
        print(f"\n✅ 验证通过：预测结果不同，确认使用了最新一期数据")
    else:
        print(f"\n⚠️  预测结果相同（可能模型对最新一期不敏感）")
    
    print(f"\n{'='*80}")
    print("结论：下期预测基于全部{0}期历史数据（包括{1}的最新一期）".format(
        len(all_animals), df.iloc[-1]['date']))
    print("="*80)

if __name__ == '__main__':
    verify_prediction_data()
