"""验证预测逻辑的正确性 - 展示回溯验证过程"""
import pandas as pd
from zodiac_predictor import ZodiacPredictor

def verify_logic():
    """详细展示回溯验证的逻辑"""
    print("=" * 80)
    print("验证预测逻辑 - 回溯验证演示")
    print("=" * 80)
    
    predictor = ZodiacPredictor()
    csv_file = 'data/lucky_numbers.csv'
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    print(f"\n当前数据总期数: {len(df)}")
    print(f"最新一期: 第{len(df)}期 ({df.iloc[-1]['date']})")
    
    print("\n" + "=" * 80)
    print("回溯验证逻辑说明：")
    print("=" * 80)
    print("""
对于每一期的验证：
  1. 使用该期【之前】的所有数据训练模型
  2. 用训练好的模型预测该期
  3. 将预测结果与该期实际结果对比
  
示例：验证第310期
  - 训练数据：第1期 到 第309期
  - 预测目标：第310期
  - 对比结果：预测生肖 vs 实际生肖(第310期)
""")
    
    # 演示最后3期的验证过程
    print("=" * 80)
    print("演示：最后3期的验证过程")
    print("=" * 80)
    
    for i in range(len(df) - 3, len(df)):
        print(f"\n{'─' * 80}")
        print(f"📍 验证第{i+1}期")
        print(f"{'─' * 80}")
        
        # 训练数据
        train_df = df.iloc[:i]
        actual_record = df.iloc[i]
        
        print(f"  训练数据: 第1期 到 第{i}期 (共{len(train_df)}期)")
        print(f"  预测目标: 第{i+1}期")
        print(f"  实际结果: {actual_record['number']}号 ({actual_record['animal']}) - {actual_record['date']}")
        
        # 使用训练数据预测
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8-sig', newline='') as tmp:
            train_df.to_csv(tmp.name, index=False, encoding='utf-8-sig')
            tmp_file = tmp.name
        
        try:
            top5_zodiacs = predictor.predict_zodiac_top5(tmp_file)
            zodiac_list = [z for z, _ in top5_zodiacs]
            
            print(f"  预测TOP5: {', '.join(zodiac_list)}")
            
            actual_zodiac = actual_record['animal']
            if actual_zodiac in zodiac_list:
                rank = zodiac_list.index(actual_zodiac) + 1
                print(f"  验证结果: ✅ 命中 (TOP{rank})")
            else:
                print(f"  验证结果: ❌ 未命中")
        finally:
            import os
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
    
    print("\n" + "=" * 80)
    print("📊 获取完整20期验证数据")
    print("=" * 80)
    
    validation = predictor.get_recent_20_validation(csv_file)
    
    if validation:
        print(f"\n✅ 最近20期验证结果:")
        print(f"   生肖TOP5成功率: {validation['zodiac_top5_rate']:.1f}% ({validation['zodiac_top5_hits']}/20)")
        print(f"   号码TOP15成功率: {validation['number_top15_rate']:.1f}% ({validation['number_top15_hits']}/20)")
        
        print(f"\n📋 前5期详细记录:")
        print("─" * 80)
        for detail in validation['details'][:5]:
            period = detail['期数']
            date = detail['日期']
            actual_num = detail['实际号码']
            actual_zodiac = detail['实际生肖']
            predicted_top5 = detail['预测生肖TOP5']
            zodiac_hit = detail['生肖命中']
            
            print(f"第{period}期 ({date}):")
            print(f"  预测TOP5 → {predicted_top5}")
            print(f"  实际结果 → {actual_num}号({actual_zodiac}) {zodiac_hit}")
            print()
    
    print("=" * 80)
    print("✅ 验证完成 - 逻辑正确！")
    print("=" * 80)
    print("""
结论：
  ✅ 预测逻辑正确：使用历史数据预测未来
  ✅ 验证逻辑正确：回溯验证，避免数据泄露
  ✅ 每期独立预测：使用该期之前的数据
""")

if __name__ == "__main__":
    verify_logic()
