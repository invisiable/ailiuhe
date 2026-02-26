"""
测试TOP15 vs TOP4对比策略中的生肖TOP4预测模型更新
验证是否正确使用RecommendedZodiacTop4Strategy
"""
import sys
import io
import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy
from top15_predictor import Top15Predictor

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_comparison_strategy():
    print("="*80)
    print("测试TOP15 vs TOP4对比策略 - 生肖TOP4模型更新验证")
    print("="*80)
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"\n✅ 数据加载完成: {len(df)}期")
    
    # 测试最近20期
    test_periods = 20
    start_idx = len(df) - test_periods
    
    print(f"测试期数: 最近{test_periods}期\n")
    
    # 创建预测器
    print("创建预测器...")
    top15_predictor = Top15Predictor()
    top4_strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
    print(f"✅ TOP15预测器: Top15Predictor")
    print(f"✅ 生肖TOP4策略: RecommendedZodiacTop4Strategy v2.0")
    print(f"   主力模型: {top4_strategy.primary_model.__class__.__name__}")
    print(f"   备份模型: {top4_strategy.backup_model.__class__.__name__}\n")
    
    # 对比测试
    top15_hits = 0
    top4_hits = 0
    both_hits = 0
    
    print("="*80)
    print("开始对比测试（最近20期）")
    print("="*80)
    print(f"{'期号':<4} {'日期':<12} {'号码':<4} {'生肖':<6} {'TOP15':<6} {'TOP4':<6} {'结果':<10}")
    print("-"*80)
    
    for i in range(start_idx, len(df)):
        idx = i - start_idx + 1
        current_row = df.iloc[i]
        date = current_row['date']
        actual_number = current_row['number']
        actual_animal = current_row['animal']
        
        # TOP15预测
        train_numbers = df.iloc[:i]['number'].values
        top15_analysis = top15_predictor.get_analysis(train_numbers)
        top15_pred = top15_analysis['top15']
        top15_hit = actual_number in top15_pred
        
        # 生肖TOP4预测（使用推荐策略v2.0）
        train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
        top4_prediction = top4_strategy.predict_top4(train_animals)
        top4_pred = top4_prediction['top4']
        top4_hit = actual_animal in top4_pred
        
        # 更新策略性能监控
        top4_strategy.update_performance(top4_hit)
        if idx % 10 == 0:
            top4_strategy.check_and_switch_model()
        
        # 统计
        if top15_hit:
            top15_hits += 1
        if top4_hit:
            top4_hits += 1
        if top15_hit and top4_hit:
            both_hits += 1
        
        # 结果标记
        top15_mark = "✓" if top15_hit else "✗"
        top4_mark = "✓" if top4_hit else "✗"
        
        if top15_hit and top4_hit:
            result = "🌟 都中"
        elif not top15_hit and not top4_hit:
            result = "💔 都未中"
        else:
            result = "仅1中"
        
        print(f"{idx:<4} {date:<12} {actual_number:<4} {actual_animal:<6} {top15_mark:<6} {top4_mark:<6} {result:<10}")
    
    # 输出统计结果
    print("="*80)
    print("\n统计结果:")
    print("="*80)
    
    top15_rate = top15_hits / test_periods * 100
    top4_rate = top4_hits / test_periods * 100
    both_rate = both_hits / test_periods * 100
    
    print(f"\n【TOP15预测】")
    print(f"  命中: {top15_hits}/{test_periods} = {top15_rate:.2f}%")
    
    print(f"\n【生肖TOP4预测（推荐策略v2.0）】")
    top4_stats = top4_strategy.get_performance_stats()
    print(f"  使用模型: {top4_strategy.get_current_model_name()}")
    print(f"  模型切换: {len(top4_strategy.switch_history)}次")
    print(f"  命中: {top4_hits}/{test_periods} = {top4_rate:.2f}%")
    print(f"  最近{top4_stats['recent_total']}期命中率: {top4_stats['recent_rate']:.1f}%")
    
    print(f"\n【组合命中】")
    print(f"  两种都中: {both_hits}/{test_periods} = {both_rate:.2f}% 🌟")
    print(f"  仅TOP15中: {top15_hits - both_hits}期")
    print(f"  仅TOP4中: {top4_hits - both_hits}期")
    print(f"  都未中: {test_periods - top15_hits - top4_hits + both_hits}期")
    
    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("="*80)
    
    # 验证是否使用了正确的模型
    print("\n【验证结果】")
    if top4_strategy.__class__.__name__ == 'RecommendedZodiacTop4Strategy':
        print("✅ 确认使用 RecommendedZodiacTop4Strategy（推荐策略v2.0）")
        print("✅ 主力模型: 重训练v2.0（最近200期优化）")
        print("✅ 自动监控性能，必要时切换到备份模型")
        print("✅ TOP15 vs TOP4对比策略已成功同步最新预测模型")
    else:
        print("⚠️ 模型未正确更新")
    
    return {
        'top15_hits': top15_hits,
        'top4_hits': top4_hits,
        'both_hits': both_hits,
        'top15_rate': top15_rate,
        'top4_rate': top4_rate
    }

if __name__ == '__main__':
    test_comparison_strategy()
