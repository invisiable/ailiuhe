"""
测试生肖TOP4投注策略的最近100期和2026年统计功能
"""
import sys
import io
import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

# 设置输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_stats():
    print("="*80)
    print("测试生肖TOP4投注策略 - 最近100期和2026年统计")
    print("="*80)
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"\n✅ 数据加载完成: {len(df)}期")
    print(f"数据范围: {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}\n")
    
    # 创建策略实例
    strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
    
    # 回测参数
    test_periods = min(200, len(df) - 50)  # 测试最近200期
    start_idx = len(df) - test_periods
    
    print(f"开始回测分析: {test_periods}期")
    print(f"起始索引: {start_idx} ({df.iloc[start_idx]['date']})\n")
    
    # 回测
    hit_records = []
    monthly_profits = {}
    predictions_list = []  # 存储预测结果
    actuals_list = []  # 存储实际结果
    dates_list = []  # 存储日期
    
    for i in range(start_idx, len(df)):
        train_animals = df['animal'].iloc[:i].tolist()
        
        # 预测
        prediction = strategy.predict_top4(train_animals)
        top4 = prediction['top4']
        
        # 实际结果
        actual = df.iloc[i]['animal']
        date_str = df.iloc[i]['date']
        hit = actual in top4
        
        hit_records.append(hit)
        predictions_list.append(top4)
        actuals_list.append(actual)
        dates_list.append(date_str)
        
        # 更新策略性能
        strategy.update_performance(hit)
        
        # 更新月度收益
        date_str = df.iloc[i]['date']
        try:
            date_obj = pd.to_datetime(date_str)
            month_key = date_obj.strftime('%Y/%m')
        except:
            month_key = date_str[:7] if len(date_str) >= 7 else '未知'
        
        if month_key not in monthly_profits:
            monthly_profits[month_key] = 0
        
        period_profit = 30 if hit else -16
        monthly_profits[month_key] += period_profit
        
        # 每10期检查模型切换
        if (i - start_idx + 1) % 10 == 0:
            strategy.check_and_switch_model()
    
    print(f"✅ 回测完成\n")
    
    # 计算总体统计
    hits = sum(hit_records)
    hit_rate = hits / len(hit_records)
    total_investment = 16 * len(hit_records)
    total_profit = hits * 30 - (len(hit_records) - hits) * 16
    roi = (total_profit / total_investment) * 100
    
    print("="*80)
    print("总体回测结果")
    print("="*80)
    print(f"测试期数: {len(hit_records)}期")
    print(f"命中次数: {hits}/{len(hit_records)} = {hit_rate*100:.2f}%")
    print(f"总投入: {total_investment}元")
    print(f"净收益: {total_profit:+.0f}元")
    print(f"投资回报率: {roi:+.2f}%\n")
    
    # ===== 测试1: 最近100期统计 =====
    if len(hit_records) >= 100:
        print("="*80)
        print("📈 最近100期表现分析")
        print("="*80)
        
        recent_100_hits = sum(hit_records[-100:])
        recent_100_rate = recent_100_hits / 100
        recent_100_profit = recent_100_hits * 30 - (100 - recent_100_hits) * 16
        recent_100_investment = 16 * 100
        recent_100_roi = (recent_100_profit / recent_100_investment) * 100
        
        print(f"命中次数: {recent_100_hits}/100 = {recent_100_rate*100:.2f}%")
        print(f"总投入: {recent_100_investment}元")
        print(f"总中奖: {recent_100_hits * 46}元")
        print(f"净收益: {recent_100_profit:+.0f}元")
        print(f"投资回报率: {recent_100_roi:+.2f}%")
        
        # 计算连续命中和连续未中
        recent_100 = hit_records[-100:]
        max_consecutive_hits = 0
        max_consecutive_misses = 0
        current_hits = 0
        current_misses = 0
        
        for h in recent_100:
            if h:
                current_hits += 1
                current_misses = 0
                max_consecutive_hits = max(max_consecutive_hits, current_hits)
            else:
                current_misses += 1
                current_hits = 0
                max_consecutive_misses = max(max_consecutive_misses, current_misses)
        
        print(f"最大连续命中: {max_consecutive_hits}期")
        print(f"最大连续未中: {max_consecutive_misses}期")
        print()
        
        # ===== 测试3: 最近100期详细列表 =====
        print("="*80)
        print("📋 最近100期详细命中记录")
        print("="*80)
        print(f"{'期号':<6} {'日期':<12} {'实际生肖':<8} {'预测TOP4':<30} {'结果':<6}")
        print("-"*80)
        
        # 获取最近100期的数据
        recent_100_start_idx = len(hit_records) - 100
        for i in range(100):
            record_idx = recent_100_start_idx + i
            
            date_str = dates_list[record_idx]
            actual_animal = actuals_list[record_idx]
            predicted_top4 = predictions_list[record_idx]
            hit = hit_records[record_idx]
            
            # 格式化预测TOP4
            top4_str = ', '.join(predicted_top4)
            
            # 结果标记
            result_mark = "✓" if hit else "✗"
            
            # 计算连续编号（从倒数第100期开始）
            period_number = i + 1
            
            print(f"{period_number:<6} {date_str:<12} {actual_animal:<8} {top4_str:<30} {result_mark:<6}")
        
        print("="*80)
        print()
    
    # ===== 测试2: 2026年统计 =====
    print("="*80)
    print("📊 每月收益统计")
    print("="*80)
    
    profit_2026 = 0
    months_2026 = []
    
    for month in sorted(monthly_profits.keys()):
        profit = monthly_profits[month]
        print(f"{month}: {profit:+10.0f}元")
        if month.startswith('2026'):
            profit_2026 += profit
            months_2026.append(month)
    
    print("="*80)
    
    # 输出2026年汇总
    if profit_2026 != 0 and months_2026:
        print(f"\n💰 2026年收益汇总")
        print("="*80)
        print(f"统计月份: {len(months_2026)}个月 ({months_2026[0]} ~ {months_2026[-1]})")
        print(f"总收益: {profit_2026:+.0f}元")
        
        # 计算2026年的期数
        year_2026_periods = 0
        for idx, hit in enumerate(hit_records):
            period_idx = start_idx + idx
            date_str = df.iloc[period_idx]['date']
            try:
                date_obj = pd.to_datetime(date_str)
                if date_obj.year == 2026:
                    year_2026_periods += 1
            except:
                pass
        
        if year_2026_periods > 0:
            investment_2026 = 16 * year_2026_periods
            roi_2026 = (profit_2026 / investment_2026) * 100
            # 计算2026年命中次数
            hits_2026 = 0
            for idx, hit in enumerate(hit_records):
                period_idx = start_idx + idx
                date_str = df.iloc[period_idx]['date']
                try:
                    date_obj = pd.to_datetime(date_str)
                    if date_obj.year == 2026 and hit:
                        hits_2026 += 1
                except:
                    pass
            
            hit_rate_2026 = (hits_2026 / year_2026_periods * 100) if year_2026_periods > 0 else 0
            
            print(f"投注期数: {year_2026_periods}期")
            print(f"命中次数: {hits_2026}次 ({hit_rate_2026:.2f}%)")
            print(f"总投入: {investment_2026}元")
            print(f"投资回报率: {roi_2026:+.2f}%")
        
        print("="*80)
    else:
        print("\n⚠️  没有2026年的数据")
    
    print("\n✅ 统计功能测试完成！")

if __name__ == '__main__':
    test_stats()
