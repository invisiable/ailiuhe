"""
Top 20 预测器 - 针对极小值优化版本
目标: 将成功率从50%提升到60%+
"""

import pandas as pd
import numpy as np
from enhanced_predictor_v2 import EnhancedPredictor
from lucky_number_predictor import LuckyNumberPredictor

class Top20Predictor:
    """
    Top 20 预测器 - 增强版
    
    特点:
    1. 使用综合预测生成Top 15基础
    2. 额外添加5个极端值候选
    3. 动态调整极端值权重
    """
    
    def __init__(self, predictors):
        """
        初始化
        
        Args:
            predictors: LuckyNumberPredictor实例列表
        """
        self.enhanced_predictor = EnhancedPredictor(predictors)
        self.predictors = predictors
    
    def analyze_extreme_trend(self, recent_numbers, window=10):
        """
        分析最近极端值趋势
        
        Args:
            recent_numbers: 最近的数字列表
            window: 分析窗口大小
        
        Returns:
            dict: 极端值统计
        """
        recent = recent_numbers[-window:]
        
        extreme_small = [n for n in recent if n <= 10]
        extreme_large = [n for n in recent if n >= 40]
        
        return {
            'small_count': len(extreme_small),
            'large_count': len(extreme_large),
            'small_ratio': len(extreme_small) / len(recent),
            'large_ratio': len(extreme_large) / len(recent),
            'has_trend': len(extreme_small) + len(extreme_large) >= window * 0.3
        }
    
    def get_extreme_candidates(self, recent_numbers, k=5):
        """
        获取极端值候选
        
        策略:
        1. 分析最近趋势
        2. 根据历史频率选择极小值和极大值
        3. 避免最近已出现的数字
        
        Args:
            recent_numbers: 最近的数字列表
            k: 需要的候选数
        
        Returns:
            list: 极端值候选列表
        """
        # 分析趋势
        trend = self.analyze_extreme_trend(recent_numbers)
        
        # 最近5期避免重复
        recent_5 = set(recent_numbers[-5:])
        
        # 极小值候选 (1-10)
        small_candidates = [n for n in range(1, 11) if n not in recent_5]
        
        # 极大值候选 (40-49)
        large_candidates = [n for n in range(40, 50) if n not in recent_5]
        
        # 根据趋势调整比例
        if trend['small_ratio'] > 0.4:  # 极小值趋势明显
            small_count = min(k - 1, len(small_candidates))
            large_count = k - small_count
        elif trend['large_ratio'] > 0.3:  # 极大值趋势明显
            large_count = min(k - 1, len(large_candidates))
            small_count = k - large_count
        else:  # 平衡分配
            small_count = k // 2
            large_count = k - small_count
        
        # 基于历史频率选择
        all_numbers = recent_numbers[-30:]  # 最近30期
        
        # 极小值频率
        small_freq = {}
        for n in small_candidates:
            small_freq[n] = all_numbers.count(n)
        
        # 极大值频率
        large_freq = {}
        for n in large_candidates:
            large_freq[n] = all_numbers.count(n)
        
        # 选择频率较高的
        selected_small = sorted(small_freq.keys(), 
                               key=lambda x: small_freq[x], 
                               reverse=True)[:small_count]
        selected_large = sorted(large_freq.keys(), 
                               key=lambda x: large_freq[x], 
                               reverse=True)[:large_count]
        
        return list(selected_small) + list(selected_large)
    
    def predict_top20(self, file_path='lucky_numbers.csv', 
                      number_col='number', 
                      date_col='date',
                      animal_col='animal',
                      element_col='element'):
        """
        Top 20 预测
        
        Args:
            file_path: 数据文件路径
            number_col: 数字列名
            date_col: 日期列名
            animal_col: 生肖列名
            element_col: 五行列名
        
        Returns:
            list: Top 20预测结果，每个元素包含number和probability
        """
        # 1. 获取Top 15基础预测
        print("\n🔮 第一步: 生成Top 15基础预测...")
        top15_results = self.enhanced_predictor.comprehensive_predict_v2(
            file_path=file_path,
            number_col=number_col,
            date_col=date_col,
            animal_col=animal_col,
            element_col=element_col,
            top_k=15
        )
        
        # 提取已预测的数字
        predicted_numbers = set([r['number'] for r in top15_results])
        print(f"   ✓ Top 15: {sorted(predicted_numbers)}")
        
        # 2. 获取历史数据
        df = pd.read_csv(file_path)
        recent_numbers = df[number_col].tolist()
        
        # 3. 分析极端值趋势
        print("\n📊 第二步: 分析极端值趋势...")
        trend = self.analyze_extreme_trend(recent_numbers)
        print(f"   • 最近10期极小值: {trend['small_count']}次 ({trend['small_ratio']*100:.1f}%)")
        print(f"   • 最近10期极大值: {trend['large_count']}次 ({trend['large_ratio']*100:.1f}%)")
        print(f"   • 趋势判断: {'⚠️ 有明显极端值趋势' if trend['has_trend'] else '✓ 正常分布'}")
        
        # 4. 获取极端值候选
        print("\n🎯 第三步: 选择极端值候选...")
        extreme_candidates = self.get_extreme_candidates(recent_numbers, k=10)
        
        # 过滤已在Top 15中的
        new_candidates = [n for n in extreme_candidates if n not in predicted_numbers]
        print(f"   • 极端值候选: {extreme_candidates}")
        print(f"   • 新增候选: {new_candidates[:5]}")
        
        # 5. 合并结果
        print("\n✨ 第四步: 合并Top 20...")
        top20_results = top15_results.copy()
        
        # 添加新候选（概率逐渐降低）
        base_prob = top15_results[-1]['probability'] * 0.8
        for i, num in enumerate(new_candidates[:5]):
            top20_results.append({
                'number': num,
                'animal': '未知',
                'element': '未知',
                'probability': base_prob * (0.9 ** i),
                'source': 'extreme_value'
            })
        
        # 按概率排序
        top20_results.sort(key=lambda x: x['probability'], reverse=True)
        
        print(f"   ✓ Top 20完成: {[r['number'] for r in top20_results[:20]]}")
        
        return top20_results[:20]
    
    def validate_on_period(self, target_period, file_path='lucky_numbers.csv',
                          number_col='number', date_col='date',
                          animal_col='animal', element_col='element'):
        """
        在指定期数上验证
        
        Args:
            target_period: 目标期数
            file_path: 数据文件路径
            其他: 列名参数
        
        Returns:
            dict: 验证结果
        """
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 获取目标期的实际值
        target_row = df[df.index == target_period]
        if len(target_row) == 0:
            return None
        
        actual_number = target_row[number_col].values[0]
        
        # 使用之前的数据进行预测
        temp_df = df.iloc[:target_period]
        temp_file = 'temp_validate.csv'
        temp_df.to_csv(temp_file, index=False)
        
        # 预测
        predictions = self.predict_top20(
            file_path=temp_file,
            number_col=number_col,
            date_col=date_col,
            animal_col=animal_col,
            element_col=element_col
        )
        
        predicted_numbers = [p['number'] for p in predictions]
        
        # 检查命中
        if actual_number in predicted_numbers:
            rank = predicted_numbers.index(actual_number) + 1
            hit_top5 = rank <= 5
            hit_top10 = rank <= 10
            hit_top15 = rank <= 15
            hit_top20 = True
        else:
            rank = -1
            hit_top5 = hit_top10 = hit_top15 = hit_top20 = False
        
        return {
            'period': target_period,
            'actual': actual_number,
            'predicted': predicted_numbers,
            'hit_top5': hit_top5,
            'hit_top10': hit_top10,
            'hit_top15': hit_top15,
            'hit_top20': hit_top20,
            'rank': rank
        }


def validate_top20_strategy():
    """
    验证Top 20策略
    """
    print("=" * 80)
    print("Top 20 策略验证 - 极端值优化版")
    print("=" * 80)
    
    # 1. 加载模型
    print("\n📦 加载模型...")
    model_names = ['gradient_boosting', 'lightgbm', 'xgboost']
    predictors = []
    
    for name in model_names:
        predictor = LuckyNumberPredictor()
        predictor.load_model(f'models/{name}_model.pkl')
        predictors.append(predictor)
        print(f"   ✓ {name}")
    
    # 2. 创建Top 20预测器
    top20 = Top20Predictor(predictors)
    
    # 3. 在最近10期上验证
    print("\n" + "=" * 80)
    print("在最近10期上验证")
    print("=" * 80)
    
    df = pd.read_csv('lucky_numbers.csv')
    total_periods = len(df)
    
    results = {
        'top5': 0,
        'top10': 0,
        'top15': 0,
        'top20': 0,
        'details': []
    }
    
    for i in range(total_periods - 10, total_periods):
        result = top20.validate_on_period(i)
        if result:
            results['details'].append(result)
            if result['hit_top5']:
                results['top5'] += 1
            if result['hit_top10']:
                results['top10'] += 1
            if result['hit_top15']:
                results['top15'] += 1
            if result['hit_top20']:
                results['top20'] += 1
            
            status = "✅" if result['hit_top20'] else "❌"
            rank_str = f"排名{result['rank']}" if result['rank'] > 0 else "未命中"
            print(f"\n  第{i}期: 实际{result['actual']} {status} {rank_str}")
    
    # 4. 统计结果
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    
    total = len(results['details'])
    print(f"\nTop 5:  {results['top5']}/{total} = {results['top5']/total*100:.1f}%")
    print(f"Top 10: {results['top10']}/{total} = {results['top10']/total*100:.1f}%")
    print(f"Top 15: {results['top15']}/{total} = {results['top15']/total*100:.1f}%")
    print(f"Top 20: {results['top20']}/{total} = {results['top20']/total*100:.1f}% ⭐")
    
    # 5. 对比分析
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)
    
    random_top15 = 15 / 49 * 100
    random_top20 = 20 / 49 * 100
    
    actual_top15 = results['top15'] / total * 100
    actual_top20 = results['top20'] / total * 100
    
    improvement_15 = actual_top15 / random_top15
    improvement_20 = actual_top20 / random_top20
    
    print(f"\nTop 15:")
    print(f"  理论随机: {random_top15:.1f}%")
    print(f"  实际成功: {actual_top15:.1f}%")
    print(f"  提升倍数: {improvement_15:.2f}x")
    
    print(f"\nTop 20:")
    print(f"  理论随机: {random_top20:.1f}%")
    print(f"  实际成功: {actual_top20:.1f}%")
    print(f"  提升倍数: {improvement_20:.2f}x")
    
    if actual_top20 > actual_top15:
        improvement = actual_top20 - actual_top15
        print(f"\n✅ Top 20相比Top 15提升: +{improvement:.1f}%")
    else:
        print(f"\n⚠️ Top 20未能提升Top 15成功率")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    validate_top20_strategy()
