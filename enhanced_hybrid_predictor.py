"""
增强版混合策略预测器 - 结合奇偶预测
将固化混合策略与奇偶预测模型结合，提升预测成功率

策略设计：
- 基础层：固化混合策略（TOP15基础预测）
- 增强层：奇偶预测模型（筛选和排序）
- 最终输出：经过奇偶性过滤和优化的TOP15预测
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from final_hybrid_predictor import FinalHybridPredictor
from improved_odd_even_predictor import ImprovedOddEvenPredictor
import sys
from io import StringIO


class EnhancedHybridPredictor:
    """增强版混合策略预测器（结合奇偶预测）"""
    
    def __init__(self):
        self.hybrid_predictor = FinalHybridPredictor()
        self.odd_even_predictor = ImprovedOddEvenPredictor()
        self.version = "2.0-Enhanced"
        self.model_name = "混合策略+奇偶预测"
        
    def _train_odd_even_model(self, csv_file, silent=True):
        """训练奇偶预测模型"""
        if silent:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
        
        try:
            self.odd_even_predictor.train_model(
                csv_file=csv_file,
                model_type='ensemble_voting',
                test_size=0.2
            )
        finally:
            if silent:
                sys.stdout = old_stdout
    
    def _get_odd_even_prediction(self, csv_file, silent=True):
        """获取奇偶预测结果"""
        if silent:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
        
        try:
            result = self.odd_even_predictor.predict(csv_file)
        finally:
            if silent:
                sys.stdout = old_stdout
        
        return result
    
    def predict(self, csv_file='data/lucky_numbers.csv', use_odd_even=True):
        """
        生成增强预测
        
        参数:
            csv_file: 数据文件路径
            use_odd_even: 是否使用奇偶预测增强
        
        返回:
            dict: 包含多个维度的预测结果
        """
        # 1. 获取基础混合策略预测
        hybrid_top15 = self.hybrid_predictor.predict(csv_file)
        
        # 2. 如果不使用奇偶预测，直接返回
        if not use_odd_even:
            return {
                'top15': hybrid_top15,
                'top10': hybrid_top15[:10],
                'top5': hybrid_top15[:5],
                'method': 'hybrid_only',
                'odd_even_used': False
            }
        
        # 3. 训练并获取奇偶预测
        self._train_odd_even_model(csv_file, silent=True)
        odd_even_result = self._get_odd_even_prediction(csv_file, silent=True)
        
        predicted_parity = odd_even_result['prediction']  # '奇数' or '偶数'
        odd_probability = odd_even_result['odd_probability']
        even_probability = odd_even_result['even_probability']
        
        # 4. 根据奇偶预测对候选数字重新评分和排序
        enhanced_top15 = self._apply_odd_even_filter(
            hybrid_top15,
            predicted_parity,
            odd_probability,
            even_probability
        )
        
        # 5. 返回完整结果
        return {
            'top15': enhanced_top15,
            'top10': enhanced_top15[:10],
            'top5': enhanced_top15[:5],
            'method': 'hybrid_with_odd_even',
            'odd_even_used': True,
            'odd_even_prediction': {
                'predicted': predicted_parity,
                'odd_prob': odd_probability,
                'even_prob': even_probability,
                'confidence': max(odd_probability, even_probability)
            },
            'original_hybrid': hybrid_top15
        }
    
    def _apply_odd_even_filter(self, candidates, predicted_parity, odd_prob, even_prob):
        """
        应用奇偶性过滤和重排序
        
        策略：
        1. 如果奇偶预测置信度高（>60%），优先推荐对应奇偶性的数字
        2. 如果置信度中等（50-60%），适度调整顺序
        3. 保持混合策略的基础逻辑
        """
        confidence = max(odd_prob, even_prob)
        is_odd_predicted = (predicted_parity == '奇数')
        
        # 将候选数字分为奇数和偶数
        odd_candidates = [n for n in candidates if n % 2 == 1]
        even_candidates = [n for n in candidates if n % 2 == 0]
        
        # 根据置信度决定调整策略
        if confidence >= 0.65:
            # 高置信度：强调奇偶性
            if is_odd_predicted:
                # 优先推荐奇数
                # 分配比例：奇数10个，偶数5个
                enhanced = odd_candidates[:10] + even_candidates[:5]
            else:
                # 优先推荐偶数
                # 分配比例：偶数10个，奇数5个
                enhanced = even_candidates[:10] + odd_candidates[:5]
        
        elif confidence >= 0.55:
            # 中等置信度：适度调整
            if is_odd_predicted:
                # 分配比例：奇数9个，偶数6个
                enhanced = odd_candidates[:9] + even_candidates[:6]
            else:
                # 分配比例：偶数9个，奇数6个
                enhanced = even_candidates[:9] + odd_candidates[:6]
        
        else:
            # 低置信度：轻微调整，保持原顺序
            if is_odd_predicted:
                # 分配比例：奇数8个，偶数7个
                enhanced = odd_candidates[:8] + even_candidates[:7]
            else:
                # 分配比例：偶数8个，奇数7个
                enhanced = even_candidates[:8] + odd_candidates[:7]
        
        # 确保返回15个数字
        enhanced = enhanced[:15]
        
        # 如果不够15个，从原候选中补充
        if len(enhanced) < 15:
            for num in candidates:
                if num not in enhanced:
                    enhanced.append(num)
                if len(enhanced) >= 15:
                    break
        
        return enhanced[:15]
    
    def get_prediction_info(self, csv_file='data/lucky_numbers.csv'):
        """获取完整预测信息"""
        # 获取基础预测信息
        base_info = self.hybrid_predictor.get_prediction_info(csv_file)
        
        # 训练并获取奇偶预测
        self._train_odd_even_model(csv_file, silent=True)
        odd_even_result = self._get_odd_even_prediction(csv_file, silent=True)
        
        # 增强信息
        base_info['model_name'] = self.model_name
        base_info['version'] = self.version
        base_info['odd_even_prediction'] = {
            'predicted': odd_even_result['prediction'],
            'odd_probability': f"{odd_even_result['odd_probability']*100:.1f}%",
            'even_probability': f"{odd_even_result['even_probability']*100:.1f}%",
            'confidence': f"{max(odd_even_result['odd_probability'], odd_even_result['even_probability'])*100:.1f}%"
        }
        
        return base_info
    
    def predict_with_analysis(self, csv_file='data/lucky_numbers.csv'):
        """
        生成带详细分析的预测
        """
        # 获取预测结果
        prediction = self.predict(csv_file, use_odd_even=True)
        
        # 获取上下文信息
        info = self.get_prediction_info(csv_file)
        
        # 读取数据进行额外分析
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        recent_10 = df.tail(10)['number'].tolist()
        
        # 分析预测数字的奇偶分布
        top15 = prediction['top15']
        odd_count = sum(1 for n in top15 if n % 2 == 1)
        even_count = 15 - odd_count
        
        # 统计最近10期的奇偶情况
        recent_odd = sum(1 for n in recent_10 if n % 2 == 1)
        recent_even = 10 - recent_odd
        
        analysis = {
            'prediction': prediction,
            'info': info,
            'statistics': {
                'predicted_distribution': {
                    'odd_count': odd_count,
                    'even_count': even_count,
                    'odd_ratio': f"{odd_count/15*100:.1f}%",
                    'even_ratio': f"{even_count/15*100:.1f}%"
                },
                'recent_10_distribution': {
                    'odd_count': recent_odd,
                    'even_count': recent_even,
                    'odd_ratio': f"{recent_odd/10*100:.1f}%",
                    'even_ratio': f"{recent_even/10*100:.1f}%"
                }
            }
        }
        
        return analysis


if __name__ == "__main__":
    print("="*80)
    print("增强版混合策略预测器 - 结合奇偶预测")
    print("="*80)
    
    predictor = EnhancedHybridPredictor()
    
    # 测试1: 不使用奇偶预测
    print("\n【测试1】基础混合策略（不使用奇偶预测）")
    print("-"*80)
    result1 = predictor.predict(use_odd_even=False)
    print(f"TOP15: {result1['top15']}")
    print(f"TOP10: {result1['top10']}")
    print(f"TOP5:  {result1['top5']}")
    
    # 测试2: 使用奇偶预测
    print("\n【测试2】增强策略（使用奇偶预测）")
    print("-"*80)
    result2 = predictor.predict(use_odd_even=True)
    print(f"奇偶预测: {result2['odd_even_prediction']['predicted']} " +
          f"(置信度: {result2['odd_even_prediction']['confidence']*100:.1f}%)")
    print(f"  奇数概率: {result2['odd_even_prediction']['odd_prob']*100:.1f}%")
    print(f"  偶数概率: {result2['odd_even_prediction']['even_prob']*100:.1f}%")
    print(f"\nTOP15: {result2['top15']}")
    print(f"TOP10: {result2['top10']}")
    print(f"TOP5:  {result2['top5']}")
    
    # 测试3: 完整分析
    print("\n【测试3】完整预测分析")
    print("-"*80)
    analysis = predictor.predict_with_analysis()
    
    print(f"模型: {analysis['info']['model_name']} v{analysis['info']['version']}")
    print(f"当前期数: {analysis['info']['latest_period']['date']}")
    print(f"最近数字: {analysis['info']['latest_period']['number']}")
    
    print(f"\n奇偶预测: {analysis['info']['odd_even_prediction']['predicted']}")
    print(f"  奇数: {analysis['info']['odd_even_prediction']['odd_probability']}")
    print(f"  偶数: {analysis['info']['odd_even_prediction']['even_probability']}")
    print(f"  置信度: {analysis['info']['odd_even_prediction']['confidence']}")
    
    print(f"\nTOP15预测分布:")
    print(f"  奇数: {analysis['statistics']['predicted_distribution']['odd_count']}个 " +
          f"({analysis['statistics']['predicted_distribution']['odd_ratio']})")
    print(f"  偶数: {analysis['statistics']['predicted_distribution']['even_count']}个 " +
          f"({analysis['statistics']['predicted_distribution']['even_ratio']})")
    
    print(f"\n最近10期分布:")
    print(f"  奇数: {analysis['statistics']['recent_10_distribution']['odd_count']}个 " +
          f"({analysis['statistics']['recent_10_distribution']['odd_ratio']})")
    print(f"  偶数: {analysis['statistics']['recent_10_distribution']['even_count']}个 " +
          f"({analysis['statistics']['recent_10_distribution']['even_ratio']})")
    
    print(f"\nTOP15: {analysis['prediction']['top15']}")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
