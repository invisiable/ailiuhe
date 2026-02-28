"""
集成多生肖预测器 - 融合方案
通过投票和加权机制融合多个优秀预测器，提升预测准确率
目标：保持TOP4投注，将最大连续不中降低到6期左右
"""

import numpy as np
from collections import Counter, defaultdict
from retrained_zodiac_predictor import RetrainedZodiacPredictor
from adaptive_zodiac_predictor import AdaptiveZodiacPredictor
from zodiac_final_predictor import ZodiacFinalPredictor


class EnsembleMultiZodiacPredictor:
    """
    集成多生肖预测器
    
    策略：
    1. 使用3个经过验证的优秀预测器
    2. 通过加权投票选出TOP4
    3. 根据历史表现动态调整权重
    """
    
    def __init__(self):
        """初始化集成预测器"""
        # 初始化多个预测器
        self.predictor1 = RetrainedZodiacPredictor()  # 重训练v2.0
        self.predictor2 = AdaptiveZodiacPredictor()   # 自适应预测器
        self.predictor3 = ZodiacFinalPredictor()      # 最终版预测器
        
        # 预测器权重（初始相等）
        self.weights = {
            'predictor1': 0.35,  # 重训练v2.0 - 稳定性好
            'predictor2': 0.30,  # 自适应 - 适应性强
            'predictor3': 0.35,  # 最终版 - 综合优化
        }
        
        # 性能跟踪
        self.predictor_hits = {
            'predictor1': 0,
            'predictor2': 0,
            'predictor3': 0,
        }
        self.predictor_total = {
            'predictor1': 0,
            'predictor2': 0,
            'predictor3': 0,
        }
        
        # 所有生肖
        self.all_zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
    
    def predict_top4(self, animals):
        """
        集成预测TOP4生肖
        
        Args:
            animals: 历史生肖列表
            
        Returns:
            dict: {
                'top4': TOP4生肖列表,
                'predictor': '集成预测器',
                'details': 各预测器结果
            }
        """
        if len(animals) < 10:
            # 数据不足时使用简单频率
            freq = Counter(animals)
            top4 = [z for z, _ in freq.most_common(4)]
            if len(top4) < 4:
                remaining = [z for z in self.all_zodiacs if z not in top4]
                top4.extend(remaining[:4-len(top4)])
            return {
                'top4': top4[:4],
                'predictor': '集成预测器(频率法)',
                'details': {}
            }
        
        # 获取各预测器的预测结果
        pred1 = self.predictor1.predict_from_history(animals, top_n=6, debug=False)
        pred2 = self.predictor2.predict_from_history(animals, top_n=6, debug=False)
        pred3_result = self.predictor3.predict_from_history(animals, top_n=6, debug=False)
        
        # 提取TOP6（为了有更多选择）
        top6_1 = pred1.get('top6', pred1.get('top4', []))[:6]
        top6_2 = pred2.get('top6', pred2.get('top4', []))[:6]
        
        # ZodiacFinalPredictor返回tuple: (top_zodiacs, sorted_zodiacs)
        if isinstance(pred3_result, tuple):
            top6_3 = pred3_result[0][:6]  # 第一个元素是top_zodiacs
        else:
            top6_3 = pred3_result.get('top6', pred3_result.get('top4', []))[:6]
        
        # 加权投票
        scores = defaultdict(float)
        
        # 预测器1的贡献
        for i, zodiac in enumerate(top6_1):
            # 排名越前，分数越高（6, 5, 4, 3, 2, 1）
            score = (6 - i) * self.weights['predictor1']
            scores[zodiac] += score
        
        # 预测器2的贡献
        for i, zodiac in enumerate(top6_2):
            score = (6 - i) * self.weights['predictor2']
            scores[zodiac] += score
        
        # 预测器3的贡献
        for i, zodiac in enumerate(top6_3):
            score = (6 - i) * self.weights['predictor3']
            scores[zodiac] += score
        
        # 按总分排序选出TOP4
        sorted_zodiacs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top4 = [zodiac for zodiac, _ in sorted_zodiacs[:4]]
        
        # 确保有4个
        if len(top4) < 4:
            remaining = [z for z in self.all_zodiacs if z not in top4]
            top4.extend(remaining[:4-len(top4)])
        
        return {
            'top4': top4[:4],
            'predictor': '集成预测器(加权投票)',
            'details': {
                'pred1_top6': top6_1,
                'pred2_top6': top6_2,
                'pred3_top6': top6_3,
                'weights': self.weights.copy()
            }
        }
    
    def update_performance(self, predictions, actual):
        """
        更新性能统计并动态调整权重
        
        Args:
            predictions: 各预测器的预测结果字典
            actual: 实际开出的生肖
        """
        # 更新各预测器的命中统计
        if 'pred1_top6' in predictions:
            self.predictor_total['predictor1'] += 1
            if actual in predictions['pred1_top6'][:4]:  # 只看TOP4
                self.predictor_hits['predictor1'] += 1
        
        if 'pred2_top6' in predictions:
            self.predictor_total['predictor2'] += 1
            if actual in predictions['pred2_top6'][:4]:
                self.predictor_hits['predictor2'] += 1
        
        if 'pred3_top6' in predictions:
            self.predictor_total['predictor3'] += 1
            if actual in predictions['pred3_top6'][:4]:
                self.predictor_hits['predictor3'] += 1
        
        # 每20期调整一次权重
        total_predictions = sum(self.predictor_total.values())
        if total_predictions > 0 and total_predictions % 20 == 0:
            self._adjust_weights()
    
    def _adjust_weights(self):
        """根据历史表现动态调整权重"""
        # 计算各预测器的命中率
        rates = {}
        for key in ['predictor1', 'predictor2', 'predictor3']:
            if self.predictor_total[key] > 0:
                rates[key] = self.predictor_hits[key] / self.predictor_total[key]
            else:
                rates[key] = 0.33  # 默认
        
        # 归一化权重（表现好的权重高）
        total_rate = sum(rates.values())
        if total_rate > 0:
            for key in self.weights:
                # 80%根据表现，20%保持基础权重
                performance_weight = rates[key] / total_rate
                self.weights[key] = 0.8 * performance_weight + 0.2 * (1/3)
    
    def get_stats(self):
        """获取统计信息"""
        stats = {}
        for key in ['predictor1', 'predictor2', 'predictor3']:
            total = self.predictor_total[key]
            hits = self.predictor_hits[key]
            rate = (hits / total * 100) if total > 0 else 0
            stats[key] = {
                'hits': hits,
                'total': total,
                'rate': rate,
                'weight': self.weights[key]
            }
        return stats


def test_ensemble_predictor():
    """测试集成预测器"""
    import pandas as pd
    
    print("="*80)
    print("集成多生肖预测器测试")
    print("="*80 + "\n")
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    predictor = EnsembleMultiZodiacPredictor()
    
    # 测试最近10期
    test_periods = 10
    start_idx = len(df) - test_periods
    
    print(f"测试最近{test_periods}期:\n")
    
    hits = 0
    for i in range(start_idx, len(df)):
        period = i + 1
        history = df['animal'].iloc[:i].tolist()
        
        # 预测
        result = predictor.predict_top4(history)
        top4 = result['top4']
        
        # 实际
        actual = df.iloc[i]['animal']
        is_hit = actual in top4
        
        if is_hit:
            hits += 1
        
        # 更新性能
        if 'details' in result:
            predictor.update_performance(result['details'], actual)
        
        # 显示
        status = "✓ 命中" if is_hit else "✗ 未中"
        print(f"第{period}期: 预测{top4} | 实际:{actual} | {status}")
    
    # 显示统计
    print(f"\n命中率: {hits}/{test_periods} = {hits/test_periods*100:.2f}%")
    
    stats = predictor.get_stats()
    print(f"\n各预测器表现:")
    for key, data in stats.items():
        print(f"  {key}: {data['hits']}/{data['total']} = {data['rate']:.1f}%, 权重={data['weight']:.3f}")


if __name__ == '__main__':
    test_ensemble_predictor()
