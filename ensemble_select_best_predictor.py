"""
集成生肖预测器 v2 - 择优选择策略
不是投票融合，而是动态选择当前表现最好的预测器
"""

import numpy as np
from collections import Counter, defaultdict, deque
from retrained_zodiac_predictor import RetrainedZodiacPredictor
from adaptive_zodiac_predictor import AdaptiveZodiacPredictor
from zodiac_simple_smart import ZodiacSimpleSmart


class EnsembleSelectBestPredictor:
    """
    集成预测器v2 - 择优选择策略
    
    策略：
    1. 跟踪3个预测器的近期表现
    2. 每次选择近期表现最好的预测器
    3. 如果当前预测器连续失误，立即切换
    """
    
    def __init__(self, window_size=20):
        """初始化集成预测器"""
        # 初始化多个预测器
        self.predictor1 = RetrainedZodiacPredictor()  # 重训练v2.0
        self.predictor2 = AdaptiveZodiacPredictor()   # 自适应预测器
        self.predictor3 = ZodiacSimpleSmart()          # 简单智能v10
        
        self.predictors = {
            'retrained': self.predictor1,
            'adaptive': self.predictor2,
            'simple_smart': self.predictor3
        }
        
        # 性能跟踪窗口
        self.window_size = window_size
        self.recent_hits = {
            'retrained': deque(maxlen=window_size),
            'adaptive': deque(maxlen=window_size),
            'simple_smart': deque(maxlen=window_size)
        }
        
        # 当前选择的预测器
        self.current_predictor = 'retrained'  # 默认使用重训练版
        self.consecutive_misses = 0
        self.SWITCH_THRESHOLD = 1  # 连续1次不中就切换（最激进）
        
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
                'predictor': 使用的预测器名称,
                'details': 详细信息(包含所有预测器的结果)
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
                'predictor': '简单频率法',
                'details': {}
            }
        
        # 获取所有预测器的预测结果（用于跟踪性能）
        all_predictions = {}
        for name, predictor in self.predictors.items():
            try:
                result = predictor.predict_from_history(animals, top_n=4, debug=False)
                
                # 处理不同的返回格式
                if isinstance(result, dict):
                    top4 = result.get('top4', result.get('top5', []))[:4]
                elif isinstance(result, tuple):
                    top4 = result[0][:4]
                elif isinstance(result, list):
                    top4 = result[:4]
                else:
                    top4 = []
                
                # 确保有4个
                if len(top4) < 4:
                    remaining = [z for z in self.all_zodiacs if z not in top4]
                    top4.extend(remaining[:4-len(top4)])
                
                all_predictions[name] = top4[:4]
                
            except Exception as e:
                print(f"预测器 {name} 出错: {e}")
                # 使用频率法作为降级
                freq = Counter(animals)
                top4 = [z for z, _ in freq.most_common(4)]
                if len(top4) < 4:
                    remaining = [z for z in self.all_zodiacs if z not in top4]
                    top4.extend(remaining[:4-len(top4)])
                all_predictions[name] = top4[:4]
        
        # 根据连续失误决定是否切换预测器
        if self.consecutive_misses >= self.SWITCH_THRESHOLD:
            self._switch_to_best_predictor()
        
        # 使用当前选择的预测器的结果
        top4 = all_predictions.get(self.current_predictor, list(all_predictions.values())[0])
        
        return {
            'top4': top4,
            'predictor': f'集成择优-{self.current_predictor}',
            'details': {
                'current_predictor': self.current_predictor,
                'consecutive_misses': self.consecutive_misses,
                'all_predictions': all_predictions  # 保存所有预测器的结果
            }
        }
    
    def _switch_to_best_predictor(self):
        """切换到近期表现最好的预测器"""
        # 计算各预测器的近期命中率
        best_rate = -1
        best_predictor = self.current_predictor
        min_records = 3  # 最少3次记录即可参与评估（降低门槛）
        
        candidates = {}
        for name, hits in self.recent_hits.items():
            if len(hits) >= min_records:
                rate = sum(hits) / len(hits)
                candidates[name] = rate
                if rate > best_rate:
                    best_rate = rate
                    best_predictor = name
        
        # 如果没有足够数据，随机尝试其他预测器
        if not candidates:
            # 轮流尝试
            predictor_names = list(self.predictors.keys())
            current_idx = predictor_names.index(self.current_predictor)
            best_predictor = predictor_names[(current_idx + 1) % len(predictor_names)]
            best_rate = 0
        
        # 如果找到更好的，切换
        if best_predictor != self.current_predictor:
            old = self.current_predictor
            self.current_predictor = best_predictor
            rate_str = f"{best_rate*100:.1f}%" if best_rate > 0 else "无数据"
            print(f"\n>>> 切换预测器: {old} → {best_predictor} (近期命中率: {rate_str})")
        
        # 重置连续失误计数
        self.consecutive_misses = 0
    
    def update_performance(self, actual, prediction_details):
        """
        更新性能统计 - 同时跟踪所有预测器的表现
        
        Args:
            actual: 实际开出的生肖
            prediction_details: 预测详细信息
        """
        if 'all_predictions' not in prediction_details:
            return
        
        # 更新所有预测器的记录
        all_preds = prediction_details['all_predictions']
        for name, top4 in all_preds.items():
            was_hit = actual in top4
            self.recent_hits[name].append(1 if was_hit else 0)
        
        # 更新连续失误计数（仅针对当前使用的预测器）
        if 'current_predictor' in prediction_details:
            predictor_name = prediction_details['current_predictor']
            top4 = all_preds.get(predictor_name, [])
            was_hit = actual in top4
            
            if was_hit:
                self.consecutive_misses = 0
            else:
                self.consecutive_misses += 1
    
    def get_stats(self):
        """获取统计信息"""
        stats = {}
        for name, hits in self.recent_hits.items():
            if len(hits) > 0:
                total = len(hits)
                hit_count = sum(hits)
                rate = hit_count / total * 100
                stats[name] = {
                    'hits': hit_count,
                    'total': total,
                    'rate': rate,
                    'is_current': (name == self.current_predictor)
                }
        return stats


def test_ensemble_select_best():
    """测试择优选择集成预测器"""
    import pandas as pd
    
    print("="*80)
    print("集成择优选择预测器测试")
    print("="*80 + "\n")
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    predictor = EnsembleSelectBestPredictor(window_size=20)
    
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
        details = result.get('details', {})
        details['prediction'] = top4
        predictor.update_performance(actual, details)
        
        # 显示
        status = "✓ 命中" if is_hit else "✗ 未中"
        print(f"第{period}期: {result['predictor']:30s} 预测{top4} | 实际:{actual} | {status}")
    
    # 显示统计
    print(f"\n命中率: {hits}/{test_periods} = {hits/test_periods*100:.2f}%")
    
    stats = predictor.get_stats()
    print(f"\n各预测器近期表现:")
    for name, data in sorted(stats.items()):
        current_mark = " ← 当前使用" if data['is_current'] else ""
        print(f"  {name:15s}: {data['hits']}/{data['total']} = {data['rate']:.1f}%{current_mark}")


if __name__ == '__main__':
    test_ensemble_select_best()
