"""
生肖预测器 v9.0 - 智能模型选择器
根据近期数据特征动态选择最佳预测模型
"""

import pandas as pd
import numpy as np
from collections import Counter

class ZodiacSmartSelector:
    """智能生肖预测器 - 动态模型选择"""
    
    def __init__(self):
        self.zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        self.zodiac_numbers = {
            '鼠': [1, 13, 25, 37, 49], '牛': [2, 14, 26, 38],
            '虎': [3, 15, 27, 39], '兔': [4, 16, 28, 40],
            '龙': [5, 17, 29, 41], '蛇': [6, 18, 30, 42],
            '马': [7, 19, 31, 43], '羊': [8, 20, 32, 44],
            '猴': [9, 21, 33, 45], '鸡': [10, 22, 34, 46],
            '狗': [11, 23, 35, 47], '猪': [12, 24, 36, 48]
        }
        
        # 定义5个不同特征的预测模型
        self.models = {
            'ultra_cold': {  # 极致冷门模型（单独49%）
                'name': '极致冷门模型',
                'weights': {
                    'ultra_cold': 0.45, 'anti_hot': 0.25, 'gap': 0.15,
                    'rotation': 0.08, 'absence_penalty': 0.04, 'diversity': 0.02, 'similarity': 0.01
                },
                '适用场景': '生肖分布均匀，无明显热门'
            },
            'balanced': {  # 平衡模型v5.0（整体最优52%）
                'name': '平衡模型',
                'weights': {
                    'ultra_cold': 0.35, 'anti_hot': 0.20, 'gap': 0.18,
                    'rotation': 0.12, 'absence_penalty': 0.08, 'diversity': 0.04, 'similarity': 0.03
                },
                '适用场景': '标准场景，整体最优'
            },
            'v5_ultra_hybrid': {  # v5.0和极致冷门的混合（新增）
                'name': 'v5混合增强',
                'weights': {
                    'ultra_cold': 0.40, 'anti_hot': 0.22, 'gap': 0.16,
                    'rotation': 0.10, 'absence_penalty': 0.06, 'diversity': 0.03, 'similarity': 0.03
                },
                '适用场景': 'v5和冷门策略混合'
            },
            'gap_focus': {  # 间隔主导模型
                'name': '间隔主导模型',
                'weights': {
                    'ultra_cold': 0.25, 'anti_hot': 0.15, 'gap': 0.30,
                    'rotation': 0.20, 'absence_penalty': 0.05, 'diversity': 0.03, 'similarity': 0.02
                },
                '适用场景': '生肖出现有规律间隔'
            },
            'hot_aware': {  # 热门感知模型
                'name': '热门感知模型',
                'weights': {
                    'ultra_cold': 0.20, 'anti_hot': 0.08, 'gap': 0.22,
                    'rotation': 0.15, 'absence_penalty': 0.05, 'hot_momentum': 0.22, 'diversity': 0.05, 'similarity': 0.03
                },
                '适用场景': '存在明显热门生肖'
            },
            'diversity': {  # 多样性模型
                'name': '多样性模型',
                'weights': {
                    'ultra_cold': 0.28, 'anti_hot': 0.18, 'gap': 0.15,
                    'rotation': 0.10, 'absence_penalty': 0.10, 'diversity': 0.12, 'similarity': 0.07
                },
                '适用场景': '生肖多样性高，分散出现'
            }
        }
    
    def _analyze_recent_pattern(self, animals, window=20):
        """
        分析最近期数的数据特征
        返回特征向量，用于选择最佳模型
        """
        if len(animals) < window:
            recent = animals
        else:
            recent = animals[-window:]
        
        counter = Counter(recent)
        
        # 特征1: 多样性（不同生肖数量）
        diversity = len(counter) / 12  # 0-1之间
        
        # 特征2: 集中度（热门生肖占比）
        hot_count = sum(1 for count in counter.values() if count >= 3)
        concentration = sum(count for zodiac, count in counter.items() if count >= 3) / len(recent) if counter else 0
        
        # 特征3: 均匀度（方差）
        counts = list(counter.values())
        variance = np.var(counts) if len(counts) > 1 else 0
        uniformity = 1 / (1 + variance)  # 归一化
        
        # 特征4: 重复率
        repeat_rate = sum(1 for count in counter.values() if count >= 2) / len(counter) if counter else 0
        
        # 特征5: 间隔规律性（计算主要生肖的间隔标准差）
        gap_regularity = 0
        for zodiac in self.zodiacs:
            indices = [i for i, a in enumerate(animals) if a == zodiac]
            if len(indices) >= 3:
                gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
                if gaps:
                    gap_std = np.std(gaps)
                    gap_regularity += 1 / (1 + gap_std)
        gap_regularity = gap_regularity / 12
        
        # 特征6: 最近趋势（最近10期 vs 前10期）
        if len(animals) >= 20:
            recent_10 = Counter(animals[-10:])
            prev_10 = Counter(animals[-20:-10])
            
            # 计算变化幅度
            changes = []
            for zodiac in self.zodiacs:
                change = abs(recent_10.get(zodiac, 0) - prev_10.get(zodiac, 0))
                changes.append(change)
            trend_change = np.mean(changes)
        else:
            trend_change = 0
        
        features = {
            'diversity': diversity,           # 多样性：高=更多不同生肖
            'concentration': concentration,   # 集中度：高=少数生肖频繁出现
            'uniformity': uniformity,         # 均匀度：高=各生肖出现次数接近
            'repeat_rate': repeat_rate,       # 重复率：高=多个生肖重复出现
            'gap_regularity': gap_regularity, # 间隔规律：高=生肖间隔有规律
            'trend_change': trend_change,     # 趋势变化：高=最近模式变化大
            'hot_count': hot_count            # 热门数量
        }
        
        return features
    
    def _select_best_model(self, features, debug=False):
        """
        根据特征选择最佳模型 - v4优化
        数据显示多样性模型在动态选择中命中率53.3%最高，应优先使用
        """
        scores = {}
        
        # 模型1: diversity - 动态选择中表现最佳53.3%，设为主力
        # 降低触发门槛，扩大使用范围
        scores['diversity'] = (
            70.0 +  # 最高基础分，优先选择
            features['diversity'] * 30 +
            (1 - features['concentration']) * 25 +
            features['trend_change'] * 15
        )
        
        # 模型2: v5_ultra_hybrid - 混合增强，单独49%，作为次选
        scores['v5_ultra_hybrid'] = 0.0
        if features['uniformity'] > 0.68 and 0.60 <= features['diversity'] <= 0.75:
            scores['v5_ultra_hybrid'] = (
                65.0 +
                features['uniformity'] * 25 +
                (1 - abs(features['diversity'] - 0.68)) * 30 +
                (1 - features['concentration']) * 15
            )
        
        # 模型3: ultra_cold - 极致冷门，单独49%，特定场景
        scores['ultra_cold'] = 0.0
        if features['uniformity'] > 0.75 and features['diversity'] > 0.75 and features['concentration'] < 0.25:
            scores['ultra_cold'] = (
                features['uniformity'] * 35 +
                features['diversity'] * 30 +
                (1 - features['concentration']) * 30 +
                58.0
            )
        
        # 模型4: balanced - v5.0原版（52%），仅标准场景
        scores['balanced'] = 0.0
        if 0.62 <= features['diversity'] <= 0.72 and features['uniformity'] > 0.70:
            scores['balanced'] = (
                features['uniformity'] * 28 +
                (1 - abs(features['diversity'] - 0.67)) * 35 +
                (1 - features['concentration']) * 18 +
                62.0
            )
        
        # 模型5: gap_focus - 间隔规律
        scores['gap_focus'] = 0.0
        if features['gap_regularity'] > 0.20:
            scores['gap_focus'] = (
                features['gap_regularity'] * 150 +
                features['uniformity'] * 18 +
                48.0
            )
        
        # 模型6: hot_aware - 热门场景，最谨慎使用
        scores['hot_aware'] = 0.0
        if features['concentration'] > 0.55 and features['hot_count'] >= 4:
            scores['hot_aware'] = (
                features['concentration'] * 40 +
                features['hot_count'] * 8 +
                features['repeat_rate'] * 25 +
                30.0  # 最低基础分
            )
        
        # 选择得分最高的模型
        best_model = max(scores.items(), key=lambda x: x[1])
        
        if debug:
            print(f"\n{'='*90}")
            print("数据特征分析（最近20期）:")
            print(f"{'='*90}")
            for feat, value in features.items():
                print(f"  {feat:20s}: {value:.3f}")
            print(f"\n模型适配度评分:")
            for model, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                marker = " <-- 已选择" if model == best_model[0] else ""
                model_name = self.models[model]['name'] if model in self.models else model
                print(f"  {model_name:15s}: {score:.3f}{marker}")
            print(f"{'='*90}\n")
        
        return best_model[0], scores
    
    def _ultra_cold_strategy(self, animals):
        """极致冷门策略"""
        scores = {}
        counter = Counter(animals[-50:] if len(animals) >= 50 else animals)
        min_count = min(counter.values()) if counter else 0
        max_count = max(counter.values()) if counter else 1
        
        for zodiac in self.zodiacs:
            count = counter.get(zodiac, 0)
            if max_count > min_count:
                normalized = (count - min_count) / (max_count - min_count)
                scores[zodiac] = (1 - normalized) * 100
            else:
                scores[zodiac] = 50.0
        return scores
    
    def _anti_hot_strategy(self, animals):
        """反热门策略"""
        scores = {}
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        for zodiac in self.zodiacs:
            count = recent_10.count(zodiac)
            if count == 0:
                scores[zodiac] = 20.0
            elif count == 1:
                scores[zodiac] = 10.0
            else:
                scores[zodiac] = -5.0 * count
        return scores
    
    def _hot_momentum_strategy(self, animals):
        """热门惯性策略"""
        scores = {}
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        
        for zodiac in self.zodiacs:
            score = 0.0
            count_20 = recent_20.count(zodiac)
            if count_20 >= 4:
                score += 40.0
            elif count_20 == 3:
                score += 28.0
            elif count_20 == 2:
                score += 16.0
            elif count_20 == 1:
                score += 6.0
            
            count_30 = recent_30.count(zodiac)
            if count_30 >= 5:
                score += 12.0
            elif count_30 >= 3:
                score += 6.0
            scores[zodiac] = score
        return scores
    
    def _gap_analysis(self, animals):
        """间隔分析"""
        scores = {}
        for zodiac in self.zodiacs:
            try:
                last_idx = len(animals) - 1 - animals[::-1].index(zodiac)
                gap = len(animals) - last_idx - 1
                if gap >= 15:
                    scores[zodiac] = 10.0
                elif gap >= 10:
                    scores[zodiac] = 7.0
                elif gap >= 6:
                    scores[zodiac] = 3.0
                elif gap >= 3:
                    scores[zodiac] = 1.0
                else:
                    scores[zodiac] = -3.0
            except ValueError:
                scores[zodiac] = 5.0
        return scores
    
    def _rotation_advanced(self, animals):
        """轮转规律"""
        scores = {}
        for zodiac in self.zodiacs:
            try:
                indices = [i for i, a in enumerate(animals) if a == zodiac]
                if len(indices) >= 2:
                    gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
                    avg_gap = sum(gaps) / len(gaps)
                    current_gap = len(animals) - indices[-1]
                    if abs(current_gap - avg_gap) <= 2:
                        scores[zodiac] = 10.0
                    elif abs(current_gap - avg_gap) <= 5:
                        scores[zodiac] = 5.0
                    else:
                        scores[zodiac] = 0.0
                else:
                    scores[zodiac] = 0.0
            except:
                scores[zodiac] = 0.0
        return scores
    
    def _continuous_absence_penalty(self, animals):
        """连续不出现惩罚"""
        scores = {}
        for zodiac in self.zodiacs:
            try:
                last_idx = len(animals) - 1 - animals[::-1].index(zodiac)
                last_appearance = len(animals) - last_idx - 1
            except ValueError:
                last_appearance = len(animals)
            
            if last_appearance >= 20:
                scores[zodiac] = -3.0
            elif last_appearance >= 15:
                scores[zodiac] = -1.0
            elif last_appearance >= 10:
                scores[zodiac] = 3.0
            elif last_appearance >= 6:
                scores[zodiac] = 3.0
            elif last_appearance >= 4:
                scores[zodiac] = 1.0
            else:
                scores[zodiac] = -1.0
        return scores
    
    def _diversity_boost(self, animals):
        """多样性提升"""
        scores = {}
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        appeared = set(recent_20)
        for zodiac in self.zodiacs:
            if zodiac not in appeared:
                scores[zodiac] = 10.0
            else:
                count = recent_20.count(zodiac)
                if count == 1:
                    scores[zodiac] = 5.0
                else:
                    scores[zodiac] = -2.0
        return scores
    
    def _historical_similarity(self, animals):
        """历史相似性"""
        scores = {}
        if len(animals) < 5:
            return {z: 0 for z in self.zodiacs}
        
        recent_5 = animals[-5:]
        for zodiac in self.zodiacs:
            score = 0.0
            for i in range(len(animals) - 10):
                window = animals[i:i+5]
                similarity = sum(1 for a, b in zip(recent_5, window) if a == b)
                if similarity >= 3 and len(animals) > i + 5:
                    next_animal = animals[i + 5]
                    if next_animal == zodiac:
                        score += 5.0 * similarity
            scores[zodiac] = score
        return scores
    
    def predict_from_history(self, animals, top_n=5, debug=False):
        """智能预测"""
        
        # 1. 分析数据特征
        features = self._analyze_recent_pattern(animals)
        
        # 2. 选择最佳模型
        best_model_name, model_scores = self._select_best_model(features, debug)
        selected_model = self.models[best_model_name]
        weights = selected_model['weights']
        
        # 3. 收集各策略评分
        strategies_scores = {
            'ultra_cold': self._ultra_cold_strategy(animals),
            'anti_hot': self._anti_hot_strategy(animals),
            'gap': self._gap_analysis(animals),
            'rotation': self._rotation_advanced(animals),
            'absence_penalty': self._continuous_absence_penalty(animals),
            'diversity': self._diversity_boost(animals),
            'similarity': self._historical_similarity(animals)
        }
        
        # 如果模型需要热门惯性策略
        if 'hot_momentum' in weights:
            strategies_scores['hot_momentum'] = self._hot_momentum_strategy(animals)
        
        # 4. 加权融合
        final_scores = {}
        for zodiac in self.zodiacs:
            score = 0.0
            for strategy_name, strategy_scores in strategies_scores.items():
                weight = weights.get(strategy_name, 0)
                score += strategy_scores.get(zodiac, 0) * weight
            final_scores[zodiac] = score
        
        # 5. 排序
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_zodiacs = [z for z, s in sorted_zodiacs[:top_n]]
        
        return {
            'top5': top_zodiacs,
            'all_scores': sorted_zodiacs,
            'selected_model': best_model_name,
            'model_name': selected_model['name'],
            'features': features,
            'model_scores': model_scores
        }
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_n=5):
        """标准预测接口"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        result = self.predict_from_history(animals, top_n, debug=True)
        
        # 推荐号码
        recommended_numbers = []
        for rank, (zodiac, score) in enumerate(result['all_scores'][:top_n], 1):
            weight = top_n + 1 - rank
            for num in self.zodiac_numbers[zodiac]:
                recommended_numbers.append((num, weight))
        
        num_scores = {}
        for num, w in recommended_numbers:
            num_scores[num] = num_scores.get(num, 0) + w
        
        sorted_nums = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_nums[:15]]
        
        return {
            'model': '智能模型选择器',
            'version': '9.0',
            'selected_model': result['model_name'],
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_animal': df.iloc[-1]['animal'],
            'top5': result['top5'],
            'top15_numbers': top_numbers
        }
    
    def get_recent_20_validation(self, csv_file='data/lucky_numbers.csv'):
        """获取最近20期验证结果（GUI兼容）"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        total = len(df)
        
        results = []
        for i in range(max(0, total - 20), total):
            animals = [str(a).strip() for a in df['animal'].values[:i]]
            prediction = self.predict_from_history(animals, top_n=5)
            
            actual = str(df['animal'].values[i]).strip()
            period = df.iloc[i]['period']
            date = df.iloc[i]['date']
            
            hit_rank = 0
            if actual in prediction['top5']:
                hit_rank = prediction['top5'].index(actual) + 1
            
            results.append({
                'period': period,
                'date': date,
                'actual': actual,
                'predicted_top5': prediction['top5'],
                'hit': hit_rank > 0,
                'rank': hit_rank,
                'model': prediction['model_name']
            })
        
        return results

if __name__ == '__main__':
    predictor = ZodiacSmartSelector()
    result = predictor.predict()
    
    print(f"\n模型: {result['model']}")
    print(f"版本: {result['version']}")
    print(f"选择的模型: {result['selected_model']}")
    print(f"总期数: {result['total_periods']}")
    print(f"最新期: {result['last_date']} - {result['last_animal']}")
    print(f"\n预测TOP5: {', '.join(result['top5'])}")
    print(f"推荐号码TOP15: {result['top15_numbers']}")
