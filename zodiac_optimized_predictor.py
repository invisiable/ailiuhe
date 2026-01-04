"""
生肖预测优化模型 - 专注TOP5命中率提升
结合多种策略，优化评分算法，提升预测准确性

优化策略：
1. 增强特征工程（150+维特征）
2. 多策略融合（统计+ML+五行+号码分布）
3. 动态权重调整
4. 强化避重机制
5. 集成多个预测器投票
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class ZodiacOptimizedPredictor:
    """优化的生肖预测器 - 专注TOP5命中率"""
    
    def __init__(self):
        # 12生肖列表
        self.zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        # 生肖对应的号码映射
        self.zodiac_numbers = {
            '鼠': [1, 13, 25, 37, 49],
            '牛': [2, 14, 26, 38],
            '虎': [3, 15, 27, 39],
            '兔': [4, 16, 28, 40],
            '龙': [5, 17, 29, 41],
            '蛇': [6, 18, 30, 42],
            '马': [7, 19, 31, 43],
            '羊': [8, 20, 32, 44],
            '猴': [9, 21, 33, 45],
            '鸡': [10, 22, 34, 46],
            '狗': [11, 23, 35, 47],
            '猪': [12, 24, 36, 48]
        }
        
        # 生肖五行属性
        self.zodiac_elements = {
            '鼠': '水', '牛': '土', '虎': '木', '兔': '木',
            '龙': '土', '蛇': '火', '马': '火', '羊': '土',
            '猴': '金', '鸡': '金', '狗': '土', '猪': '水'
        }
        
        # 反向映射
        self.number_to_zodiac = {}
        for zodiac, numbers in self.zodiac_numbers.items():
            for num in numbers:
                self.number_to_zodiac[num] = zodiac
        
        self.model_name = "生肖优化预测模型(TOP5专用)"
        self.version = "3.0"
        
        # ML相关
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _enhanced_features(self, df, index):
        """增强特征提取 - 150+维"""
        features = {}
        
        animals = [str(a).strip() for a in df['animal'].values[:index]]
        numbers = df['number'].values[:index]
        
        if len(animals) < 15:
            return None
        
        # === 基础频率特征（60维） ===
        for window in [5, 10, 15, 20, 30]:
            recent = animals[-window:] if len(animals) >= window else animals
            counter = Counter(recent)
            for zodiac in self.zodiacs:
                features[f'freq_{window}_{zodiac}'] = counter.get(zodiac, 0)
        
        # === 间隔特征（12维） ===
        for zodiac in self.zodiacs:
            positions = [i for i, a in enumerate(animals) if a == zodiac]
            if positions:
                gap = len(animals) - 1 - positions[-1]
                features[f'gap_{zodiac}'] = gap
            else:
                features[f'gap_{zodiac}'] = 999
        
        # === 趋势特征（24维） ===
        for zodiac in self.zodiacs:
            # 近期趋势（最近20期 vs 之前20期）
            recent_20 = animals[-20:] if len(animals) >= 20 else animals
            prev_20 = animals[-40:-20] if len(animals) >= 40 else []
            
            recent_count = recent_20.count(zodiac)
            prev_count = prev_20.count(zodiac) if prev_20 else 0
            
            features[f'trend_{zodiac}'] = recent_count - prev_count
            features[f'momentum_{zodiac}'] = recent_count / (prev_count + 1)
        
        # === 周期特征（24维） ===
        for zodiac in self.zodiacs:
            positions = [i for i, a in enumerate(animals) if a == zodiac]
            if len(positions) >= 2:
                gaps = [positions[j+1] - positions[j] for j in range(len(positions)-1)]
                features[f'avg_cycle_{zodiac}'] = np.mean(gaps)
                features[f'std_cycle_{zodiac}'] = np.std(gaps) if len(gaps) > 1 else 0
            else:
                features[f'avg_cycle_{zodiac}'] = 0
                features[f'std_cycle_{zodiac}'] = 0
        
        # === 五行特征（15维） ===
        recent_15 = animals[-15:]
        element_counter = Counter([self.zodiac_elements.get(z, '') for z in recent_15])
        for element in ['金', '木', '水', '火', '土']:
            features[f'element_freq_{element}'] = element_counter.get(element, 0)
        
        # 五行缺失检测
        for element in ['金', '木', '水', '火', '土']:
            last_seen = -1
            for i in range(len(animals)-1, -1, -1):
                if self.zodiac_elements.get(animals[i], '') == element:
                    last_seen = len(animals) - 1 - i
                    break
            features[f'element_gap_{element}'] = last_seen if last_seen >= 0 else 999
        
        # === 号码分布特征（10维） ===
        recent_nums = numbers[-15:] if len(numbers) >= 15 else numbers
        features['avg_number'] = np.mean(recent_nums)
        features['std_number'] = np.std(recent_nums)
        features['min_number'] = np.min(recent_nums)
        features['max_number'] = np.max(recent_nums)
        
        # 号码范围分布
        features['small_count'] = sum(1 for n in recent_nums if n <= 12)
        features['mid_count'] = sum(1 for n in recent_nums if 13 <= n <= 36)
        features['large_count'] = sum(1 for n in recent_nums if n >= 37)
        
        # 奇偶分布
        features['odd_count'] = sum(1 for n in recent_nums if n % 2 == 1)
        features['even_count'] = sum(1 for n in recent_nums if n % 2 == 0)
        
        # 号码跳跃度
        if len(recent_nums) >= 2:
            jumps = [abs(recent_nums[i+1] - recent_nums[i]) for i in range(len(recent_nums)-1)]
            features['avg_jump'] = np.mean(jumps)
        else:
            features['avg_jump'] = 0
        
        # === 多样性特征（5维） ===
        for window in [5, 10, 15]:
            recent = animals[-window:] if len(animals) >= window else animals
            features[f'diversity_{window}'] = len(set(recent))
        
        # 重复度
        features['repeat_rate_5'] = 1 - len(set(animals[-5:])) / 5 if len(animals) >= 5 else 0
        features['repeat_rate_10'] = 1 - len(set(animals[-10:])) / 10 if len(animals) >= 10 else 0
        
        return features
    
    def _strategy_1_frequency_cold(self, animals):
        """策略1: 强化冷门分析"""
        scores = {}
        
        recent_50 = animals[-50:] if len(animals) >= 50 else animals
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        recent_15 = animals[-15:] if len(animals) >= 15 else animals
        recent_10 = animals[-10:]
        recent_5 = animals[-5:]
        
        freq_50 = Counter(recent_50)
        freq_30 = Counter(recent_30)
        freq_15 = Counter(recent_15)
        freq_10 = Counter(recent_10)
        freq_5 = Counter(recent_5)
        
        for zodiac in self.zodiacs:
            score = 0.0
            
            # 超级冷门加分（权重加强）
            f50 = freq_50.get(zodiac, 0)
            if f50 == 0:
                score += 8.0
            elif f50 == 1:
                score += 6.0
            elif f50 == 2:
                score += 4.0
            elif f50 == 3:
                score += 2.0
            
            # 中期冷门
            f30 = freq_30.get(zodiac, 0)
            if f30 == 0:
                score += 5.0
            elif f30 == 1:
                score += 3.5
            elif f30 == 2:
                score += 2.0
            
            # 近期冷门
            f15 = freq_15.get(zodiac, 0)
            if f15 == 0:
                score += 4.0
            elif f15 == 1:
                score += 2.5
            
            f10 = freq_10.get(zodiac, 0)
            if f10 == 0:
                score += 3.0
            
            # 最近5期强力避重
            f5 = freq_5.get(zodiac, 0)
            if f5 > 0:
                score -= 6.0 * f5  # 大幅惩罚
            else:
                score += 4.0  # 大幅奖励
            
            scores[zodiac] = score
        
        return scores
    
    def _strategy_2_rotation(self, animals):
        """策略2: 生肖轮转与相邻"""
        scores = {}
        
        if len(animals) < 2:
            return {z: 0.0 for z in self.zodiacs}
        
        last_zodiac = animals[-1]
        if last_zodiac not in self.zodiacs:
            return {z: 0.0 for z in self.zodiacs}
        
        last_idx = self.zodiacs.index(last_zodiac)
        
        for zodiac in self.zodiacs:
            score = 0.0
            zodiac_idx = self.zodiacs.index(zodiac)
            
            # 计算距离
            forward = (zodiac_idx - last_idx) % 12
            backward = (last_idx - zodiac_idx) % 12
            
            # 相邻生肖加分（优化权重）
            if forward in [1, 2]:
                score += 3.5
            elif forward == 3:
                score += 2.0
            elif forward in [4, 5]:
                score += 1.0
            
            if backward in [1, 2]:
                score += 3.0
            elif backward == 3:
                score += 1.5
            
            # 对冲生肖
            if forward == 6:
                score += 2.0
            
            scores[zodiac] = score
        
        return scores
    
    def _strategy_3_cycle(self, animals):
        """策略3: 周期性预测"""
        scores = {}
        
        recent_40 = animals[-40:] if len(animals) >= 40 else animals
        
        for zodiac in self.zodiacs:
            score = 0.0
            
            positions = [i for i, a in enumerate(recent_40) if a == zodiac]
            
            if len(positions) >= 2:
                gaps = [positions[j+1] - positions[j] for j in range(len(positions)-1)]
                avg_cycle = np.mean(gaps)
                
                # 计算距离上次出现的期数
                last_pos = positions[-1]
                current_gap = len(recent_40) - 1 - last_pos
                
                # 接近周期点加分
                diff = abs(current_gap - avg_cycle)
                if diff <= 1:
                    score += 4.0
                elif diff <= 2:
                    score += 3.0
                elif diff <= 3:
                    score += 2.0
                elif diff <= 4:
                    score += 1.0
            
            scores[zodiac] = score
        
        return scores
    
    def _strategy_4_element_balance(self, animals):
        """策略4: 五行平衡"""
        scores = {}
        
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        element_counter = Counter([self.zodiac_elements.get(z, '') for z in recent_20])
        
        # 计算五行平均出现次数
        avg_element_freq = len(recent_20) / 5
        
        for zodiac in self.zodiacs:
            element = self.zodiac_elements.get(zodiac, '')
            element_freq = element_counter.get(element, 0)
            
            # 五行缺失加分
            deviation = avg_element_freq - element_freq
            
            if deviation >= 2:
                score = 3.5
            elif deviation >= 1:
                score = 2.5
            elif deviation >= 0:
                score = 1.5
            else:
                score = -1.0 * abs(deviation)
            
            scores[zodiac] = score
        
        return scores
    
    def _strategy_5_number_pattern(self, animals, numbers):
        """策略5: 号码模式分析"""
        scores = {}
        
        recent_nums = numbers[-15:] if len(numbers) >= 15 else numbers
        avg_num = np.mean(recent_nums)
        
        # 分析号码趋势
        for zodiac in self.zodiacs:
            score = 0.0
            zodiac_nums = self.zodiac_numbers[zodiac]
            
            # 号码范围匹配
            for num in zodiac_nums:
                if abs(num - avg_num) <= 10:
                    score += 0.8
                elif abs(num - avg_num) <= 20:
                    score += 0.4
            
            # 奇偶匹配
            odd_count = sum(1 for n in recent_nums if n % 2 == 1)
            even_count = len(recent_nums) - odd_count
            
            zodiac_odd = sum(1 for n in zodiac_nums if n % 2 == 1)
            zodiac_even = len(zodiac_nums) - zodiac_odd
            
            if odd_count > even_count and zodiac_odd > zodiac_even:
                score += 1.5
            elif even_count > odd_count and zodiac_even > zodiac_odd:
                score += 1.5
            
            scores[zodiac] = score
        
        return scores
    
    def _ensemble_predict(self, animals, numbers):
        """集成多策略预测"""
        
        # 获取各策略评分
        score_1 = self._strategy_1_frequency_cold(animals)
        score_2 = self._strategy_2_rotation(animals)
        score_3 = self._strategy_3_cycle(animals)
        score_4 = self._strategy_4_element_balance(animals)
        score_5 = self._strategy_5_number_pattern(animals, numbers)
        
        # 策略权重（优化后）
        weights = {
            'freq_cold': 0.35,      # 冷门分析权重最高
            'rotation': 0.25,       # 轮转规律
            'cycle': 0.20,          # 周期性
            'element': 0.12,        # 五行平衡
            'number': 0.08          # 号码模式
        }
        
        # 加权融合
        final_scores = {}
        for zodiac in self.zodiacs:
            score = (
                weights['freq_cold'] * score_1.get(zodiac, 0) +
                weights['rotation'] * score_2.get(zodiac, 0) +
                weights['cycle'] * score_3.get(zodiac, 0) +
                weights['element'] * score_4.get(zodiac, 0) +
                weights['number'] * score_5.get(zodiac, 0)
            )
            final_scores[zodiac] = score
        
        return final_scores
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_n=5):
        """
        预测TOP5生肖
        
        Returns:
            dict: 预测结果
        """
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        numbers = df['number'].values
        
        # 集成预测
        final_scores = self._ensemble_predict(animals, numbers)
        
        # 排序
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_zodiacs = sorted_zodiacs[:top_n]
        
        # 推荐号码
        recommended_numbers = []
        for rank, (zodiac, score) in enumerate(top_zodiacs, 1):
            weight = top_n + 1 - rank
            nums = self.zodiac_numbers[zodiac]
            for num in nums:
                recommended_numbers.append((num, weight))
        
        # 去重并排序
        num_scores = {}
        for num, weight in recommended_numbers:
            if num not in num_scores:
                num_scores[num] = 0
            num_scores[num] += weight
        
        sorted_nums = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_nums[:15]]
        
        return {
            'model': self.model_name,
            'version': self.version,
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_number': df.iloc[-1]['number'],
            'last_zodiac': df.iloc[-1]['animal'],
            f'top{top_n}_zodiacs': top_zodiacs,
            'top15_numbers': top_numbers,
            'all_scores': final_scores
        }


if __name__ == "__main__":
    print("="*80)
    print("生肖优化预测模型 - TOP5专用版")
    print("="*80)
    
    predictor = ZodiacOptimizedPredictor()
    result = predictor.predict(top_n=5)
    
    print(f"\n模型: {result['model']} v{result['version']}")
    print(f"\n最新一期（第{result['total_periods']}期）")
    print(f"  日期: {result['last_date']}")
    print(f"  开出: {result['last_number']} - {result['last_zodiac']}")
    
    print(f"\n下一期预测（第{result['total_periods']+1}期）")
    print("\n生肖 TOP 5:")
    for i, (zodiac, score) in enumerate(result['top5_zodiacs'], 1):
        nums = predictor.zodiac_numbers[zodiac]
        level = "强推" if i <= 2 else "推荐" if i <= 3 else "备选"
        print(f"  {i}. {zodiac} [{level}] 评分: {score:6.2f}  号码: {nums}")
    
    print(f"\n推荐号码 TOP 15: {result['top15_numbers']}")
    print("\n" + "="*80)
