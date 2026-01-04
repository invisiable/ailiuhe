"""
生肖预测模型 - 机器学习混合版
结合统计逻辑和机器学习算法，提升预测准确率

特点：
1. 保留原有的统计分析逻辑（频率、轮转、冷热度、周期性）
2. 新增机器学习模型（随机森林、XGBoost、LightGBM）
3. 特征工程：提取多维度特征
4. 模型融合：统计评分 + ML预测概率
5. 动态权重调整
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("警告: 部分机器学习库未安装，将使用基础统计模式")


class ZodiacMLPredictor:
    """生肖预测器 - 机器学习混合版"""
    
    def __init__(self, ml_weight=0.4):
        """
        初始化预测器
        
        Args:
            ml_weight: 机器学习权重 (0-1)，默认0.4
                      统计权重 = 1 - ml_weight
        """
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
        
        # 反向映射：号码到生肖
        self.number_to_zodiac = {}
        for zodiac, numbers in self.zodiac_numbers.items():
            for num in numbers:
                self.number_to_zodiac[num] = zodiac
        
        self.ml_weight = ml_weight
        self.stat_weight = 1 - ml_weight
        self.model_name = "生肖预测模型(统计+机器学习)"
        self.version = "2.0"
        
        # 机器学习模型
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.zodiacs)
        
        # 训练状态
        self.is_trained = False
    
    def _extract_features(self, df, index):
        """
        提取特征
        
        Args:
            df: 数据DataFrame
            index: 当前索引位置
        
        Returns:
            dict: 特征字典
        """
        features = {}
        
        # 基础信息 - 清理数据
        animals = [str(a).strip() for a in df['animal'].values[:index]]
        numbers = df['number'].values[:index]
        
        if len(animals) < 10:
            return None
        
        # 特征1: 各生肖最近出现频率
        recent_50 = animals[-50:] if len(animals) >= 50 else animals
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        recent_10 = animals[-10:]
        recent_5 = animals[-5:]
        
        for zodiac in self.zodiacs:
            features[f'freq_50_{zodiac}'] = list(recent_50).count(zodiac)
            features[f'freq_30_{zodiac}'] = list(recent_30).count(zodiac)
            features[f'freq_20_{zodiac}'] = list(recent_20).count(zodiac)
            features[f'freq_10_{zodiac}'] = list(recent_10).count(zodiac)
            features[f'freq_5_{zodiac}'] = list(recent_5).count(zodiac)
        
        # 特征2: 距离上次出现的间隔
        for zodiac in self.zodiacs:
            try:
                positions = [i for i, a in enumerate(animals) if a == zodiac]
                if positions:
                    gap = len(animals) - 1 - positions[-1]
                    features[f'gap_{zodiac}'] = gap
                else:
                    features[f'gap_{zodiac}'] = 999  # 从未出现
            except:
                features[f'gap_{zodiac}'] = 999
        
        # 特征3: 最近N期的生肖模式
        last_zodiac = animals[-1] if len(animals) > 0 else None
        if last_zodiac and last_zodiac in self.zodiacs:
            last_idx = self.zodiacs.index(last_zodiac)
            features['last_zodiac_idx'] = last_idx
            
            # 计算与其他生肖的相对位置
            for i, zodiac in enumerate(self.zodiacs):
                features[f'relative_pos_{zodiac}'] = (i - last_idx) % 12
        else:
            features['last_zodiac_idx'] = -1
            for zodiac in self.zodiacs:
                features[f'relative_pos_{zodiac}'] = 0
        
        # 特征4: 连续性特征
        features['has_consecutive'] = int(len(animals) >= 2 and animals[-1] == animals[-2])
        features['has_triple'] = int(len(animals) >= 3 and 
                                    animals[-1] == animals[-2] == animals[-3])
        
        # 特征5: 周期性特征
        for zodiac in self.zodiacs:
            positions = [i for i, a in enumerate(animals) if a == zodiac]
            if len(positions) >= 2:
                gaps = [positions[j+1] - positions[j] for j in range(len(positions)-1)]
                features[f'avg_cycle_{zodiac}'] = np.mean(gaps)
                features[f'std_cycle_{zodiac}'] = np.std(gaps) if len(gaps) > 1 else 0
            else:
                features[f'avg_cycle_{zodiac}'] = 0
                features[f'std_cycle_{zodiac}'] = 0
        
        # 特征6: 号码分布特征（可能影响生肖）
        recent_numbers_10 = numbers[-10:] if len(numbers) >= 10 else numbers
        features['avg_number_10'] = np.mean(recent_numbers_10)
        features['std_number_10'] = np.std(recent_numbers_10)
        features['max_number_10'] = np.max(recent_numbers_10)
        features['min_number_10'] = np.min(recent_numbers_10)
        
        # 特征7: 生肖多样性
        features['unique_zodiacs_5'] = len(set(recent_5))
        features['unique_zodiacs_10'] = len(set(recent_10))
        
        # 特征8: 热度方差
        counter_30 = Counter(recent_30)
        freqs = [counter_30.get(z, 0) for z in self.zodiacs]
        features['freq_variance'] = np.var(freqs)
        features['freq_std'] = np.std(freqs)
        
        return features
    
    def _build_training_data(self, df):
        """
        构建训练数据集
        
        Args:
            df: 原始数据DataFrame
        
        Returns:
            X, y: 特征矩阵和标签
        """
        X_list = []
        y_list = []
        
        # 从第11期开始（需要至少10期历史数据）
        for i in range(10, len(df)):
            features = self._extract_features(df, i)
            if features is None:
                continue
            
            # 获取标签并清理
            label = str(df.iloc[i]['animal']).strip()
            
            # 只添加有效的生肖标签
            if label in self.zodiacs:
                X_list.append(features)
                y_list.append(label)
        
        if len(X_list) == 0:
            return None, None
        
        # 转换为DataFrame
        X_df = pd.DataFrame(X_list)
        
        # 填充缺失值
        X_df = X_df.fillna(0)
        
        # 编码标签
        y_encoded = self.label_encoder.transform(y_list)
        
        return X_df, y_encoded
    
    def train_models(self, csv_file='data/lucky_numbers.csv'):
        """
        训练机器学习模型
        
        Args:
            csv_file: 数据文件路径
        """
        if not ML_AVAILABLE:
            print("机器学习库未安装，跳过模型训练")
            return
        
        print("开始训练机器学习模型...")
        
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        # 构建训练数据
        X, y = self._build_training_data(df)
        
        if X is None or len(X) < 20:
            print("数据不足，无法训练模型")
            return
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练多个模型
        print(f"训练数据: {len(X)} 个样本, {X.shape[1]} 个特征")
        
        # 1. 随机森林
        print("  训练随机森林...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.models['rf'].fit(X_scaled, y)
        
        # 2. 梯度提升
        print("  训练梯度提升...")
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.models['gb'].fit(X_scaled, y)
        
        # 3. XGBoost (如果可用)
        try:
            print("  训练XGBoost...")
            self.models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            self.models['xgb'].fit(X_scaled, y)
        except Exception as e:
            print(f"  XGBoost训练失败: {e}")
        
        # 4. LightGBM (如果可用)
        try:
            print("  训练LightGBM...")
            self.models['lgb'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
            self.models['lgb'].fit(X_scaled, y)
        except Exception as e:
            print(f"  LightGBM训练失败: {e}")
        
        self.is_trained = True
        print(f"✓ 模型训练完成，共训练 {len(self.models)} 个模型\n")
    
    def _get_ml_predictions(self, features_dict):
        """
        获取机器学习模型的预测
        
        Args:
            features_dict: 特征字典
        
        Returns:
            dict: {生肖: 概率} 的字典
        """
        if not ML_AVAILABLE or not self.is_trained or len(self.models) == 0:
            return {zodiac: 1.0/12 for zodiac in self.zodiacs}  # 均等概率
        
        # 转换为DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # 标准化
        features_scaled = self.scaler.transform(features_df)
        
        # 收集所有模型的预测概率
        all_probs = []
        
        for model_name, model in self.models.items():
            try:
                probs = model.predict_proba(features_scaled)[0]
                all_probs.append(probs)
            except:
                continue
        
        if len(all_probs) == 0:
            return {zodiac: 1.0/12 for zodiac in self.zodiacs}
        
        # 平均所有模型的预测
        avg_probs = np.mean(all_probs, axis=0)
        
        # 转换为字典
        prob_dict = {}
        for i, zodiac in enumerate(self.zodiacs):
            prob_dict[zodiac] = avg_probs[i]
        
        return prob_dict
    
    def _calculate_statistical_scores(self, animals):
        """
        计算统计评分（保留原有逻辑）
        
        Args:
            animals: 历史生肖数据
        
        Returns:
            dict: {生肖: 评分} 的字典
        """
        pattern = self._analyze_zodiac_pattern(animals)
        scores = {}
        
        for zodiac in self.zodiacs:
            score = 0.0
            
            # 多时间窗口频率分析
            freq_50 = pattern['freq_50'].get(zodiac, 0)
            freq_30 = pattern['freq_30'].get(zodiac, 0)
            freq_20 = pattern['freq_20'].get(zodiac, 0)
            freq_10 = pattern['freq_10'].get(zodiac, 0)
            
            if freq_50 <= 2:
                score += 4.0
            elif freq_50 <= 3:
                score += 2.5
            elif freq_50 <= 4:
                score += 1.0
            
            if freq_30 == 0:
                score += 3.5
            elif freq_30 == 1:
                score += 2.5
            elif freq_30 == 2:
                score += 1.5
            
            if freq_20 == 0:
                score += 2.5
            elif freq_20 == 1:
                score += 1.5
            
            if freq_10 == 0:
                score += 1.5
            
            # 避重机制
            if zodiac in pattern['recent_5']:
                last_appear_idx = len(pattern['recent_5']) - 1 - list(reversed(pattern['recent_5'])).index(zodiac)
                gap = len(pattern['recent_5']) - 1 - last_appear_idx
                
                if gap == 0:
                    score -= 4.5
                elif gap == 1:
                    score -= 3.0
                elif gap == 2:
                    score -= 2.0
                elif gap == 3:
                    score -= 1.0
                else:
                    score -= 0.5
            else:
                score += 3.0
            
            if pattern['has_consecutive'] and pattern['last_zodiac'] == zodiac:
                score -= 3.0
            
            # 生肖轮转
            last_zodiac = pattern['last_zodiac']
            if last_zodiac and last_zodiac in self.zodiacs:
                last_idx = self.zodiacs.index(last_zodiac)
                zodiac_idx = self.zodiacs.index(zodiac)
                
                forward_dist = (zodiac_idx - last_idx) % 12
                backward_dist = (last_idx - zodiac_idx) % 12
                
                if forward_dist in [1, 2]:
                    score += 2.0
                elif forward_dist == 3:
                    score += 1.0
                elif backward_dist in [1, 2]:
                    score += 1.5
                elif backward_dist == 3:
                    score += 0.5
                
                if forward_dist == 6:
                    score += 1.0
            
            # 周期性
            cycle = pattern['cycle_pattern'].get(zodiac, 0)
            if cycle > 0 and freq_30 > 0:
                try:
                    positions = [idx for idx, animal in enumerate(pattern['recent_30']) 
                                if animal.strip() == zodiac]
                    if positions:
                        last_pos = positions[-1]
                        gap_since_last = len(pattern['recent_30']) - 1 - last_pos
                        
                        if abs(gap_since_last - cycle) <= 2:
                            score += 2.0
                        elif abs(gap_since_last - cycle) <= 4:
                            score += 1.0
                except:
                    pass
            
            # 热度均衡
            avg_freq_30 = len(pattern['recent_30']) / 12
            deviation = freq_30 - avg_freq_30
            
            if deviation < -1.5:
                score += 2.0
            elif deviation < -0.5:
                score += 1.0
            elif deviation > 1.5:
                score -= 1.5
            elif deviation > 0.5:
                score -= 0.5
            
            scores[zodiac] = score
        
        return scores
    
    def _analyze_zodiac_pattern(self, animals):
        """分析生肖规律"""
        recent_50 = animals[-50:] if len(animals) >= 50 else animals
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        recent_5 = animals[-5:] if len(animals) >= 5 else animals
        
        freq_50 = Counter(recent_50)
        freq_30 = Counter(recent_30)
        freq_20 = Counter(recent_20)
        freq_10 = Counter(recent_10)
        freq_5 = Counter(recent_5)
        
        has_consecutive = len(recent_5) >= 2 and recent_5[-1] == recent_5[-2]
        
        zodiac_cycle_pattern = {}
        for zodiac in self.zodiacs:
            positions = [idx for idx, animal in enumerate(recent_30) if animal.strip() == zodiac]
            if len(positions) >= 2:
                gaps = [positions[j+1] - positions[j] for j in range(len(positions)-1)]
                zodiac_cycle_pattern[zodiac] = np.mean(gaps) if gaps else 0
            else:
                zodiac_cycle_pattern[zodiac] = 0
        
        return {
            'recent_50': recent_50,
            'recent_30': recent_30,
            'recent_20': recent_20,
            'recent_10': recent_10,
            'recent_5': recent_5,
            'freq_50': freq_50,
            'freq_30': freq_30,
            'freq_20': freq_20,
            'freq_10': freq_10,
            'freq_5': freq_5,
            'has_consecutive': has_consecutive,
            'last_zodiac': recent_5[-1].strip() if len(recent_5) > 0 else None,
            'cycle_pattern': zodiac_cycle_pattern
        }
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_n=6):
        """
        预测下一期最可能的生肖
        
        Args:
            csv_file: 数据文件路径
            top_n: 返回TOP N个生肖，默认6
        
        Returns:
            dict: 预测结果
        """
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = df['animal'].values
        numbers = df['number'].values
        
        # 如果未训练，先训练模型
        if not self.is_trained and ML_AVAILABLE:
            self.train_models(csv_file)
        
        # 1. 获取统计评分
        stat_scores = self._calculate_statistical_scores(animals)
        
        # 归一化统计评分到0-1
        stat_values = list(stat_scores.values())
        stat_min = min(stat_values)
        stat_max = max(stat_values)
        stat_range = stat_max - stat_min if stat_max > stat_min else 1
        
        stat_scores_norm = {
            zodiac: (score - stat_min) / stat_range 
            for zodiac, score in stat_scores.items()
        }
        
        # 2. 获取机器学习预测概率
        features = self._extract_features(df, len(df))
        ml_probs = self._get_ml_predictions(features) if features else {}
        
        # 3. 融合统计评分和ML概率
        final_scores = {}
        for zodiac in self.zodiacs:
            stat_score = stat_scores_norm.get(zodiac, 0)
            ml_prob = ml_probs.get(zodiac, 1.0/12)
            
            # 加权融合
            final_score = (self.stat_weight * stat_score + 
                          self.ml_weight * ml_prob * 10)  # ML概率放大10倍以匹配scale
            
            final_scores[zodiac] = final_score
        
        # 排序
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_zodiacs = sorted_zodiacs[:top_n]
        
        # 推荐号码
        recommended_numbers = self._predict_numbers_by_zodiac(top_zodiacs, numbers)
        
        # 返回结果
        last_period = len(df)
        last_date = df.iloc[-1]['date']
        last_number = df.iloc[-1]['number']
        last_zodiac = df.iloc[-1]['animal']
        
        return {
            'model': self.model_name,
            'version': self.version,
            'ml_enabled': ML_AVAILABLE and self.is_trained,
            'ml_weight': self.ml_weight,
            'stat_weight': self.stat_weight,
            'total_periods': last_period,
            'last_date': last_date,
            'last_number': last_number,
            'last_zodiac': last_zodiac,
            f'top{top_n}_zodiacs': top_zodiacs,
            'top18_numbers': recommended_numbers[:18],
            'all_scores': final_scores,
            'stat_scores': stat_scores,
            'ml_probs': ml_probs if ml_probs else None
        }
    
    def _predict_numbers_by_zodiac(self, top_zodiacs, recent_numbers=None):
        """根据预测的生肖推荐号码"""
        number_scores = {}
        
        for rank, (zodiac, zodiac_score) in enumerate(top_zodiacs, 1):
            numbers = self.zodiac_numbers.get(zodiac, [])
            weight = 7 - rank
            
            for num in numbers:
                if num not in number_scores:
                    number_scores[num] = 0
                number_scores[num] += weight * (1 + zodiac_score * 0.1)
        
        if recent_numbers is not None and len(recent_numbers) > 0:
            recent_5 = set(recent_numbers[-5:]) if len(recent_numbers) >= 5 else set(recent_numbers)
            recent_10 = set(recent_numbers[-10:]) if len(recent_numbers) >= 10 else set(recent_numbers)
            
            for num in number_scores:
                if num in recent_5:
                    number_scores[num] *= 0.3
                elif num in recent_10:
                    number_scores[num] *= 0.6
        
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        recommended = [num for num, score in sorted_numbers]
        
        if len(recommended) < 18:
            all_numbers = list(range(1, 50))
            for num in all_numbers:
                if num not in recommended:
                    if 15 <= num <= 35:
                        recommended.append(num)
                    if len(recommended) >= 18:
                        break
        
        return recommended[:18]


if __name__ == "__main__":
    print("="*80)
    print("🤖 生肖预测模型 - 机器学习混合版")
    print("="*80)
    
    # 创建预测器
    predictor = ZodiacMLPredictor(ml_weight=0.4)
    
    # 预测
    result = predictor.predict()
    
    # 显示结果
    print(f"\n模型: {result['model']} v{result['version']}")
    print(f"机器学习: {'✓ 已启用' if result['ml_enabled'] else '✗ 未启用'}")
    print(f"权重配比: 统计{result['stat_weight']*100:.0f}% + ML{result['ml_weight']*100:.0f}%")
    
    print(f"\n最新一期（第{result['total_periods']}期）")
    print(f"  日期: {result['last_date']}")
    print(f"  开出: {result['last_number']} - {result['last_zodiac']}")
    
    print(f"\n下一期预测（第{result['total_periods']+1}期）")
    print("\n⭐ 生肖 TOP 6:")
    for i, (zodiac, score) in enumerate(result['top6_zodiacs'], 1):
        nums = predictor.zodiac_numbers[zodiac]
        level = "强推" if i <= 2 else "推荐" if i <= 4 else "备选"
        print(f"  {i}. {zodiac} [{level}] 评分: {score:6.2f}  →  号码: {nums}")
    
    print(f"\n📋 推荐号码 TOP 18:")
    top18 = result['top18_numbers']
    print(f"  强推: {top18[0:6]}")
    print(f"  推荐: {top18[6:12]}")
    print(f"  备选: {top18[12:18]}")
    
    print("\n" + "="*80)
