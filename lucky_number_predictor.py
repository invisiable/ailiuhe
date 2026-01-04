"""
幸运数字预测模块
基于历史幸运数字数据进行预测
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime

# 导入高级模型
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class LuckyNumberPredictor:
    """幸运数字预测器"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = None
        self.sequence_length = 10  # 使用过去10个数字来预测下一个
        self.feature_names = []
        
    def create_sequences(self, data, seq_length):
        """
        将时间序列数据转换为监督学习问题
        
        Args:
            data: 原始数字序列
            seq_length: 序列长度
            
        Returns:
            X, y: 特征和目标
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            # 使用过去seq_length个数字作为特征
            sequence = data[i:i + seq_length]
            target = data[i + seq_length]
            
            # 提取特征：历史值、统计特征
            features = list(sequence)
            features.extend([
                np.mean(sequence),      # 平均值
                np.std(sequence),       # 标准差
                np.max(sequence),       # 最大值
                np.min(sequence),       # 最小值
                sequence[-1] - sequence[-2] if len(sequence) > 1 else 0,  # 最近趋势
            ])
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def load_data(self, file_path, number_column='number', date_column='date', 
                  animal_column='animal', element_column='element'):
        """
        加载幸运数字历史数据
        
        Args:
            file_path: CSV文件路径
            number_column: 数字列名
            date_column: 日期列名
            animal_column: 生肖列名
            element_column: 五行列名
            
        Returns:
            成功返回True
        """
        try:
            # 读取数据（尝试多种编码）
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='gbk')
            
            if number_column not in df.columns:
                raise ValueError(f"列 '{number_column}' 不存在于数据中")
            
            # 按日期排序
            if date_column and date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df = df.sort_values(date_column)
            
            # 生肖编码映射
            animal_mapping = {
                '鼠': 0, '牛': 1, '虎': 2, '兔': 3, '龙': 4, '蛇': 5,
                '马': 6, '羊': 7, '猴': 8, '鸡': 9, '狗': 10, '猪': 11
            }
            # 五行编码映射
            element_mapping = {'金': 0, '木': 1, '水': 2, '火': 3, '土': 4}
            
            # 五行对应的数字范围
            self.element_numbers = {
                '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
                '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
                '水': [13, 14, 21, 22, 29, 30, 43, 44],
                '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
                '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
            }
            
            # 反向映射：从数字到五行
            self.number_to_element = {}
            for element, numbers in self.element_numbers.items():
                for num in numbers:
                    self.number_to_element[num] = element
            
            # 提取数字序列
            numbers = df[number_column].values
            
            # 编码生肖和五行
            animals = df[animal_column].map(animal_mapping).values if animal_column in df.columns else None
            elements = df[element_column].map(element_mapping).values if element_column in df.columns else None
            
            # 检查数据量
            if len(numbers) < self.sequence_length + 10:
                raise ValueError(f"数据量不足，至少需要 {self.sequence_length + 10} 个历史数据点")
            
            # 创建序列特征（包含生肖和五行）
            self.X, self.y = self.create_sequences_with_features(
                numbers, animals, elements, self.sequence_length
            )
            
            # 保存原始数据用于预测
            self.raw_numbers = numbers
            self.raw_animals = animals
            self.raw_elements = elements
            self.number_column = number_column
            self.animal_mapping = animal_mapping
            self.element_mapping = element_mapping
            self.reverse_animal_mapping = {v: k for k, v in animal_mapping.items()}
            self.reverse_element_mapping = {v: k for k, v in element_mapping.items()}
            
            # 生成特征名称
            self.feature_names = [f'lag_{i+1}' for i in range(self.sequence_length)]
            self.feature_names.extend(['mean', 'std', 'max', 'min', 'trend'])
            if animals is not None:
                self.feature_names.extend([f'animal_lag_{i+1}' for i in range(self.sequence_length)])
            if elements is not None:
                self.feature_names.extend([f'element_lag_{i+1}' for i in range(self.sequence_length)])
            
            return True
            
        except Exception as e:
            raise Exception(f"加载数据失败: {str(e)}")
    
    def create_sequences_with_features(self, numbers, animals, elements, seq_length):
        """
        创建包含生肖和五行特征的序列
        """
        X, y = [], []
        
        for i in range(len(numbers) - seq_length):
            # 数字序列
            sequence = numbers[i:i + seq_length]
            target = numbers[i + seq_length]
            
            # 基础特征（处理NaN）
            features = list(sequence)
            features.extend([
                np.nan_to_num(np.mean(sequence), nan=0.0),
                np.nan_to_num(np.std(sequence), nan=0.0),
                np.nan_to_num(np.max(sequence), nan=0.0),
                np.nan_to_num(np.min(sequence), nan=0.0),
                np.nan_to_num(sequence[-1] - sequence[-2] if len(sequence) > 1 else 0, nan=0.0),
            ])
            
            # 添加生肖特征
            if animals is not None:
                animal_seq = animals[i:i + seq_length]
                features.extend(list(animal_seq))
            
            # 添加五行特征
            if elements is not None:
                element_seq = elements[i:i + seq_length]
                features.extend(list(element_seq))
            
            X.append(features)
            y.append(target)
        
        # 转换为数组并处理所有NaN
        X = np.array(X)
        y = np.array(y)
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        return X, y
    
    def train_model(self, model_type='random_forest', test_size=0.2, random_state=42):
        """
        训练预测模型
        
        Args:
            model_type: 模型类型 ('random_forest', 'gradient_boosting', 'neural_network')
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练结果字典
        """
        try:
            # 分割数据（注意时间序列应该使用顺序分割）
            split_idx = int(len(self.X) * (1 - test_size))
            X_train, X_test = self.X[:split_idx], self.X[split_idx:]
            y_train, y_test = self.y[:split_idx], self.y[split_idx:]
            
            # 标准化特征
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 创建模型
            if model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=random_state,
                    n_jobs=-1
                )
                self.model_type = 'RandomForest'
                
            elif model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=random_state
                )
                self.model_type = 'GradientBoosting'
                
            elif model_type == 'neural_network':
                self.model = MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    activation='relu',
                    max_iter=1000,
                    random_state=random_state,
                    early_stopping=True
                )
                self.model_type = 'NeuralNetwork'
            
            elif model_type == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    raise ValueError("XGBoost未安装，请运行: pip install xgboost")
                self.model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    n_jobs=-1
                )
                self.model_type = 'XGBoost'
            
            elif model_type == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    raise ValueError("LightGBM未安装，请运行: pip install lightgbm")
                self.model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1
                )
                self.model_type = 'LightGBM'
            
            elif model_type == 'catboost':
                if not CATBOOST_AVAILABLE:
                    raise ValueError("CatBoost未安装，请运行: pip install catboost")
                self.model = CatBoostRegressor(
                    iterations=200,
                    depth=6,
                    learning_rate=0.1,
                    random_state=random_state,
                    verbose=False
                )
                self.model_type = 'CatBoost'
            
            elif model_type == 'svr':
                self.model = SVR(
                    kernel='rbf',
                    C=100,
                    epsilon=0.1,
                    gamma='scale'
                )
                self.model_type = 'SVR'
            
            elif model_type == 'ensemble':
                # 集成多个模型
                from sklearn.ensemble import VotingRegressor
                
                rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
                gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state)
                
                if XGBOOST_AVAILABLE:
                    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=random_state, n_jobs=-1)
                    self.model = VotingRegressor([('rf', rf), ('gb', gb), ('xgb', xgb_model)])
                else:
                    self.model = VotingRegressor([('rf', rf), ('gb', gb)])
                
                self.model_type = 'Ensemble'
                
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 训练模型
            self.model.fit(X_train_scaled, y_train)
            
            # 预测
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            # 评估
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            results = {
                'model_type': self.model_type,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'sequence_length': self.sequence_length,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'metrics': (f"测试集MAE: {test_mae:.4f}, "
                          f"RMSE: {test_rmse:.4f}, "
                          f"R²: {test_r2:.4f}"),
                'y_test': y_test,
                'y_pred': y_pred_test
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"训练模型失败: {str(e)}")
    
    def predict_next(self, n_predictions=1):
        """
        预测接下来的幸运数字（包含生肖和五行）
        
        Args:
            n_predictions: 预测未来多少个数字
            
        Returns:
            预测结果列表，每个元素包含 {'number', 'animal', 'element'}
        """
        try:
            if self.model is None:
                raise ValueError("模型尚未训练")
            
            predictions = []
            current_sequence = list(self.raw_numbers[-self.sequence_length:])
            current_animals = list(self.raw_animals[-self.sequence_length:]) if self.raw_animals is not None else None
            current_elements = list(self.raw_elements[-self.sequence_length:]) if self.raw_elements is not None else None
            
            for i in range(n_predictions):
                # 构造特征
                features = list(current_sequence[-self.sequence_length:])
                features.extend([
                    np.mean(current_sequence[-self.sequence_length:]),
                    np.std(current_sequence[-self.sequence_length:]),
                    np.max(current_sequence[-self.sequence_length:]),
                    np.min(current_sequence[-self.sequence_length:]),
                    current_sequence[-1] - current_sequence[-2] if len(current_sequence) > 1 else 0,
                ])
                
                # 添加生肖和五行特征
                if current_animals is not None:
                    features.extend(current_animals[-self.sequence_length:])
                if current_elements is not None:
                    features.extend(current_elements[-self.sequence_length:])
                
                # 标准化并预测
                features_scaled = self.scaler.transform([features])
                next_number = self.model.predict(features_scaled)[0]
                
                # 四舍五入到整数
                next_number = int(round(max(1, min(49, next_number))))
                
                # 根据数字自动关联五行
                next_element_name = self.number_to_element.get(next_number, '金')
                
                # 预测生肖（按周期循环）
                last_animal = current_animals[-1] if current_animals is not None else 0
                next_animal = (last_animal + 1) % 12
                next_animal_name = self.reverse_animal_mapping.get(next_animal, '鼠')
                
                result = {
                    'number': next_number,
                    'animal': next_animal_name,
                    'element': next_element_name
                }
                
                predictions.append(result)
                current_sequence.append(next_number)
                if current_animals is not None:
                    current_animals.append(next_animal)
                if current_elements is not None:
                    # 更新五行编码（根据数字对应的五行）
                    element_code = self.element_mapping.get(next_element_name, 0)
                    current_elements.append(element_code)
            
            return predictions
            
        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")
    
    def predict_top_probabilities(self, top_k=3):
        """
        预测下一期最可能的幸运数字及概率
        
        Args:
            top_k: 返回前K个最可能的数字（默认3个）
            
        Returns:
            列表，包含 {'number', 'probability', 'animal', 'element'}
        """
        try:
            if self.model is None:
                raise ValueError("模型尚未训练")
            
            # 构造当前特征
            current_sequence = list(self.raw_numbers[-self.sequence_length:])
            current_animals = list(self.raw_animals[-self.sequence_length:]) if self.raw_animals is not None else None
            current_elements = list(self.raw_elements[-self.sequence_length:]) if self.raw_elements is not None else None
            
            features = list(current_sequence[-self.sequence_length:])
            features.extend([
                np.mean(current_sequence[-self.sequence_length:]),
                np.std(current_sequence[-self.sequence_length:]),
                np.max(current_sequence[-self.sequence_length:]),
                np.min(current_sequence[-self.sequence_length:]),
                current_sequence[-1] - current_sequence[-2] if len(current_sequence) > 1 else 0,
            ])
            
            if current_animals is not None:
                features.extend(current_animals[-self.sequence_length:])
            if current_elements is not None:
                features.extend(current_elements[-self.sequence_length:])
            
            # 标准化特征
            features_scaled = self.scaler.transform([features])
            
            # 使用随机森林的每棵树进行预测
            if self.model_type == 'RandomForest':
                # 获取所有树的预测结果
                tree_predictions = []
                for estimator in self.model.estimators_:
                    pred = estimator.predict(features_scaled)[0]
                    tree_predictions.append(int(round(max(1, min(49, pred)))))
                
                # 统计每个数字出现的频率
                from collections import Counter
                number_counts = Counter(tree_predictions)
                total_trees = len(tree_predictions)
                
                # 计算概率并排序
                number_probs = []
                for number, count in number_counts.most_common(top_k):
                    probability = count / total_trees
                    
                    # 计算对应的生肖
                    last_animal = current_animals[-1] if current_animals is not None else 0
                    next_animal = (last_animal + 1) % 12
                    animal_name = self.reverse_animal_mapping.get(next_animal, '鼠')
                    
                    # 根据数字自动关联五行
                    element_name = self.number_to_element.get(number, '金')
                    
                    number_probs.append({
                        'number': number,
                        'probability': probability,
                        'animal': animal_name,
                        'element': element_name
                    })
                
                return number_probs
            
            else:
                # 对于非随机森林模型，使用单点预测
                pred = self.model.predict(features_scaled)[0]
                number = int(round(max(1, min(49, pred))))
                
                # 生成nearby数字作为候选
                candidates = []
                for offset in [0, -1, 1, -2, 2]:
                    candidate = max(1, min(49, number + offset))
                    if candidate not in [c['number'] for c in candidates]:
                        # 计算距离预测值的相似度作为概率
                        distance = abs(candidate - pred)
                        prob = 1.0 / (1.0 + distance)
                        
                        last_animal = current_animals[-1] if current_animals is not None else 0
                        next_animal = (last_animal + 1) % 12
                        animal_name = self.reverse_animal_mapping.get(next_animal, '鼠')
                        
                        # 根据数字自动关联五行
                        element_name = self.number_to_element.get(candidate, '金')
                        
                        candidates.append({
                            'number': candidate,
                            'probability': prob,
                            'animal': animal_name,
                            'element': element_name
                        })
                
                # 归一化概率
                total_prob = sum(c['probability'] for c in candidates)
                for c in candidates:
                    c['probability'] = c['probability'] / total_prob
                
                # 按概率排序并返回top_k
                candidates.sort(key=lambda x: x['probability'], reverse=True)
                return candidates[:top_k]
            
        except Exception as e:
            raise Exception(f"预测概率失败: {str(e)}")
    
    def predict_separately(self, top_k=3):
        """
        分别预测数字、生肖、五行的Top K结果
        
        Args:
            top_k: 每个类别返回前K个最可能的结果（默认3个）
            
        Returns:
            字典，包含三个列表：
            {
                'numbers': [{'value': 17, 'probability': 0.105}, ...],
                'animals': [{'value': '龙', 'probability': 0.25}, ...],
                'elements': [{'value': '木', 'probability': 0.30}, ...]
            }
        """
        try:
            if self.model is None:
                raise ValueError("模型尚未训练")
            
            # 构造当前特征
            current_sequence = list(self.raw_numbers[-self.sequence_length:])
            current_animals = list(self.raw_animals[-self.sequence_length:]) if self.raw_animals is not None else None
            current_elements = list(self.raw_elements[-self.sequence_length:]) if self.raw_elements is not None else None
            
            features = list(current_sequence[-self.sequence_length:])
            features.extend([
                np.mean(current_sequence[-self.sequence_length:]),
                np.std(current_sequence[-self.sequence_length:]),
                np.max(current_sequence[-self.sequence_length:]),
                np.min(current_sequence[-self.sequence_length:]),
                current_sequence[-1] - current_sequence[-2] if len(current_sequence) > 1 else 0,
            ])
            
            if current_animals is not None:
                features.extend(current_animals[-self.sequence_length:])
            if current_elements is not None:
                features.extend(current_elements[-self.sequence_length:])
            
            # 标准化特征
            features_scaled = self.scaler.transform([features])
            
            result = {
                'numbers': [],
                'animals': [],
                'elements': []
            }
            
            # 1. 预测数字（使用随机森林的每棵树）
            if self.model_type == 'RandomForest':
                from collections import Counter
                
                # 收集所有树的数字预测
                tree_number_predictions = []
                for estimator in self.model.estimators_:
                    pred = estimator.predict(features_scaled)[0]
                    number = int(round(max(1, min(49, pred))))
                    tree_number_predictions.append(number)
                
                # 统计数字频率
                number_counts = Counter(tree_number_predictions)
                total_trees = len(tree_number_predictions)
                
                for number, count in number_counts.most_common(top_k):
                    result['numbers'].append({
                        'value': number,
                        'probability': count / total_trees
                    })
            else:
                # 对于非随机森林模型，使用单点预测及其邻近值
                pred = self.model.predict(features_scaled)[0]
                base_number = int(round(max(1, min(49, pred))))
                
                candidates = []
                for offset in [0, -1, 1, -2, 2, -3, 3]:
                    candidate = max(1, min(49, base_number + offset))
                    if candidate not in [c['value'] for c in candidates]:
                        distance = abs(candidate - pred)
                        prob = 1.0 / (1.0 + distance)
                        candidates.append({'value': candidate, 'probability': prob})
                
                # 归一化概率
                total_prob = sum(c['probability'] for c in candidates)
                for c in candidates:
                    c['probability'] = c['probability'] / total_prob
                
                candidates.sort(key=lambda x: x['probability'], reverse=True)
                result['numbers'] = candidates[:top_k]
            
            # 2. 预测生肖（基于历史周期模式）
            if current_animals is not None:
                # 分析历史生肖出现模式
                animal_history = list(self.raw_animals[-20:])  # 取最近20个
                from collections import Counter
                animal_counter = Counter(animal_history)
                
                # 获取当前位置应该的生肖（周期）
                last_animal = current_animals[-1]
                expected_animal = (last_animal + 1) % 12
                
                # 基于历史频率和周期给出概率
                total_count = sum(animal_counter.values())
                animal_probs = []
                
                # 优先考虑周期性
                animal_probs.append({
                    'value': self.reverse_animal_mapping.get(expected_animal, '鼠'),
                    'probability': 0.5  # 周期预测给50%权重
                })
                
                # 加入历史高频生肖
                for animal_code, count in animal_counter.most_common(5):
                    animal_name = self.reverse_animal_mapping.get(animal_code, '鼠')
                    if animal_name not in [a['value'] for a in animal_probs]:
                        prob = (count / total_count) * 0.5  # 历史频率给50%权重
                        animal_probs.append({
                            'value': animal_name,
                            'probability': prob
                        })
                
                # 归一化
                total_prob = sum(a['probability'] for a in animal_probs)
                for a in animal_probs:
                    a['probability'] = a['probability'] / total_prob
                
                animal_probs.sort(key=lambda x: x['probability'], reverse=True)
                result['animals'] = animal_probs[:top_k]
            else:
                # 如果没有生肖数据，均匀分布
                animal_names = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
                for i in range(top_k):
                    result['animals'].append({
                        'value': animal_names[i],
                        'probability': 1.0 / 12
                    })
            
            # 3. 预测五行（基于数字预测结果和历史频率）
            if current_elements is not None:
                # 从数字预测中获取对应的五行
                element_from_numbers = {}
                for num_pred in result['numbers']:
                    num = num_pred['value']
                    element = self.number_to_element.get(num)
                    if element:
                        if element not in element_from_numbers:
                            element_from_numbers[element] = 0
                        # 根据数字概率累加
                        element_from_numbers[element] += num_pred['probability']
                
                # 分析历史五行出现频率
                element_history = list(self.raw_elements[-20:])  # 取最近20个
                from collections import Counter
                element_counter = Counter(element_history)
                total_count = sum(element_counter.values())
                
                # 综合数字关联和历史频率
                element_probs = []
                all_elements = ['金', '木', '水', '火', '土']
                
                for element in all_elements:
                    # 从数字预测得到的概率（70%权重）
                    prob_from_numbers = element_from_numbers.get(element, 0) * 0.7
                    
                    # 历史频率（30%权重）
                    element_code = self.element_mapping.get(element, 0)
                    historical_count = element_counter.get(element_code, 0)
                    prob_from_history = (historical_count / total_count if total_count > 0 else 0.2) * 0.3
                    
                    total_prob = prob_from_numbers + prob_from_history
                    if total_prob > 0:
                        element_probs.append({
                            'value': element,
                            'probability': total_prob
                        })
                
                # 归一化
                total_prob = sum(e['probability'] for e in element_probs)
                if total_prob > 0:
                    for e in element_probs:
                        e['probability'] = e['probability'] / total_prob
                
                element_probs.sort(key=lambda x: x['probability'], reverse=True)
                result['elements'] = element_probs[:top_k]
            else:
                # 如果没有五行数据，基于数字预测推断
                element_from_numbers = {}
                for num_pred in result['numbers']:
                    num = num_pred['value']
                    element = self.number_to_element.get(num, '金')
                    if element not in element_from_numbers:
                        element_from_numbers[element] = 0
                    element_from_numbers[element] += num_pred['probability']
                
                element_probs = [{'value': k, 'probability': v} for k, v in element_from_numbers.items()]
                element_probs.sort(key=lambda x: x['probability'], reverse=True)
                result['elements'] = element_probs[:top_k]
            
            return result
            
        except Exception as e:
            raise Exception(f"分别预测失败: {str(e)}")
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return list(zip(self.feature_names, importance))
        return None
    
    def save_model(self, model_dir='models'):
        """保存模型"""
        try:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"LuckyNumber_{self.model_type}_{timestamp}.joblib"
            filepath = os.path.join(model_dir, filename)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'feature_names': self.feature_names,
                'raw_numbers': self.raw_numbers,
                'raw_animals': self.raw_animals,
                'raw_elements': self.raw_elements,
                'number_column': self.number_column,
                'animal_mapping': self.animal_mapping,
                'element_mapping': self.element_mapping,
                'reverse_animal_mapping': self.reverse_animal_mapping,
                'reverse_element_mapping': self.reverse_element_mapping,
                'element_numbers': self.element_numbers,
                'number_to_element': self.number_to_element,
                'X': self.X,
                'y': self.y
            }
            
            joblib.dump(model_data, filepath)
            return filepath
            
        except Exception as e:
            raise Exception(f"保存模型失败: {str(e)}")
    
    def load_model(self, filepath):
        """加载模型"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.sequence_length = model_data['sequence_length']
            self.feature_names = model_data['feature_names']
            self.raw_numbers = model_data['raw_numbers']
            self.number_column = model_data['number_column']
            
            # 加载生肖和五行相关数据
            self.raw_animals = model_data.get('raw_animals')
            self.raw_elements = model_data.get('raw_elements')
            self.animal_mapping = model_data.get('animal_mapping', {
                '鼠': 0, '牛': 1, '虎': 2, '兔': 3, '龙': 4, '蛇': 5,
                '马': 6, '羊': 7, '猴': 8, '鸡': 9, '狗': 10, '猪': 11
            })
            self.element_mapping = model_data.get('element_mapping', {'金': 0, '木': 1, '水': 2, '火': 3, '土': 4})
            self.reverse_animal_mapping = model_data.get('reverse_animal_mapping', {v: k for k, v in self.animal_mapping.items()})
            self.reverse_element_mapping = model_data.get('reverse_element_mapping', {v: k for k, v in self.element_mapping.items()})
            
            # 加载五行数字映射关系
            self.element_numbers = model_data.get('element_numbers', {
                '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
                '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
                '水': [13, 14, 21, 22, 29, 30, 43, 44],
                '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
                '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
            })
            self.number_to_element = model_data.get('number_to_element', {})
            if not self.number_to_element:
                # 重建映射
                self.number_to_element = {}
                for element, numbers in self.element_numbers.items():
                    for num in numbers:
                        self.number_to_element[num] = element
            
            # 加载训练数据
            self.X = model_data.get('X')
            self.y = model_data.get('y')
            
            return True
            
        except Exception as e:
            raise Exception(f"加载模型失败: {str(e)}")
