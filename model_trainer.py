"""
模型训练核心模块
提供机器学习模型的训练、评估和保存功能
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import joblib
import os
from datetime import datetime


class ModelTrainer:
    """模型训练器类"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model_type = None
        self.feature_names = None
        self.target_name = None
        
    def load_data(self, file_path, target_column):
        """
        加载训练数据
        
        Args:
            file_path: CSV文件路径
            target_column: 目标列名称
            
        Returns:
            成功返回True，失败返回False
        """
        try:
            # 读取CSV文件
            self.data = pd.read_csv(file_path)
            
            if target_column not in self.data.columns:
                raise ValueError(f"目标列 '{target_column}' 不存在于数据中")
            
            self.target_name = target_column
            self.feature_names = [col for col in self.data.columns if col != target_column]
            
            # 分离特征和目标
            X = self.data[self.feature_names]
            y = self.data[target_column]
            
            # 处理分类特征
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            
            # 处理目标变量（如果是分类问题）
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                self.label_encoders[target_column] = le
                self.is_classification = True
            else:
                self.is_classification = False
            
            self.X = X
            self.y = y
            
            return True
            
        except Exception as e:
            raise Exception(f"加载数据失败: {str(e)}")
    
    def train_model(self, model_type='auto', test_size=0.2, random_state=42):
        """
        训练模型
        
        Args:
            model_type: 模型类型 ('auto', 'linear', 'logistic', 'random_forest')
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练结果字典
        """
        try:
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )
            
            # 标准化特征
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 自动选择模型
            if model_type == 'auto':
                if self.is_classification:
                    model_type = 'random_forest_classifier'
                else:
                    model_type = 'random_forest_regressor'
            
            # 创建模型
            if model_type == 'linear':
                self.model = LinearRegression()
                self.model_type = 'LinearRegression'
            elif model_type == 'logistic':
                self.model = LogisticRegression(max_iter=1000)
                self.model_type = 'LogisticRegression'
            elif model_type == 'random_forest_classifier':
                self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                self.model_type = 'RandomForestClassifier'
            elif model_type == 'random_forest_regressor':
                self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                self.model_type = 'RandomForestRegressor'
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 训练模型
            self.model.fit(X_train_scaled, y_train)
            
            # 预测
            y_pred = self.model.predict(X_test_scaled)
            
            # 评估模型
            results = {
                'model_type': self.model_type,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(self.feature_names),
                'features': self.feature_names
            }
            
            if self.is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                results['accuracy'] = accuracy
                results['metrics'] = f"准确率: {accuracy:.4f}"
                
                # 如果有标签编码器，获取原始标签
                if self.target_name in self.label_encoders:
                    le = self.label_encoders[self.target_name]
                    y_test_labels = le.inverse_transform(y_test)
                    y_pred_labels = le.inverse_transform(y_pred.astype(int))
                    results['classification_report'] = classification_report(
                        y_test_labels, y_pred_labels
                    )
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results['mse'] = mse
                results['r2'] = r2
                results['metrics'] = f"均方误差: {mse:.4f}, R²分数: {r2:.4f}"
            
            return results
            
        except Exception as e:
            raise Exception(f"训练模型失败: {str(e)}")
    
    def save_model(self, model_dir='models'):
        """
        保存模型
        
        Args:
            model_dir: 模型保存目录
            
        Returns:
            保存的文件路径
        """
        try:
            # 创建模型目录
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_{timestamp}.joblib"
            filepath = os.path.join(model_dir, filename)
            
            # 保存模型和相关信息
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'is_classification': self.is_classification
            }
            
            joblib.dump(model_data, filepath)
            
            return filepath
            
        except Exception as e:
            raise Exception(f"保存模型失败: {str(e)}")
    
    def load_model(self, filepath):
        """
        加载已保存的模型
        
        Args:
            filepath: 模型文件路径
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.target_name = model_data['target_name']
            self.is_classification = model_data['is_classification']
            
            return True
            
        except Exception as e:
            raise Exception(f"加载模型失败: {str(e)}")
    
    def predict(self, X_new):
        """
        使用训练好的模型进行预测
        
        Args:
            X_new: 新数据（DataFrame或ndarray）
            
        Returns:
            预测结果
        """
        try:
            if self.model is None:
                raise ValueError("模型尚未训练")
            
            # 确保特征顺序正确
            if isinstance(X_new, pd.DataFrame):
                X_new = X_new[self.feature_names]
            
            # 标准化
            X_new_scaled = self.scaler.transform(X_new)
            
            # 预测
            predictions = self.model.predict(X_new_scaled)
            
            # 如果是分类问题，转换回原始标签
            if self.is_classification and self.target_name in self.label_encoders:
                le = self.label_encoders[self.target_name]
                predictions = le.inverse_transform(predictions.astype(int))
            
            return predictions
            
        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")
