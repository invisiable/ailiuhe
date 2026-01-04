"""
改进的幸运数字奇偶性预测模块
使用集成学习和更多特征提升预测准确率
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              VotingClassifier, StackingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib
import os
from datetime import datetime
from collections import Counter

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
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class ImprovedOddEvenPredictor:
    """改进的幸运数字奇偶性预测器"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = None
        self.sequence_length = 15  # 增加序列长度
        self.feature_names = []
        
    def create_features(self, df):
        """
        创建更丰富的特征工程
        """
        df = df.copy()
        
        # 1. 基本奇偶特征
        df['is_odd'] = (df['number'] % 2 == 1).astype(int)
        
        # 2. 历史奇偶统计特征（增加序列长度）
        for i in range(1, self.sequence_length + 1):
            df[f'prev_{i}_odd'] = df['is_odd'].shift(i)
        
        # 3. 最近N期奇数比例（更多窗口）
        for window in [3, 5, 7, 10, 15, 20]:
            df[f'odd_ratio_{window}'] = df['is_odd'].rolling(window=window).mean()
        
        # 4. 奇偶连续性特征
        df['odd_streak'] = 0
        df['even_streak'] = 0
        odd_streak = 0
        even_streak = 0
        for idx in range(len(df)):
            if idx == 0:
                odd_streak = 0
                even_streak = 0
            elif df['is_odd'].iloc[idx-1] == 1:
                odd_streak = odd_streak + 1 if idx > 1 and df['is_odd'].iloc[idx-2] == 1 else 1
                even_streak = 0
            else:
                even_streak = even_streak + 1 if idx > 1 and df['is_odd'].iloc[idx-2] == 0 else 1
                odd_streak = 0
            df.loc[df.index[idx], 'odd_streak'] = odd_streak
            df.loc[df.index[idx], 'even_streak'] = even_streak
        
        # 5. 最长连续奇数/偶数记录（近N期）
        for window in [5, 10, 15]:
            df[f'max_odd_streak_{window}'] = df['odd_streak'].rolling(window=window).max()
            df[f'max_even_streak_{window}'] = df['even_streak'].rolling(window=window).max()
        
        # 6. 生肖编码及相关特征
        animal_map = {
            '鼠': 0, '牛': 1, '虎': 2, '兔': 3, '龙': 4, '蛇': 5,
            '马': 6, '羊': 7, '猴': 8, '鸡': 9, '狗': 10, '猪': 11
        }
        df['animal_code'] = df['animal'].map(animal_map)
        
        # 生肖的奇偶倾向统计
        animal_odd_tendency = {}
        for animal in animal_map.keys():
            mask = df['animal'] == animal
            if mask.sum() > 0:
                animal_odd_tendency[animal] = df[mask]['is_odd'].mean()
        df['animal_odd_tendency'] = df['animal'].map(animal_odd_tendency)
        
        # 7. 五行编码及相关特征
        element_map = {'金': 0, '木': 1, '水': 2, '火': 3, '土': 4}
        df['element_code'] = df['element'].map(element_map)
        
        # 五行的奇偶倾向统计
        element_odd_tendency = {}
        for element in element_map.keys():
            mask = df['element'] == element
            if mask.sum() > 0:
                element_odd_tendency[element] = df[mask]['is_odd'].mean()
        df['element_odd_tendency'] = df['element'].map(element_odd_tendency)
        
        # 8. 数字区间特征（更细粒度）
        df['number_range'] = pd.cut(df['number'], bins=[0, 10, 20, 30, 40, 50], 
                                     labels=[0, 1, 2, 3, 4]).astype(int)
        df['number_range_fine'] = pd.cut(df['number'], bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], 
                                          labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(int)
        
        # 每个区间的奇偶倾向
        for i in range(5):
            mask = df['number_range'] == i
            if mask.sum() > 0:
                df.loc[mask, 'range_odd_tendency'] = df[mask]['is_odd'].mean()
        df['range_odd_tendency'] = df['range_odd_tendency'].fillna(0.5)
        
        # 9. 生肖五行的历史特征
        for i in range(1, 6):
            df[f'prev_{i}_animal'] = df['animal_code'].shift(i)
            df[f'prev_{i}_element'] = df['element_code'].shift(i)
        
        # 10. 周期性特征
        df['position_in_cycle'] = df.index % 7  # 7天周期
        df['position_in_cycle_14'] = df.index % 14  # 14天周期
        df['position_in_cycle_30'] = df.index % 30  # 30天周期
        
        # 11. 数字模运算特征
        df['mod_3'] = df['number'] % 3
        df['mod_5'] = df['number'] % 5
        df['mod_7'] = df['number'] % 7
        
        # 12. 奇偶交替模式特征
        df['odd_even_alternation'] = 0
        for idx in range(2, len(df)):
            if (df['is_odd'].iloc[idx-2] == 1 and df['is_odd'].iloc[idx-1] == 0) or \
               (df['is_odd'].iloc[idx-2] == 0 and df['is_odd'].iloc[idx-1] == 1):
                df.loc[df.index[idx], 'odd_even_alternation'] = 1
        
        # 13. 波动性特征
        for window in [5, 10]:
            df[f'odd_std_{window}'] = df['is_odd'].rolling(window=window).std()
            df[f'number_std_{window}'] = df['number'].rolling(window=window).std()
        
        # 14. 数字大小与奇偶的关系
        df['number_normalized'] = (df['number'] - df['number'].min()) / (df['number'].max() - df['number'].min())
        df['number_large'] = (df['number'] > 24).astype(int)  # 大于中位数
        
        # 15. 统计学特征
        for window in [5, 10, 15]:
            df[f'odd_skew_{window}'] = df['is_odd'].rolling(window=window).apply(lambda x: pd.Series(x).skew())
            df[f'odd_kurt_{window}'] = df['is_odd'].rolling(window=window).apply(lambda x: pd.Series(x).kurt())
        
        # 16. 组合生肖和五行特征
        df['animal_element_combo'] = df['animal_code'] * 10 + df['element_code']
        
        # 17. 前N期奇数个数
        for window in [3, 5, 7, 10]:
            df[f'odd_count_{window}'] = df['is_odd'].rolling(window=window).sum()
        
        # 18. 最近奇偶数的间隔
        df['last_odd_distance'] = 0
        df['last_even_distance'] = 0
        last_odd_idx = -1
        last_even_idx = -1
        for idx in range(len(df)):
            if df['is_odd'].iloc[idx] == 1:
                last_odd_idx = idx
            else:
                last_even_idx = idx
            
            if last_odd_idx >= 0:
                df.loc[df.index[idx], 'last_odd_distance'] = idx - last_odd_idx
            if last_even_idx >= 0:
                df.loc[df.index[idx], 'last_even_distance'] = idx - last_even_idx
        
        return df
    
    def prepare_data(self, df):
        """准备训练数据"""
        df = self.create_features(df)
        
        # 删除包含NaN的行
        df = df.dropna()
        
        # 定义特征列
        feature_cols = [col for col in df.columns if col not in 
                       ['date', 'number', 'animal', 'element', 'is_odd']]
        
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['is_odd'].values
        
        return X, y, df
    
    def load_data(self, csv_file):
        """加载数据"""
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        return self.df
    
    def train_model(self, csv_file=None, model_type='ensemble_voting', test_size=0.2):
        """
        训练改进的模型
        
        参数:
            model_type: 模型类型
                - 'ensemble_voting': 投票集成（多个模型投票）
                - 'ensemble_stacking': 堆叠集成（多层模型）
                - 'catboost': CatBoost分类器
                - 'neural_network': 神经网络
                - 'svm': 支持向量机
                - 'adaboost': AdaBoost分类器
                - 其他原有模型类型
        """
        if csv_file:
            self.load_data(csv_file)
        
        # 准备数据
        X, y, df = self.prepare_data(self.df)
        
        print(f"特征数量: {X.shape[1]}")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 选择模型
        self.model_type = model_type
        
        print(f"开始训练 {model_type} 模型...")
        
        if model_type == 'ensemble_voting':
            # 投票集成：多个强模型投票
            estimators = []
            
            # 添加梯度提升
            estimators.append(('gb', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            )))
            
            # 添加随机森林
            estimators.append(('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )))
            
            # 添加XGBoost
            if XGBOOST_AVAILABLE:
                estimators.append(('xgb', xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )))
            
            # 添加LightGBM
            if LIGHTGBM_AVAILABLE:
                estimators.append(('lgb', lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )))
            
            # 添加逻辑回归
            estimators.append(('lr', LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0
            )))
            
            self.model = VotingClassifier(
                estimators=estimators,
                voting='soft',  # 软投票，使用概率
                n_jobs=-1
            )
            
        elif model_type == 'ensemble_stacking':
            # 堆叠集成：多层模型
            base_estimators = [
                ('gb', GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ))
            ]
            
            if XGBOOST_AVAILABLE:
                base_estimators.append(('xgb', xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )))
            
            if LIGHTGBM_AVAILABLE:
                base_estimators.append(('lgb', lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )))
            
            self.model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                n_jobs=-1
            )
            
        elif model_type == 'catboost' and CATBOOST_AVAILABLE:
            self.model = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=False
            )
            
        elif model_type == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
            
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            
        elif model_type == 'adaboost':
            self.model = AdaBoostClassifier(
                base_estimator=GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3,
                    random_state=42
                ),
                n_estimators=100,
                learning_rate=0.5,
                random_state=42
            )
            
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=4,
                random_state=42
            )
            
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        
        # 评估模型
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n模型性能:")
        print(f"训练集准确率: {train_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        
        print(f"\n测试集分类报告:")
        target_names = ['偶数', '奇数']
        print(classification_report(y_test, test_pred, target_names=target_names))
        
        print(f"混淆矩阵:")
        cm = confusion_matrix(y_test, test_pred)
        cm_df = pd.DataFrame(cm, index=['实际偶数', '实际奇数'], columns=['预测偶数', '预测奇数'])
        print(cm_df)
        
        # 特征重要性（如果模型支持）
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                '特征': self.feature_names,
                '重要性': self.model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            print(f"\n特征重要性 Top 15:")
            print(feature_importance.head(15))
        
        return test_acc
    
    def predict(self, csv_file=None):
        """预测下一期的奇偶性"""
        if csv_file:
            self.load_data(csv_file)
        
        # 准备数据
        X, y, df = self.prepare_data(self.df)
        
        # 使用最后一行数据进行预测
        last_features = X[-1].reshape(1, -1)
        last_features_scaled = self.scaler.transform(last_features)
        
        # 预测
        prediction = self.model.predict(last_features_scaled)[0]
        probability = self.model.predict_proba(last_features_scaled)[0]
        
        result = {
            'prediction': '奇数' if prediction == 1 else '偶数',
            'probability': probability[1] if prediction == 1 else probability[0],
            'odd_probability': probability[1],
            'even_probability': probability[0]
        }
        
        print(f"\n预测结果: {result['prediction']} (置信度: {result['probability']*100:.2f}%)")
        print(f"详细概率 - 奇数: {result['odd_probability']*100:.2f}%, 偶数: {result['even_probability']*100:.2f}%")
        
        return result
    
    def save_model(self, filename=None):
        """保存模型"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"improved_odd_even_{self.model_type}_{timestamp}.joblib"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length
        }
        
        joblib.dump(model_data, filename)
        print(f"模型已保存到: {filename}")
        return filename
    
    def load_model(self, filename):
        """加载模型"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.sequence_length = model_data['sequence_length']
        print(f"模型已加载: {filename}")


if __name__ == "__main__":
    # 测试改进的预测器
    predictor = ImprovedOddEvenPredictor()
    
    # 测试不同模型
    models_to_test = [
        'ensemble_voting',
        'ensemble_stacking',
        'neural_network',
        'catboost',
        'gradient_boosting',
        'xgboost',
        'lightgbm'
    ]
    
    print("=" * 80)
    print("测试改进的奇偶预测模型")
    print("=" * 80)
    
    results = {}
    
    for model_type in models_to_test:
        try:
            print(f"\n{'='*80}")
            print(f"测试模型: {model_type}")
            print(f"{'='*80}")
            
            predictor = ImprovedOddEvenPredictor()
            test_acc = predictor.train_model('data/lucky_numbers.csv', model_type=model_type)
            results[model_type] = test_acc
            
            # 进行预测
            predictor.predict()
            
        except Exception as e:
            print(f"模型 {model_type} 训练失败: {e}")
            results[model_type] = 0
    
    # 总结结果
    print("\n" + "=" * 80)
    print("模型性能总结")
    print("=" * 80)
    for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model:25s}: {acc:.4f}")
