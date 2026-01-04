"""
幸运数字奇偶性预测模块
预测下一期幸运数字是奇数还是偶数
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


class OddEvenPredictor:
    """幸运数字奇偶性预测器"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = None
        self.sequence_length = 10  # 使用过去10个数字的特征来预测
        self.feature_names = []
        
    def create_features(self, df):
        """
        创建特征工程
        基于历史数字的奇偶性、生肖、五行等创建特征
        """
        df = df.copy()
        
        # 1. 基本奇偶特征
        df['is_odd'] = (df['number'] % 2 == 1).astype(int)
        
        # 2. 历史奇偶统计特征
        for i in range(1, self.sequence_length + 1):
            df[f'prev_{i}_odd'] = df['is_odd'].shift(i)
        
        # 3. 最近N期奇数比例
        for window in [3, 5, 7, 10]:
            df[f'odd_ratio_{window}'] = df['is_odd'].rolling(window=window).mean()
        
        # 4. 奇偶连续性特征
        df['odd_streak'] = 0
        streak = 0
        for idx in range(len(df)):
            if idx == 0:
                streak = 0
            elif df['is_odd'].iloc[idx-1] == 1:
                if idx > 1 and df['is_odd'].iloc[idx-2] == 1:
                    streak = streak + 1
                else:
                    streak = 1
            else:
                streak = 0
            df.loc[df.index[idx], 'odd_streak'] = streak
        
        # 5. 生肖编码
        animal_map = {
            '鼠': 0, '牛': 1, '虎': 2, '兔': 3, '龙': 4, '蛇': 5,
            '马': 6, '羊': 7, '猴': 8, '鸡': 9, '狗': 10, '猪': 11
        }
        df['animal_code'] = df['animal'].map(animal_map)
        
        # 6. 五行编码
        element_map = {'金': 0, '木': 1, '水': 2, '火': 3, '土': 4}
        df['element_code'] = df['element'].map(element_map)
        
        # 7. 数字区间特征
        df['number_range'] = pd.cut(df['number'], bins=[0, 10, 20, 30, 40, 49], 
                                     labels=[0, 1, 2, 3, 4]).astype(int)
        
        # 8. 生肖五行的奇偶倾向
        for i in range(1, 4):
            df[f'prev_{i}_animal'] = df['animal_code'].shift(i)
            df[f'prev_{i}_element'] = df['element_code'].shift(i)
        
        # 9. 周期性特征（假设有周期规律）
        df['position_in_cycle'] = df.index % 7  # 7天周期
        
        # 10. 数字模运算特征
        df['mod_3'] = df['number'] % 3
        df['mod_5'] = df['number'] % 5
        df['mod_7'] = df['number'] % 7
        
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
    
    def train_model(self, csv_file=None, model_type='random_forest', test_size=0.2):
        """
        训练模型
        
        参数:
            csv_file: CSV数据文件路径
            model_type: 模型类型
                - 'random_forest': 随机森林分类器
                - 'gradient_boosting': 梯度提升分类器
                - 'logistic': 逻辑回归
                - 'xgboost': XGBoost分类器
                - 'lightgbm': LightGBM分类器
            test_size: 测试集比例
        """
        if csv_file:
            self.load_data(csv_file)
        
        # 准备数据
        X, y, df = self.prepare_data(self.df)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False  # 时间序列不打乱
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 选择模型
        self.model_type = model_type
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0
            )
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练模型
        print(f"开始训练 {model_type} 模型...")
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        print(f"特征数量: {len(self.feature_names)}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # 评估模型
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n模型性能:")
        print(f"训练集准确率: {train_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        
        # 详细分类报告
        print("\n测试集分类报告:")
        print(classification_report(y_test, test_pred, 
                                   target_names=['偶数', '奇数']))
        
        # 混淆矩阵
        print("\n混淆矩阵:")
        cm = confusion_matrix(y_test, test_pred)
        print(f"              预测偶数  预测奇数")
        print(f"实际偶数        {cm[0][0]:4d}     {cm[0][1]:4d}")
        print(f"实际奇数        {cm[1][0]:4d}     {cm[1][1]:4d}")
        
        # 特征重要性（如果模型支持）
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                '特征': self.feature_names,
                '重要性': self.model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            print("\n特征重要性 Top 10:")
            print(feature_importance.head(10).to_string(index=False))
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'model_type': model_type
        }
    
    def predict(self, csv_file=None):
        """
        预测下一期数字的奇偶性
        
        返回:
            dict: 包含预测结果和概率
        """
        if csv_file:
            self.load_data(csv_file)
        
        # 准备数据
        df = self.create_features(self.df)
        
        # 使用最后一行数据进行预测
        last_row = df.iloc[[-1]]
        
        feature_cols = self.feature_names
        X_pred = last_row[feature_cols].values
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # 预测
        prediction = self.model.predict(X_pred_scaled)[0]
        
        # 获取预测概率
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_pred_scaled)[0]
            even_prob = proba[0]
            odd_prob = proba[1]
        else:
            even_prob = 1 - prediction
            odd_prob = prediction
        
        result = {
            'prediction': '奇数' if prediction == 1 else '偶数',
            'prediction_value': int(prediction),
            'odd_probability': float(odd_prob),
            'even_probability': float(even_prob),
            'confidence': float(max(odd_prob, even_prob))
        }
        
        return result
    
    def get_statistics(self, csv_file=None):
        """获取历史奇偶统计信息"""
        if csv_file:
            self.load_data(csv_file)
        
        df = self.df.copy()
        df['is_odd'] = df['number'] % 2 == 1
        
        total = len(df)
        odd_count = df['is_odd'].sum()
        even_count = total - odd_count
        
        # 最近N期统计
        recent_stats = {}
        for n in [10, 20, 30, 50]:
            if len(df) >= n:
                recent_df = df.tail(n)
                recent_odd = recent_df['is_odd'].sum()
                recent_even = n - recent_odd
                recent_stats[f'recent_{n}'] = {
                    'odd_count': int(recent_odd),
                    'even_count': int(recent_even),
                    'odd_ratio': float(recent_odd / n),
                    'even_ratio': float(recent_even / n)
                }
        
        # 最长连续奇数/偶数
        max_odd_streak = 0
        max_even_streak = 0
        current_streak = 0
        current_type = None
        
        for is_odd in df['is_odd']:
            if current_type == is_odd:
                current_streak += 1
            else:
                if current_type == True:
                    max_odd_streak = max(max_odd_streak, current_streak)
                elif current_type == False:
                    max_even_streak = max(max_even_streak, current_streak)
                current_type = is_odd
                current_streak = 1
        
        # 最后检查一次
        if current_type == True:
            max_odd_streak = max(max_odd_streak, current_streak)
        elif current_type == False:
            max_even_streak = max(max_even_streak, current_streak)
        
        stats = {
            'total_count': total,
            'odd_count': int(odd_count),
            'even_count': int(even_count),
            'odd_ratio': float(odd_count / total),
            'even_ratio': float(even_count / total),
            'max_odd_streak': max_odd_streak,
            'max_even_streak': max_even_streak,
            'recent_stats': recent_stats,
            'last_5_numbers': df['number'].tail(5).tolist(),
            'last_5_odd_even': ['奇' if x % 2 == 1 else '偶' 
                                for x in df['number'].tail(5).tolist()]
        }
        
        return stats
    
    def save_model(self, save_dir='models'):
        """保存模型"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = os.path.join(save_dir, 
                                  f'OddEven_{self.model_type}_{timestamp}.joblib')
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length
        }
        
        joblib.dump(model_data, model_file)
        print(f"模型已保存到: {model_file}")
        return model_file
    
    def load_model(self, model_file):
        """加载模型"""
        model_data = joblib.load(model_file)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.sequence_length = model_data['sequence_length']
        print(f"模型已从 {model_file} 加载")


def main():
    """示例用法"""
    print("=" * 80)
    print("幸运数字奇偶性预测系统")
    print("=" * 80)
    
    predictor = OddEvenPredictor()
    
    # 1. 显示历史统计
    print("\n1. 历史奇偶统计:")
    print("-" * 80)
    stats = predictor.get_statistics('data/lucky_numbers.csv')
    print(f"总期数: {stats['total_count']}")
    print(f"奇数: {stats['odd_count']} 期 ({stats['odd_ratio']*100:.2f}%)")
    print(f"偶数: {stats['even_count']} 期 ({stats['even_ratio']*100:.2f}%)")
    print(f"最长连续奇数: {stats['max_odd_streak']} 期")
    print(f"最长连续偶数: {stats['max_even_streak']} 期")
    print(f"\n最近5期: {stats['last_5_numbers']}")
    print(f"奇偶分布: {stats['last_5_odd_even']}")
    
    # 2. 训练模型
    print("\n2. 训练预测模型:")
    print("-" * 80)
    
    # 尝试多个模型
    models = ['random_forest', 'gradient_boosting', 'logistic']
    if XGBOOST_AVAILABLE:
        models.append('xgboost')
    if LIGHTGBM_AVAILABLE:
        models.append('lightgbm')
    
    best_model = None
    best_acc = 0
    
    for model_type in models:
        print(f"\n训练 {model_type} 模型...")
        predictor = OddEvenPredictor()
        result = predictor.train_model('data/lucky_numbers.csv', 
                                       model_type=model_type, 
                                       test_size=0.2)
        
        if result['test_accuracy'] > best_acc:
            best_acc = result['test_accuracy']
            best_model = predictor
            best_model_type = model_type
    
    # 3. 使用最佳模型进行预测
    print("\n" + "=" * 80)
    print(f"3. 使用最佳模型 ({best_model_type}) 进行预测:")
    print("-" * 80)
    
    prediction = best_model.predict()
    print(f"\n预测结果: {prediction['prediction']}")
    print(f"置信度: {prediction['confidence']*100:.2f}%")
    print(f"奇数概率: {prediction['odd_probability']*100:.2f}%")
    print(f"偶数概率: {prediction['even_probability']*100:.2f}%")
    
    # 4. 保存模型
    print("\n4. 保存模型:")
    print("-" * 80)
    model_file = best_model.save_model()
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
