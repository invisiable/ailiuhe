"""
高级模型训练 - 2025版
使用314期数据，多模型集成提升预测成功率
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost未安装")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM未安装")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost未安装")


class AdvancedLuckyNumberPredictor:
    """高级幸运数字预测器"""
    
    def __init__(self, sequence_length=15):
        self.sequence_length = sequence_length
        self.models = {}
        self.scaler = StandardScaler()
        self.element_numbers = {
            '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
            '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
            '水': [13, 14, 21, 22, 29, 30, 43, 44],
            '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
            '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
        }
        
    def create_features(self, numbers, elements):
        """创建高级特征"""
        features = []
        
        # 1. 基础序列特征（最近N个数字）
        recent = list(numbers[-self.sequence_length:])
        features.extend(recent)
        
        # 2. 统计特征
        features.append(np.mean(recent))  # 平均值
        features.append(np.std(recent))   # 标准差
        features.append(np.max(recent))   # 最大值
        features.append(np.min(recent))   # 最小值
        features.append(np.median(recent)) # 中位数
        
        # 3. 趋势特征
        features.append(recent[-1] - recent[0])  # 总趋势
        features.append(recent[-1] - recent[-2] if len(recent) > 1 else 0)  # 短期趋势
        
        # 4. 差分特征
        diffs = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        features.append(np.mean(diffs))
        features.append(np.std(diffs))
        features.append(max(diffs) if diffs else 0)
        features.append(min(diffs) if diffs else 0)
        
        # 5. 区间分布特征
        bins = [(1, 10), (11, 20), (21, 29), (30, 39), (40, 49)]
        for low, high in bins:
            count = sum(1 for n in recent if low <= n <= high)
            features.append(count)
            features.append(count / len(recent))  # 比例
        
        # 6. 奇偶特征
        odd_count = sum(1 for n in recent if n % 2 == 1)
        features.append(odd_count)
        features.append(odd_count / len(recent))
        
        # 7. 五行分布特征
        for element, nums in self.element_numbers.items():
            count = sum(1 for n in recent if n in nums)
            features.append(count)
        
        # 8. 频率特征（最近10期）
        recent_10 = numbers[-10:] if len(numbers) >= 10 else numbers
        freq = Counter(recent_10)
        most_common = freq.most_common(5)
        for i in range(5):
            if i < len(most_common):
                features.append(most_common[i][0])
                features.append(most_common[i][1])
            else:
                features.append(0)
                features.append(0)
        
        # 9. 距离特征（与最近数字的距离）
        last_num = recent[-1]
        for num in range(1, 50):
            features.append(abs(num - last_num))
        
        # 10. 周期特征（位置）
        features.append(len(numbers) % 7)  # 周几
        features.append(len(numbers) % 30) # 月中位置
        
        return np.array(features)
    
    def prepare_dataset(self, df):
        """准备训练数据集"""
        numbers = df['number'].values
        elements = df['element'].values if 'element' in df.columns else [None] * len(numbers)
        
        X, y = [], []
        
        # 创建序列数据
        for i in range(self.sequence_length, len(numbers)):
            # 使用前i期数据创建特征
            train_numbers = numbers[:i]
            train_elements = elements[:i]
            
            features = self.create_features(train_numbers, train_elements)
            target = numbers[i]
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train_models(self, X_train, y_train):
        """训练多个模型"""
        print("\n" + "="*80)
        print("开始训练高级模型集成...")
        print("="*80)
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        models_to_train = []
        
        # 1. Random Forest
        models_to_train.append(('RandomForest', RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )))
        
        # 2. Gradient Boosting
        models_to_train.append(('GradientBoosting', GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=3,
            random_state=42
        )))
        
        # 3. Extra Trees
        models_to_train.append(('ExtraTrees', ExtraTreesRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        )))
        
        # 4. XGBoost (如果可用)
        if XGBOOST_AVAILABLE:
            models_to_train.append(('XGBoost', xgb.XGBRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )))
        
        # 5. LightGBM (如果可用)
        if LIGHTGBM_AVAILABLE:
            models_to_train.append(('LightGBM', lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )))
        
        # 6. CatBoost (如果可用)
        if CATBOOST_AVAILABLE:
            models_to_train.append(('CatBoost', CatBoostRegressor(
                iterations=150,
                learning_rate=0.05,
                depth=8,
                random_state=42,
                verbose=False
            )))
        
        # 训练所有模型
        for name, model in models_to_train:
            print(f"\n训练 {name}...")
            try:
                model.fit(X_train_scaled, y_train)
                self.models[name] = model
                print(f"✅ {name} 训练完成")
            except Exception as e:
                print(f"❌ {name} 训练失败: {str(e)}")
        
        print(f"\n成功训练 {len(self.models)} 个模型")
    
    def predict_top_k(self, numbers, elements, k=15):
        """预测TOP K个数字"""
        features = self.create_features(numbers, elements)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # 收集所有模型的预测
        all_predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(features_scaled)[0]
                # 四舍五入到最近的整数
                pred_int = int(round(pred))
                # 限制在1-49范围内
                pred_int = max(1, min(49, pred_int))
                
                # 记录预测
                if pred_int not in all_predictions:
                    all_predictions[pred_int] = 0
                all_predictions[pred_int] += 1
            except Exception as e:
                print(f"⚠️ {name} 预测失败: {str(e)}")
        
        # 基于频率和统计方法补充预测
        recent_10 = numbers[-10:]
        freq = Counter(recent_10)
        
        # 统计方法1: 高频数字
        for num, count in freq.most_common(10):
            if num not in all_predictions:
                all_predictions[num] = count * 0.5
            else:
                all_predictions[num] += count * 0.5
        
        # 统计方法2: 区域分布
        extreme_count = sum(1 for n in recent_10 if n <= 10 or n >= 40)
        if extreme_count > 5:  # 极端值趋势
            for n in list(range(1, 11)) + list(range(40, 50)):
                if n not in recent_10[-3:]:  # 排除最近3期
                    if n not in all_predictions:
                        all_predictions[n] = 0.3
                    else:
                        all_predictions[n] += 0.3
        
        # 排序并返回TOP K
        sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        top_k_numbers = [num for num, score in sorted_predictions[:k]]
        
        return top_k_numbers
    
    def save_models(self, prefix='advanced'):
        """保存模型"""
        import os
        from datetime import datetime
        
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存scaler
        scaler_path = f'models/{prefix}_scaler_{timestamp}.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler 已保存: {scaler_path}")
        
        # 保存每个模型
        for name, model in self.models.items():
            model_path = f'models/{prefix}_{name}_{timestamp}.pkl'
            joblib.dump(model, model_path)
            print(f"✅ {name} 已保存: {model_path}")


def main():
    """主函数"""
    print("="*80)
    print("高级幸运数字预测模型训练 - 基于314期数据")
    print("="*80)
    
    # 加载数据
    print("\n加载数据...")
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"✅ 数据加载完成: {len(df)} 期")
    print(f"   日期范围: {df.iloc[0]['date']} 至 {df.iloc[-1]['date']}")
    print(f"   最新号码: {df.iloc[-1]['number']}")
    
    # 创建预测器
    predictor = AdvancedLuckyNumberPredictor(sequence_length=15)
    
    # 准备数据集
    print("\n准备训练数据集...")
    X, y = predictor.prepare_dataset(df)
    print(f"✅ 特征维度: {X.shape}")
    print(f"   样本数: {len(X)}")
    print(f"   特征数: {X.shape[1]}")
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=False
    )
    print(f"\n训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    
    # 训练模型
    predictor.train_models(X_train, y_train)
    
    # 验证模型
    print("\n" + "="*80)
    print("模型验证 - 最近10期预测")
    print("="*80)
    
    # 使用最近10期进行验证
    validation_results = []
    for i in range(10):
        idx = -(10 - i)
        test_df = df.iloc[:idx]
        actual = df.iloc[idx]['number']
        
        numbers = test_df['number'].values
        elements = test_df['element'].values
        
        top15 = predictor.predict_top_k(numbers, elements, k=15)
        top10 = top15[:10]
        top5 = top15[:5]
        
        hit_top5 = actual in top5
        hit_top10 = actual in top10
        hit_top15 = actual in top15
        
        validation_results.append({
            'actual': actual,
            'top5': top5,
            'top15': top15,
            'hit_top5': hit_top5,
            'hit_top10': hit_top10,
            'hit_top15': hit_top15
        })
        
        status = "✅" if hit_top15 else "❌"
        rank = top15.index(actual) + 1 if hit_top15 else "-"
        print(f"{status} 期{i+1}: 实际={actual:2d} | TOP5={top5} | 命中={rank}")
    
    # 统计验证结果
    print("\n" + "="*80)
    print("验证结果统计")
    print("="*80)
    
    top5_hits = sum(1 for r in validation_results if r['hit_top5'])
    top10_hits = sum(1 for r in validation_results if r['hit_top10'])
    top15_hits = sum(1 for r in validation_results if r['hit_top15'])
    
    print(f"TOP 5  命中: {top5_hits}/10 = {top5_hits*10}%")
    print(f"TOP 10 命中: {top10_hits}/10 = {top10_hits*10}%")
    print(f"TOP 15 命中: {top15_hits}/10 = {top15_hits*10}%")
    
    # 预测下一期
    print("\n" + "="*80)
    print("预测下一期 (2025/12/14)")
    print("="*80)
    
    numbers = df['number'].values
    elements = df['element'].values
    
    top15 = predictor.predict_top_k(numbers, elements, k=15)
    print(f"\n🎯 TOP 5:  {top15[:5]}")
    print(f"📊 TOP 10: {top15[:10]}")
    print(f"📋 TOP 15: {top15}")
    
    # 保存模型
    print("\n" + "="*80)
    print("保存模型")
    print("="*80)
    predictor.save_models(prefix='advanced_v2')
    
    print("\n" + "="*80)
    print("✅ 训练完成！")
    print("="*80)


if __name__ == '__main__':
    main()
