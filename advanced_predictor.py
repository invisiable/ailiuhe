"""
高级预测模型 - 尝试提升预测成功率
策略：
1. 深度学习模型 (LSTM)
2. 时间序列分解
3. 集成更多模型
4. 动态权重调整
5. 区间预测策略
"""
import sys
sys.path.insert(0, 'd:\\AIagent')

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    advanced_models_available = True
except:
    advanced_models_available = False


class AdvancedPredictor:
    """高级预测器 - 多策略融合"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.raw_numbers = []
        self.sequence_length = 10
        
    def create_advanced_features(self, numbers):
        """创建高级特征"""
        features = []
        
        # 基础特征：最近N个数字
        recent = list(numbers[-self.sequence_length:])
        features.extend(recent)
        
        # 统计特征
        features.append(np.mean(recent))
        features.append(np.std(recent))
        features.append(np.max(recent))
        features.append(np.min(recent))
        features.append(recent[-1] - recent[0])  # 趋势
        
        # 差分特征（连续数字的变化）
        diffs = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        features.append(np.mean(diffs))
        features.append(np.std(diffs))
        features.append(max(diffs))
        features.append(min(diffs))
        
        # 区间特征（数字分布）
        bins = [0, 10, 20, 30, 40, 50]
        for i in range(len(bins)-1):
            count = sum(1 for n in recent if bins[i] <= n < bins[i+1])
            features.append(count)
        
        # 奇偶特征
        odd_count = sum(1 for n in recent if n % 2 == 1)
        features.append(odd_count)
        features.append(len(recent) - odd_count)
        
        # 距离特征（与最近数字的距离）
        last_num = recent[-1]
        for distance in [1, 2, 3, 5, 10]:
            features.append(1 if any(abs(n - last_num) <= distance for n in recent[:-1]) else 0)
        
        return np.array(features)
    
    def load_and_prepare_data(self, file_path):
        """加载并准备数据"""
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        self.raw_numbers = df['number'].values
        
        X_list, y_list = [], []
        for i in range(self.sequence_length, len(self.raw_numbers)):
            features = self.create_advanced_features(self.raw_numbers[:i])
            target = self.raw_numbers[i]
            X_list.append(features)
            y_list.append(target)
        
        return np.array(X_list), np.array(y_list)
    
    def train_ensemble_models(self, X_train, y_train):
        """训练集成模型"""
        print("训练高级集成模型...")
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['main'] = scaler
        
        # 模型1: 梯度提升（调优参数）
        print("  - 梯度提升 (优化参数)")
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        self.models['gb'].fit(X_train, y_train)
        
        # 模型2: 随机森林（增强版）
        print("  - 随机森林 (增强版)")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42
        )
        self.models['rf'].fit(X_train, y_train)
        
        # 模型3: 极端随机树
        print("  - 极端随机树")
        self.models['et'] = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            random_state=42
        )
        self.models['et'].fit(X_train, y_train)
        
        # 模型4: 深度神经网络
        print("  - 深度神经网络")
        self.models['dnn'] = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        self.models['dnn'].fit(X_train_scaled, y_train)
        
        if advanced_models_available:
            # 模型5: XGBoost（调优）
            print("  - XGBoost (优化参数)")
            self.models['xgb'] = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            self.models['xgb'].fit(X_train, y_train)
            
            # 模型6: LightGBM（调优）
            print("  - LightGBM (优化参数)")
            self.models['lgb'] = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                random_state=42,
                verbose=-1
            )
            self.models['lgb'].fit(X_train, y_train)
            
            # 模型7: CatBoost
            print("  - CatBoost")
            self.models['cat'] = CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                random_state=42,
                verbose=0
            )
            self.models['cat'].fit(X_train, y_train)
        
        print(f"✓ 训练完成，共{len(self.models)}个模型")
    
    def predict_with_strategy(self, strategy='ensemble', top_k=10):
        """使用不同策略预测"""
        recent_features = self.create_advanced_features(self.raw_numbers)
        recent_features_scaled = self.scalers['main'].transform([recent_features])
        
        predictions = {}
        
        if strategy in ['ensemble', 'all']:
            # 策略1: 集成预测
            ensemble_preds = []
            for name, model in self.models.items():
                if name == 'dnn':
                    pred = model.predict(recent_features_scaled)[0]
                else:
                    pred = model.predict([recent_features])[0]
                ensemble_preds.append(pred)
            
            avg_pred = np.mean(ensemble_preds)
            predictions['ensemble'] = int(round(np.clip(avg_pred, 1, 49)))
        
        if strategy in ['neighbors', 'all']:
            # 策略2: 邻近数字（基于最近预测的±范围）
            base_pred = predictions.get('ensemble', 25)
            neighbors = []
            for offset in range(-10, 11):
                num = base_pred + offset
                if 1 <= num <= 49:
                    neighbors.append(num)
            predictions['neighbors'] = neighbors[:top_k]
        
        if strategy in ['frequency', 'all']:
            # 策略3: 频率分析（最近30期）
            recent_30 = self.raw_numbers[-30:]
            from collections import Counter
            freq = Counter(recent_30)
            # 反向：选择出现少的
            all_nums = set(range(1, 50))
            rare_nums = all_nums - set(recent_30)
            if rare_nums:
                predictions['rare'] = sorted(rare_nums)[:top_k]
            else:
                predictions['rare'] = sorted(freq.items(), key=lambda x: x[1])[:top_k]
        
        if strategy in ['intervals', 'all']:
            # 策略4: 区间平衡（选择未充分代表的区间）
            recent_30 = self.raw_numbers[-30:]
            interval_counts = {
                '1-10': sum(1 for n in recent_30 if 1 <= n <= 10),
                '11-20': sum(1 for n in recent_30 if 11 <= n <= 20),
                '21-30': sum(1 for n in recent_30 if 21 <= n <= 30),
                '31-40': sum(1 for n in recent_30 if 31 <= n <= 40),
                '41-49': sum(1 for n in recent_30 if 41 <= n <= 49),
            }
            
            # 找出出现最少的区间
            min_interval = min(interval_counts.items(), key=lambda x: x[1])[0]
            interval_ranges = {
                '1-10': range(1, 11),
                '11-20': range(11, 21),
                '21-30': range(21, 31),
                '31-40': range(31, 41),
                '41-49': range(41, 50),
            }
            predictions['interval'] = list(interval_ranges[min_interval])[:top_k]
        
        return predictions
    
    def comprehensive_predict(self, top_k=10):
        """综合多策略预测"""
        # 获取所有策略的预测
        all_strategies = self.predict_with_strategy('all', top_k=20)
        
        # 统计每个数字的出现次数和来源
        number_scores = {}
        
        # 集成预测（权重最高）
        if 'ensemble' in all_strategies:
            pred = all_strategies['ensemble']
            number_scores[pred] = number_scores.get(pred, 0) + 10
        
        # 邻近数字
        if 'neighbors' in all_strategies:
            for i, num in enumerate(all_strategies['neighbors'][:top_k]):
                score = 8 - i * 0.5
                number_scores[num] = number_scores.get(num, 0) + score
        
        # 稀有数字
        if 'rare' in all_strategies:
            for num in all_strategies['rare'][:top_k]:
                number_scores[num] = number_scores.get(num, 0) + 5
        
        # 区间平衡
        if 'interval' in all_strategies:
            for num in all_strategies['interval'][:top_k]:
                number_scores[num] = number_scores.get(num, 0) + 3
        
        # 排序并返回Top K
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for num, score in sorted_numbers[:top_k]:
            results.append({
                'number': num,
                'score': score,
                'probability': score / sum(s for _, s in sorted_numbers[:top_k])
            })
        
        return results


def validate_advanced_model(test_periods=10):
    """验证高级模型"""
    print("=" * 80)
    print("高级预测模型验证")
    print("=" * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_records = len(df)
    
    print(f"\n数据集: {total_records}期")
    print(f"验证: 最近{test_periods}期")
    print(f"训练样本: {total_records - test_periods}期\n")
    
    results = []
    top5_hits = 0
    top10_hits = 0
    
    for i in range(test_periods):
        test_index = total_records - test_periods + i
        period_num = test_index + 1
        
        # 准备数据
        train_df = df.iloc[:test_index]
        actual = df.iloc[test_index]['number']
        actual_date = df.iloc[test_index]['date']
        
        print(f"{'='*80}")
        print(f"测试第{period_num}期 ({actual_date})")
        
        # 保存临时训练数据
        temp_file = f'data/temp_advanced_{i}.csv'
        train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        
        try:
            # 创建并训练模型
            predictor = AdvancedPredictor()
            X, y = predictor.load_and_prepare_data(temp_file)
            
            # 分割训练集
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            
            predictor.train_ensemble_models(X_train, y_train)
            
            # 预测
            predictions = predictor.comprehensive_predict(top_k=10)
            top10 = [p['number'] for p in predictions]
            top5 = top10[:5]
            
            print(f"\nTop 5: {top5}")
            print(f"Top 10: {top10}")
            print(f"实际: {actual}")
            
            # 检查命中
            if actual in top5:
                rank = top5.index(actual) + 1
                status = f"✅ Top 5命中 (第{rank}名)"
                top5_hits += 1
                top10_hits += 1
            elif actual in top10:
                rank = top10.index(actual) + 1
                status = f"✓ Top 10命中 (第{rank}名)"
                top10_hits += 1
            else:
                status = "❌ 未命中"
            
            print(f"结果: {status}\n")
            
            results.append({
                'period': period_num,
                'date': actual_date,
                'actual': actual,
                'top5': top5,
                'top10': top10,
                'hit': actual in top10
            })
            
            # 清理
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            print(f"❌ 错误: {e}\n")
    
    # 统计
    print("=" * 80)
    print("统计结果")
    print("=" * 80)
    print(f"\nTop 5 命中: {top5_hits}/{test_periods} = {top5_hits/test_periods*100:.1f}%")
    print(f"Top 10 命中: {top10_hits}/{test_periods} = {top10_hits/test_periods*100:.1f}%")
    
    return {
        'top5_rate': top5_hits/test_periods*100,
        'top10_rate': top10_hits/test_periods*100,
        'results': results
    }


if __name__ == "__main__":
    try:
        print("启动高级预测模型验证...\n")
        results = validate_advanced_model(test_periods=10)
        print(f"\n✅ 完成！Top 5: {results['top5_rate']:.1f}%, Top 10: {results['top10_rate']:.1f}%")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
