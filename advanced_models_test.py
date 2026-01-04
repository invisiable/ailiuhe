"""
高级模型测试 - 针对时序数据的流行模型
包括：LSTM, Transformer, Prophet, ARIMA等
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 尝试导入深度学习库
try:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except:
    KERAS_AVAILABLE = False

# 尝试导入时序分析库
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False


class AdvancedModelsTester:
    """高级模型测试器"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def prepare_data(self, df, test_size=10):
        """准备数据"""
        numbers = df['number'].values
        
        # 分割训练和测试
        train_numbers = numbers[:-test_size]
        test_numbers = numbers[-test_size:]
        
        return train_numbers, test_numbers
    
    def create_sequences(self, data, seq_length=10):
        """创建序列数据"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    # ========== 模型1: LSTM神经网络 ==========
    def test_lstm(self, train_data, test_data):
        """测试LSTM模型"""
        if not KERAS_AVAILABLE:
            return None
        
        print("\n[模型1] LSTM 神经网络")
        print("-" * 60)
        
        try:
            seq_length = 10
            X_train, y_train = self.create_sequences(train_data, seq_length)
            
            # 归一化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            # 构建LSTM模型
            model = Sequential([
                LSTM(64, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_scaled.reshape(-1, seq_length, 1), 
                     y_train_scaled, 
                     epochs=50, 
                     batch_size=32, 
                     verbose=0)
            
            print("   训练完成")
            
            # 测试预测
            predictions = []
            for i in range(len(test_data)):
                if i == 0:
                    seq = train_data[-seq_length:]
                else:
                    seq = np.concatenate([train_data[-seq_length+i:], test_data[:i]])
                
                seq_scaled = scaler.transform(seq.reshape(-1, 1)).flatten()
                pred = model.predict(seq_scaled.reshape(1, seq_length, 1), verbose=0)
                pred_original = scaler.inverse_transform(pred)[0][0]
                predictions.append(int(np.clip(pred_original, 1, 49)))
            
            return predictions
            
        except Exception as e:
            print(f"   错误: {str(e)}")
            return None
    
    # ========== 模型2: 集成投票 ==========
    def test_ensemble_voting(self, train_data, test_data):
        """集成投票模型"""
        print("\n[模型2] 集成投票模型")
        print("-" * 60)
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        try:
            import xgboost as xgb
            import lightgbm as lgb
        except:
            print("   缺少XGBoost/LightGBM库")
            return None
        
        try:
            seq_length = 10
            X_train, y_train = self.create_sequences(train_data, seq_length)
            
            # 训练多个模型
            models = [
                ('RF', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('GB', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('XGB', xgb.XGBRegressor(n_estimators=100, random_state=42)),
                ('LGB', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
            ]
            
            trained_models = []
            for name, model in models:
                model.fit(X_train, y_train)
                trained_models.append((name, model))
                print(f"   {name} 训练完成")
            
            # 预测
            predictions = []
            for i in range(len(test_data)):
                if i == 0:
                    seq = train_data[-seq_length:]
                else:
                    seq = np.concatenate([train_data[-seq_length+i:], test_data[:i]])
                
                preds = [model.predict(seq.reshape(1, -1))[0] for _, model in trained_models]
                avg_pred = int(np.clip(np.mean(preds), 1, 49))
                predictions.append(avg_pred)
            
            return predictions
            
        except Exception as e:
            print(f"   错误: {str(e)}")
            return None
    
    # ========== 模型3: 频率+趋势混合 ==========
    def test_frequency_trend(self, train_data, test_data):
        """频率+趋势混合模型"""
        print("\n[模型3] 频率+趋势混合模型")
        print("-" * 60)
        
        try:
            predictions = []
            
            for i in range(len(test_data)):
                # 获取历史数据
                if i == 0:
                    history = train_data
                else:
                    history = np.concatenate([train_data, test_data[:i]])
                
                # 最近30期频率分析
                recent_30 = history[-30:]
                freq_counter = Counter(recent_30)
                
                # 最近10期趋势分析
                recent_10 = history[-10:]
                
                # 区域分析
                zones = {
                    'extreme_small': sum(1 for n in recent_10 if 1 <= n <= 10),
                    'small': sum(1 for n in recent_10 if 11 <= n <= 20),
                    'mid': sum(1 for n in recent_10 if 21 <= n <= 30),
                    'large': sum(1 for n in recent_10 if 31 <= n <= 40),
                    'extreme_large': sum(1 for n in recent_10 if 41 <= n <= 49)
                }
                
                # 根据趋势选择候选区域
                if zones['extreme_small'] + zones['extreme_large'] >= 5:
                    # 极端值趋势
                    candidates = list(range(1, 11)) + list(range(40, 50))
                elif zones['mid'] >= 5:
                    # 中间值趋势
                    candidates = list(range(21, 31))
                else:
                    # 混合
                    candidates = list(range(1, 50))
                
                # 在候选中选择频率最高的
                candidate_freq = {n: freq_counter.get(n, 0) for n in candidates}
                best = max(candidate_freq.items(), key=lambda x: x[1])[0]
                
                predictions.append(best)
            
            print("   预测完成")
            return predictions
            
        except Exception as e:
            print(f"   错误: {str(e)}")
            return None
    
    # ========== 模型4: 马尔可夫链 ==========
    def test_markov_chain(self, train_data, test_data):
        """马尔可夫链模型"""
        print("\n[模型4] 马尔可夫链模型")
        print("-" * 60)
        
        try:
            # 构建转移矩阵
            transition_matrix = np.zeros((49, 49))
            
            for i in range(len(train_data) - 1):
                current = int(train_data[i]) - 1
                next_val = int(train_data[i+1]) - 1
                transition_matrix[current][next_val] += 1
            
            # 归一化
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零
            transition_matrix = transition_matrix / row_sums
            
            print("   转移矩阵构建完成")
            
            # 预测
            predictions = []
            for i in range(len(test_data)):
                if i == 0:
                    last = int(train_data[-1]) - 1
                else:
                    last = int(test_data[i-1]) - 1
                
                # 获取最可能的下一个状态
                probs = transition_matrix[last]
                if probs.sum() > 0:
                    pred = np.argmax(probs) + 1
                else:
                    pred = np.random.randint(1, 50)
                
                predictions.append(pred)
            
            print("   预测完成")
            return predictions
            
        except Exception as e:
            print(f"   错误: {str(e)}")
            return None
    
    # ========== 模型5: 模式识别 ==========
    def test_pattern_recognition(self, train_data, test_data):
        """模式识别模型"""
        print("\n[模型5] 模式识别模型")
        print("-" * 60)
        
        try:
            predictions = []
            
            for i in range(len(test_data)):
                if i == 0:
                    history = train_data
                else:
                    history = np.concatenate([train_data, test_data[:i]])
                
                # 寻找相似模式
                seq_length = 5
                if len(history) < seq_length:
                    predictions.append(np.random.randint(1, 50))
                    continue
                
                last_pattern = history[-seq_length:]
                
                # 在历史中寻找相似模式
                best_match_idx = -1
                min_distance = float('inf')
                
                for j in range(len(history) - seq_length - 1):
                    pattern = history[j:j+seq_length]
                    distance = np.sum(np.abs(pattern - last_pattern))
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match_idx = j
                
                # 使用匹配后的下一个值
                if best_match_idx >= 0:
                    pred = int(history[best_match_idx + seq_length])
                else:
                    pred = int(np.mean(history[-10:]))
                
                predictions.append(np.clip(pred, 1, 49))
            
            print("   预测完成")
            return predictions
            
        except Exception as e:
            print(f"   错误: {str(e)}")
            return None
    
    # ========== 评估方法 ==========
    def evaluate_top_k(self, predictions, actual, k=15):
        """评估Top K命中率"""
        # 对于单个预测值，扩展为Top K候选
        # 使用预测值周围的数字作为候选
        top_k_candidates = []
        
        for pred in predictions:
            candidates = set([pred])
            
            # 添加周围的数字
            for offset in range(1, k):
                if pred - offset >= 1:
                    candidates.add(pred - offset)
                if pred + offset <= 49:
                    candidates.add(pred + offset)
                if len(candidates) >= k:
                    break
            
            top_k_candidates.append(list(candidates)[:k])
        
        # 计算命中率
        hits = sum(1 for i, actual_val in enumerate(actual) 
                  if actual_val in top_k_candidates[i])
        
        return hits / len(actual) * 100
    
    def run_all_tests(self, df):
        """运行所有测试"""
        print("=" * 80)
        print("高级模型测试 - 时序数据专用模型")
        print("=" * 80)
        
        print(f"\n数据集: {len(df)}期")
        print(f"测试: 最近10期")
        
        # 准备数据
        train_data, test_data = self.prepare_data(df, test_size=10)
        
        print(f"\n训练集: {len(train_data)}期")
        print(f"测试集: {len(test_data)}期")
        print(f"实际值: {test_data.tolist()}")
        
        # 测试所有模型
        models_to_test = [
            ('集成投票', lambda: self.test_ensemble_voting(train_data, test_data)),
            ('频率+趋势', lambda: self.test_frequency_trend(train_data, test_data)),
            ('马尔可夫链', lambda: self.test_markov_chain(train_data, test_data)),
            ('模式识别', lambda: self.test_pattern_recognition(train_data, test_data)),
        ]
        
        if KERAS_AVAILABLE:
            models_to_test.insert(0, ('LSTM', lambda: self.test_lstm(train_data, test_data)))
        
        results = {}
        
        for model_name, test_func in models_to_test:
            predictions = test_func()
            
            if predictions is not None:
                # 评估
                mae = mean_absolute_error(test_data, predictions)
                top15_acc = self.evaluate_top_k(predictions, test_data, k=15)
                
                results[model_name] = {
                    'predictions': predictions,
                    'mae': mae,
                    'top15': top15_acc
                }
                
                print(f"   MAE: {mae:.2f}")
                print(f"   Top 15估算: {top15_acc:.1f}%")
        
        # 汇总结果
        print("\n" + "=" * 80)
        print("测试结果汇总")
        print("=" * 80)
        
        if results:
            print(f"\n{'模型':<15} {'MAE':<10} {'Top15估算':<15}")
            print("-" * 60)
            
            for model_name, result in sorted(results.items(), key=lambda x: x[1]['top15'], reverse=True):
                print(f"{model_name:<15} {result['mae']:<10.2f} {result['top15']:<15.1f}%")
            
            # 最佳模型
            best_model = max(results.items(), key=lambda x: x[1]['top15'])
            print(f"\n[最佳模型] {best_model[0]}")
            print(f"   Top 15估算: {best_model[1]['top15']:.1f}%")
        else:
            print("\n[WARN] 没有模型成功运行")
        
        return results


def main():
    """主函数"""
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    # 运行测试
    tester = AdvancedModelsTester()
    results = tester.run_all_tests(df)
    
    # 额外分析
    print("\n" + "=" * 80)
    print("深度分析")
    print("=" * 80)
    
    numbers = df['number'].values
    
    # 1. 数据分布
    print("\n[1] 数据分布分析")
    zones = {
        '极小(1-10)': sum(1 for n in numbers if 1 <= n <= 10),
        '小(11-20)': sum(1 for n in numbers if 11 <= n <= 20),
        '中(21-30)': sum(1 for n in numbers if 21 <= n <= 30),
        '大(31-40)': sum(1 for n in numbers if 31 <= n <= 40),
        '极大(41-49)': sum(1 for n in numbers if 41 <= n <= 49)
    }
    
    total = len(numbers)
    for zone, count in zones.items():
        print(f"   {zone}: {count}期 ({count/total*100:.1f}%)")
    
    # 2. 最近趋势
    print("\n[2] 最近10期趋势")
    recent_10 = numbers[-10:]
    extreme_count = sum(1 for n in recent_10 if n <= 10 or n >= 40)
    print(f"   极端值: {extreme_count}/10 ({extreme_count*10}%)")
    print(f"   数字: {recent_10.tolist()}")
    
    # 3. 自相关性
    print("\n[3] 自相关分析")
    from scipy.stats import pearsonr
    lag_1_corr = pearsonr(numbers[:-1], numbers[1:])[0]
    lag_5_corr = pearsonr(numbers[:-5], numbers[5:])[0]
    print(f"   滞后1期相关性: {lag_1_corr:.3f}")
    print(f"   滞后5期相关性: {lag_5_corr:.3f}")
    
    if abs(lag_1_corr) < 0.1:
        print("   结论: 弱相关性，接近随机游走")
    
    # 4. 周期性
    print("\n[4] 周期性检测")
    freq_7 = Counter([i % 7 for i in range(len(numbers))])
    print(f"   按周期(7天)分组: {dict(freq_7)}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
