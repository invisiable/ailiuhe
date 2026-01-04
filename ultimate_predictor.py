"""
终极预测器 - 基于高级模型测试结果的最优组合
结合：频率+趋势(50%) + 集成投票(30%) + 极端值感知(40%)
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_AVAILABLE = True
except:
    ADVANCED_AVAILABLE = False


class UltimatePredictor:
    """终极预测器 - 多策略融合"""
    
    def __init__(self):
        self.element_numbers = {
            '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
            '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
            '水': [13, 14, 21, 22, 29, 30, 43, 44],
            '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
            '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
        }
    
    def analyze_data_pattern(self, numbers):
        """深度分析数据模式"""
        recent_30 = numbers[-30:]
        recent_10 = numbers[-10:]
        
        # 区域分析
        zones_10 = {
            'extreme_small': sum(1 for n in recent_10 if 1 <= n <= 10),
            'small': sum(1 for n in recent_10 if 11 <= n <= 20),
            'mid': sum(1 for n in recent_10 if 21 <= n <= 30),
            'large': sum(1 for n in recent_10 if 31 <= n <= 40),
            'extreme_large': sum(1 for n in recent_10 if 41 <= n <= 49)
        }
        
        # 判断趋势
        extreme_ratio = (zones_10['extreme_small'] + zones_10['extreme_large']) / 10
        mid_ratio = zones_10['mid'] / 10
        
        if extreme_ratio >= 0.5:
            trend = 'extreme'
        elif mid_ratio >= 0.5:
            trend = 'middle'
        else:
            trend = 'balanced'
        
        return {
            'recent_30': recent_30,
            'recent_10': recent_10,
            'zones': zones_10,
            'trend': trend,
            'extreme_ratio': extreme_ratio
        }
    
    def strategy_frequency_trend(self, pattern, top_k=15):
        """策略1: 频率+趋势（最佳单一策略）"""
        recent_30 = pattern['recent_30']
        freq_counter = Counter(recent_30)
        
        # 根据趋势筛选候选范围
        if pattern['trend'] == 'extreme':
            # 极端值趋势 - 重点关注1-10和40-49
            candidates_weight = {}
            for n in range(1, 11):
                candidates_weight[n] = freq_counter.get(n, 0) * 2.0  # 加倍权重
            for n in range(11, 40):
                candidates_weight[n] = freq_counter.get(n, 0) * 0.5  # 降低权重
            for n in range(40, 50):
                candidates_weight[n] = freq_counter.get(n, 0) * 2.0  # 加倍权重
        elif pattern['trend'] == 'middle':
            # 中间值趋势
            for n in range(1, 50):
                if 21 <= n <= 30:
                    candidates_weight[n] = freq_counter.get(n, 0) * 2.0
                else:
                    candidates_weight[n] = freq_counter.get(n, 0) * 0.8
        else:
            # 平衡分布
            candidates_weight = {n: freq_counter.get(n, 0) for n in range(1, 50)}
        
        # 排序返回Top K
        sorted_candidates = sorted(candidates_weight.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)
        
        return [num for num, _ in sorted_candidates[:top_k]]
    
    def strategy_ensemble_ml(self, numbers, top_k=15):
        """策略2: 机器学习集成"""
        if not ADVANCED_AVAILABLE:
            return []
        
        try:
            seq_length = 10
            X, y = [], []
            
            for i in range(len(numbers) - seq_length):
                X.append(numbers[i:i+seq_length])
                y.append(numbers[i+seq_length])
            
            X = np.array(X)
            y = np.array(y)
            
            # 训练多个模型
            models = [
                RandomForestRegressor(n_estimators=50, random_state=42),
                GradientBoostingRegressor(n_estimators=50, random_state=42),
                xgb.XGBRegressor(n_estimators=50, random_state=42),
                lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            ]
            
            predictions = []
            for model in models:
                model.fit(X, y)
                # 预测多个候选
                last_seq = numbers[-seq_length:].reshape(1, -1)
                pred = model.predict(last_seq)[0]
                predictions.append(int(np.clip(pred, 1, 49)))
            
            # 扩展到Top K（添加周围数字）
            candidates = set(predictions)
            for pred in predictions:
                for offset in range(1, top_k):
                    if len(candidates) >= top_k:
                        break
                    if pred - offset >= 1:
                        candidates.add(pred - offset)
                    if pred + offset <= 49:
                        candidates.add(pred + offset)
            
            return list(candidates)[:top_k]
            
        except:
            return []
    
    def strategy_extreme_aware(self, pattern, top_k=15):
        """策略3: 极端值感知"""
        recent_30 = pattern['recent_30']
        recent_3 = set(recent_30[-3:])
        
        freq_counter = Counter(recent_30)
        
        # 极小值候选
        small_candidates = [n for n in range(1, 11) if n not in recent_3]
        small_freq = {n: freq_counter.get(n, 0) for n in small_candidates}
        
        # 极大值候选
        large_candidates = [n for n in range(40, 50) if n not in recent_3]
        large_freq = {n: freq_counter.get(n, 0) for n in large_candidates}
        
        # 中间值候选
        mid_candidates = [n for n in range(11, 40) if n not in recent_3]
        mid_freq = {n: freq_counter.get(n, 0) for n in mid_candidates}
        
        # 根据趋势分配
        if pattern['extreme_ratio'] >= 0.5:
            small_k = top_k // 3 + 1
            large_k = top_k // 3
            mid_k = top_k - small_k - large_k
        else:
            small_k = top_k // 5
            large_k = top_k // 5
            mid_k = top_k - small_k - large_k
        
        # 选择
        selected = []
        selected.extend(sorted(small_freq.keys(), key=lambda x: small_freq[x], reverse=True)[:small_k])
        selected.extend(sorted(large_freq.keys(), key=lambda x: large_freq[x], reverse=True)[:large_k])
        selected.extend(sorted(mid_freq.keys(), key=lambda x: mid_freq[x], reverse=True)[:mid_k])
        
        return selected[:top_k]
    
    def strategy_zone_balance(self, pattern, top_k=15):
        """策略4: 区域平衡"""
        recent_30 = pattern['recent_30']
        freq_counter = Counter(recent_30)
        
        # 每个区域选择2-3个
        zone_ranges = [
            (1, 10, 3),    # 极小
            (11, 20, 2),   # 小
            (21, 30, 4),   # 中
            (31, 40, 3),   # 大
            (41, 49, 3)    # 极大
        ]
        
        selected = []
        for start, end, count in zone_ranges:
            candidates = {n: freq_counter.get(n, 0) for n in range(start, end+1)}
            zone_selected = sorted(candidates.keys(), key=lambda x: candidates[x], reverse=True)[:count]
            selected.extend(zone_selected)
        
        return selected[:top_k]
    
    def predict_ultimate(self, numbers, top_k=15):
        """终极预测 - 多策略融合"""
        print(f"\n[终极预测器] Top {top_k}")
        print("-" * 70)
        
        # 1. 分析模式
        print("1. 分析数据模式...")
        pattern = self.analyze_data_pattern(numbers)
        print(f"   趋势: {pattern['trend']}")
        print(f"   极端值比例: {pattern['extreme_ratio']*100:.1f}%")
        
        # 2. 运行各策略
        print("2. 运行预测策略...")
        
        strategies_results = {}
        
        # 策略1: 频率+趋势 (权重: 0.35)
        print("   [1/4] 频率+趋势...")
        s1 = self.strategy_frequency_trend(pattern, top_k=top_k)
        strategies_results['freq_trend'] = (s1, 0.35)
        
        # 策略2: 机器学习集成 (权重: 0.25)
        print("   [2/4] 机器学习集成...")
        s2 = self.strategy_ensemble_ml(numbers, top_k=top_k)
        if s2:
            strategies_results['ml_ensemble'] = (s2, 0.25)
        
        # 策略3: 极端值感知 (权重: 0.25)
        print("   [3/4] 极端值感知...")
        s3 = self.strategy_extreme_aware(pattern, top_k=top_k)
        strategies_results['extreme'] = (s3, 0.25)
        
        # 策略4: 区域平衡 (权重: 0.15)
        print("   [4/4] 区域平衡...")
        s4 = self.strategy_zone_balance(pattern, top_k=top_k)
        strategies_results['zone'] = (s4, 0.15)
        
        # 3. 融合结果
        print("3. 融合结果...")
        candidate_scores = {}
        
        for strategy_name, (candidates, weight) in strategies_results.items():
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                candidate_scores[num] = candidate_scores.get(num, 0) + score
        
        # 排序
        final_candidates = sorted(candidate_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:top_k]
        
        result = [num for num, _ in final_candidates]
        
        print(f"   最终Top {top_k}: {result}")
        
        return result


def test_ultimate_predictor():
    """测试终极预测器"""
    print("=" * 80)
    print("终极预测器测试")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"\n数据集: {len(df)}期")
    
    numbers = df['number'].values
    
    # 创建预测器
    predictor = UltimatePredictor()
    
    # 在最近10期上测试
    print("\n" + "=" * 80)
    print("在最近10期上测试")
    print("=" * 80)
    
    results = {
        'top5': 0, 'top10': 0, 'top15': 0, 'top20': 0,
        'details': []
    }
    
    for i in range(len(numbers) - 10, len(numbers)):
        actual = numbers[i]
        history = numbers[:i]
        
        print(f"\n第{i+1}期: 实际 = {actual}")
        
        # 预测Top 20
        predictions = predictor.predict_ultimate(history, top_k=20)
        
        # 检查命中
        if actual in predictions:
            rank = predictions.index(actual) + 1
            
            if rank <= 5:
                level = "[*] Top 5"
                results['top5'] += 1
                results['top10'] += 1
                results['top15'] += 1
                results['top20'] += 1
            elif rank <= 10:
                level = "[v] Top 10"
                results['top10'] += 1
                results['top15'] += 1
                results['top20'] += 1
            elif rank <= 15:
                level = "[o] Top 15"
                results['top15'] += 1
                results['top20'] += 1
            else:
                level = "[+] Top 20"
                results['top20'] += 1
            
            print(f"   [HIT] 命中! 排名: {rank} {level}")
        else:
            print(f"   [MISS] 未命中")
        
        print(f"   预测Top15: {predictions[:15]}")
        
        results['details'].append({
            'period': i + 1,
            'actual': actual,
            'hit': actual in predictions[:15],
            'rank': predictions.index(actual) + 1 if actual in predictions else -1
        })
    
    # 统计
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    
    total = len(results['details'])
    
    print(f"\n命中统计 (最近{total}期):")
    print(f"   Top 5:  {results['top5']}/{total} = {results['top5']/total*100:.1f}%")
    print(f"   Top 10: {results['top10']}/{total} = {results['top10']/total*100:.1f}%")
    print(f"   Top 15: {results['top15']}/{total} = {results['top15']/total*100:.1f}%")
    print(f"   Top 20: {results['top20']}/{total} = {results['top20']/total*100:.1f}%")
    
    # 对比
    for key in ['top5', 'top10', 'top15', 'top20']:
        k = int(key[3:])
        actual_rate = results[key] / total * 100
        random_rate = k / 49 * 100
        improvement = actual_rate / random_rate
        status = "[OK]" if improvement > 1.2 else "[WARN]"
        
        print(f"\n{key.upper()}:")
        print(f"   实际: {actual_rate:.1f}%")
        print(f"   随机: {random_rate:.1f}%")
        print(f"   提升: {improvement:.2f}x {status}")
    
    # 评估
    top15_rate = results['top15'] / total * 100
    top20_rate = results['top20'] / total * 100
    
    print("\n" + "=" * 80)
    print("最终评估")
    print("=" * 80)
    
    if top15_rate >= 60:
        print(f"\n[SUCCESS] Top 15 达到 {top15_rate:.1f}%! 超过60%目标!")
    elif top15_rate >= 50:
        print(f"\n[GOOD] Top 15 达到 {top15_rate:.1f}%，接近目标")
    else:
        print(f"\n[PROGRESS] Top 15 当前 {top15_rate:.1f}%")
    
    if top20_rate >= 60:
        print(f"[SUCCESS] Top 20 达到 {top20_rate:.1f}%! 超过60%目标!")
    elif top20_rate >= 50:
        print(f"[GOOD] Top 20 达到 {top20_rate:.1f}%，接近目标")
    
    return results


if __name__ == '__main__':
    test_ultimate_predictor()
