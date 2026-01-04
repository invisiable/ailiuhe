"""
集成Top15预测器 - 多模型融合
结合多种预测模型的优势，使用集成学习方法提升成功率

集成方法：
1. 加权投票法（Weighted Voting）
2. 排名融合法（Rank Fusion）
3. 概率叠加法（Probability Stacking）
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# 导入各个预测器
from top15_zodiac_enhanced_v2 import Top15ZodiacEnhancedV2
from top15_statistical_predictor import Top15StatisticalPredictor
from zodiac_balanced_smart import ZodiacBalancedSmart


class EnsembleTop15Predictor:
    """集成Top15预测器 - 多模型融合"""
    
    def __init__(self):
        # 初始化各个子模型
        self.zodiac_hybrid = Top15ZodiacEnhancedV2()  # 46% Top20
        self.statistical = Top15StatisticalPredictor()  # 44% Top20
        self.zodiac_smart = ZodiacBalancedSmart()  # 52% Top6
        
        # 生肖映射
        self.zodiac_numbers = {
            '鼠': [4, 16, 28, 40],
            '牛': [5, 17, 29, 41],
            '虎': [6, 18, 30, 42],
            '兔': [7, 19, 31, 43],
            '龙': [8, 20, 32, 44],
            '蛇': [9, 21, 33, 45],
            '马': [10, 22, 34, 46],
            '羊': [11, 23, 35, 47],
            '猴': [12, 24, 36, 48],
            '鸡': [1, 13, 25, 37, 49],
            '狗': [2, 14, 26, 38],
            '猪': [3, 15, 27, 39]
        }
        
        # 模型权重（基于历史表现）
        self.model_weights = {
            'zodiac_hybrid': 0.35,   # 46%成功率
            'statistical': 0.30,      # 44%成功率
            'zodiac_smart': 0.25,     # 52%成功率（但只有6个）
            'frequency': 0.10         # 基础频率
        }
    
    def predict(self, numbers):
        """主预测方法 - 返回Top15"""
        top20 = self.predict_top20(numbers)
        return top20[:15]
    
    def predict_top20(self, numbers):
        """预测Top20 - 集成多个模型"""
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        if len(numbers_list) < 30:
            return list(range(1, 21))
        
        # 方法1：加权投票法
        scores_voting = self._weighted_voting(numbers_list)
        
        # 方法2：排名融合法
        scores_ranking = self._rank_fusion(numbers_list)
        
        # 方法3：概率叠加法
        scores_stacking = self._probability_stacking(numbers_list)
        
        # 综合三种方法 - 调整权重，增加投票法和排名融合的比重
        final_scores = defaultdict(float)
        for num in range(1, 50):
            final_scores[num] = (
                scores_voting.get(num, 0) * 0.5 +      # 提高到50%
                scores_ranking.get(num, 0) * 0.35 +    # 保持35%
                scores_stacking.get(num, 0) * 0.15     # 降低到15%
            )
        
        # 识别高共识数字（多个模型都推荐）- 给予额外加分
        consensus_bonus = self._get_consensus_bonus(numbers_list)
        for num, bonus in consensus_bonus.items():
            final_scores[num] += bonus
        
        # 惩罚最近5期 - 加强惩罚
        recent_5 = set(numbers_list[-5:])
        for num in recent_5:
            final_scores[num] *= 0.1
        
        # 奖励最近未出现但历史频繁的数字
        recent_30 = numbers_list[-30:]
        recent_50 = numbers_list[-50:]
        freq_50 = Counter(recent_50)
        freq_30 = Counter(recent_30)
        
        for num in range(1, 50):
            # 如果在50期内出现>=3次，但最近30期未出现
            if freq_50.get(num, 0) >= 3 and freq_30.get(num, 0) == 0:
                final_scores[num] += 50
        
        # 排序返回Top20
        sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:20]]
    
    def _get_consensus_bonus(self, numbers):
        """计算共识加分 - 多个模型都推荐的数字"""
        predictions = defaultdict(int)
        
        # 收集各模型的预测
        try:
            zodiac_top20 = self.zodiac_hybrid.predict_top20(numbers)
            for num in zodiac_top20[:15]:  # 只看Top15
                predictions[num] += 1
        except:
            pass
        
        try:
            stat_top20 = self.statistical.predict_top20(numbers)
            for num in stat_top20[:15]:
                predictions[num] += 1
        except:
            pass
        
        try:
            zodiac_top6 = self.zodiac_smart.predict_top6(recent_periods=100)
            for zodiac in zodiac_top6[:3]:  # 只看Top3生肖
                nums = self.zodiac_numbers.get(zodiac, [])
                for num in nums:
                    predictions[num] += 0.5  # 生肖权重降低
        except:
            pass
        
        # 计算加分
        bonus = {}
        for num, count in predictions.items():
            if count >= 3:  # 3个模型都推荐
                bonus[num] = 100
            elif count >= 2:  # 2个模型推荐
                bonus[num] = 50
            elif count >= 1.5:  # 统计+生肖
                bonus[num] = 30
        
        return bonus
    
    def _weighted_voting(self, numbers):
        """方法1：加权投票法"""
        scores = defaultdict(float)
        
        # 1. 生肖混合预测器
        try:
            zodiac_top20 = self.zodiac_hybrid.predict_top20(numbers)
            for i, num in enumerate(zodiac_top20[:20], 1):
                # 排名越前，分数越高
                score = (21 - i) * 5
                scores[num] += score * self.model_weights['zodiac_hybrid']
        except:
            pass
        
        # 2. 统计分布预测器
        try:
            stat_top20 = self.statistical.predict_top20(numbers)
            for i, num in enumerate(stat_top20[:20], 1):
                score = (21 - i) * 5
                scores[num] += score * self.model_weights['statistical']
        except:
            pass
        
        # 3. 生肖智能预测器
        try:
            zodiac_top6 = self.zodiac_smart.predict_top6(recent_periods=100)
            # 将生肖转换为数字
            for i, zodiac in enumerate(zodiac_top6, 1):
                nums = self.zodiac_numbers.get(zodiac, [])
                score = (7 - i) * 10
                for num in nums:
                    scores[num] += score * self.model_weights['zodiac_smart']
        except:
            pass
        
        # 4. 基础频率
        recent_50 = numbers[-50:]
        freq = Counter(recent_50)
        for num in range(1, 50):
            f = freq.get(num, 0)
            # 适中频率加分
            if 1 <= f <= 3:
                scores[num] += 60 * self.model_weights['frequency']
            elif f == 0:
                scores[num] += 80 * self.model_weights['frequency']
        
        return scores
    
    def _rank_fusion(self, numbers):
        """方法2：排名融合法（类似Borda Count）"""
        ranks = defaultdict(list)
        
        # 收集各个模型的排名
        models_predictions = []
        
        # 模型1：生肖混合
        try:
            models_predictions.append({
                'predictions': self.zodiac_hybrid.predict_top20(numbers),
                'weight': self.model_weights['zodiac_hybrid']
            })
        except:
            pass
        
        # 模型2：统计分布
        try:
            models_predictions.append({
                'predictions': self.statistical.predict_top20(numbers),
                'weight': self.model_weights['statistical']
            })
        except:
            pass
        
        # 模型3：生肖智能
        try:
            zodiac_top6 = self.zodiac_smart.predict_top6(recent_periods=100)
            zodiac_nums = []
            for zodiac in zodiac_top6:
                zodiac_nums.extend(self.zodiac_numbers.get(zodiac, []))
            models_predictions.append({
                'predictions': zodiac_nums[:20],
                'weight': self.model_weights['zodiac_smart']
            })
        except:
            pass
        
        # 计算加权排名分数
        scores = defaultdict(float)
        for model in models_predictions:
            for rank, num in enumerate(model['predictions'][:20], 1):
                # 排名越前，分数越高（20, 19, 18, ...）
                rank_score = (21 - rank)
                scores[num] += rank_score * model['weight']
        
        return scores
    
    def _probability_stacking(self, numbers):
        """方法3：概率叠加法"""
        scores = defaultdict(float)
        
        recent_100 = numbers[-100:] if len(numbers) >= 100 else numbers
        recent_50 = numbers[-50:]
        recent_30 = numbers[-30:]
        recent_20 = numbers[-20:]
        
        # 1. 基于历史频率的概率
        freq_100 = Counter(recent_100)
        freq_50 = Counter(recent_50)
        total_100 = len(recent_100)
        total_50 = len(recent_50)
        
        for num in range(1, 50):
            # 长期概率
            prob_100 = freq_100.get(num, 0) / total_100
            # 短期概率
            prob_50 = freq_50.get(num, 0) / total_50
            
            # 加权平均
            avg_prob = prob_100 * 0.4 + prob_50 * 0.6
            scores[num] += avg_prob * 1000
        
        # 2. 间隔概率
        last_seen = {}
        for i, n in enumerate(recent_100):
            last_seen[n] = len(recent_100) - i
        
        for num in range(1, 50):
            gap = last_seen.get(num, 100)
            
            # 最优间隔区间
            if 5 <= gap <= 20:
                gap_score = 100
            elif 20 < gap <= 40:
                gap_score = 80
            elif gap > 40:
                gap_score = 90
            else:
                gap_score = 30
            
            scores[num] += gap_score
        
        # 3. 趋势概率
        recent_trend = numbers[-10:]
        avg_recent = np.mean(recent_trend)
        
        for num in range(1, 50):
            # 如果数字接近最近平均值，加分
            distance = abs(num - avg_recent)
            if distance <= 10:
                scores[num] += 50
            elif distance <= 20:
                scores[num] += 30
        
        return scores
    
    def get_ensemble_analysis(self, numbers):
        """获取集成分析报告"""
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        if len(numbers_list) < 30:
            return "数据不足"
        
        # 获取各个模型的预测
        models_results = {}
        
        try:
            models_results['zodiac_hybrid'] = self.zodiac_hybrid.predict_top20(numbers_list)
        except:
            models_results['zodiac_hybrid'] = []
        
        try:
            models_results['statistical'] = self.statistical.predict_top20(numbers_list)
        except:
            models_results['statistical'] = []
        
        try:
            zodiac_top6 = self.zodiac_smart.predict_top6(recent_periods=100)
            zodiac_nums = []
            for zodiac in zodiac_top6:
                zodiac_nums.extend(self.zodiac_numbers.get(zodiac, []))
            models_results['zodiac_smart'] = zodiac_nums[:20]
        except:
            models_results['zodiac_smart'] = []
        
        # 找出共同推荐的数字
        all_predictions = []
        for preds in models_results.values():
            all_predictions.extend(preds[:20])
        
        common_nums = [num for num, count in Counter(all_predictions).items() if count >= 2]
        high_consensus = [num for num, count in Counter(all_predictions).items() if count >= 3]
        
        return {
            'models': models_results,
            'common': common_nums,  # 至少2个模型推荐
            'high_consensus': high_consensus,  # 3个模型都推荐
            'total_unique': len(set(all_predictions))
        }


def validate():
    """验证集成预测器性能"""
    print("=" * 80)
    print("集成Top15预测器 - 多模型融合验证")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    predictor = EnsembleTop15Predictor()
    
    # 100期回测
    test_periods = 100
    hits_15 = 0
    hits_20 = 0
    total = 0
    
    # 记录各模型的贡献
    hits_by_method = {
        'voting': 0,
        'ranking': 0,
        'stacking': 0
    }
    
    print("\n开始100期回测验证...")
    print(f"{'期数':<8} {'实际':<6} {'Top15':<8} {'Top20':<8} {'共识':<8}")
    print("-" * 80)
    
    for i in range(len(numbers) - test_periods, len(numbers)):
        actual = numbers[i]
        history = numbers[:i]
        
        if len(history) < 30:
            continue
        
        # 预测
        top15 = predictor.predict(history)
        top20 = predictor.predict_top20(history)
        
        # 获取分析
        analysis = predictor.get_ensemble_analysis(history)
        is_consensus = actual in analysis['high_consensus']
        
        # 检查命中
        hit_15 = actual in top15
        hit_20 = actual in top20
        
        if hit_15:
            hits_15 += 1
        if hit_20:
            hits_20 += 1
        
        total += 1
        
        status_15 = "Y" if hit_15 else "N"
        status_20 = "Y" if hit_20 else "N"
        consensus_mark = "***" if is_consensus else ""
        
        if i % 10 == 0 or hit_15:
            print(f"第{i+1:<5}期 {actual:<6} {status_15:<8} {status_20:<8} {consensus_mark:<8}")
    
    # 统计结果
    print("-" * 80)
    print(f"\n验证完成!")
    print(f"测试期数: {total} 期")
    print(f"Top15命中: {hits_15} 期")
    print(f"Top15成功率: {hits_15/total*100:.1f}%")
    print(f"Top20命中: {hits_20} 期")
    print(f"Top20成功率: {hits_20/total*100:.1f}%")
    
    # 显示模型组合分析
    print("\n【模型组合优势】")
    print(f"生肖混合预测器: 46% (Top20)")
    print(f"统计分布预测器: 44% (Top20)")
    print(f"生肖智能预测器: 52% (Top6)")
    print(f"→ 集成后成功率: {hits_20/total*100:.1f}% (Top20)")
    
    # 预测下一期
    print("\n【下一期预测】")
    top20 = predictor.predict_top20(numbers)
    analysis = predictor.get_ensemble_analysis(numbers)
    
    print(f"Top20推荐: {top20}")
    print(f"\n高共识数字（3个模型都推荐）: {analysis['high_consensus']}")
    print(f"中共识数字（2个模型推荐）: {analysis['common']}")


if __name__ == '__main__':
    validate()
