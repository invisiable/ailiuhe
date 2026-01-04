"""
Top15统计分布预测器
基于概率论和统计学的科学预测模型

使用的统计方法：
1. 泊松分布 - 预测数字出现频率
2. 正态分布 - 数字分布拟合和异常检测
3. 卡方检验 - 验证分布均匀性
4. t检验 - 比较不同周期的显著性差异
5. 多项分布 - 多类别概率建模
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from scipy.stats import poisson, norm, chi2, t, binom
import warnings
warnings.filterwarnings('ignore')


class Top15StatisticalPredictor:
    """基于统计分布的Top15预测器"""
    
    def __init__(self):
        self.alpha = 0.05  # 显著性水平
        
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
    
    def predict(self, numbers):
        """主预测方法"""
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        if len(numbers_list) < 30:
            return list(range(1, 16))
        
        # 获取分析结果
        analysis = self._comprehensive_analysis(numbers_list)
        
        # 综合打分
        scores = self._calculate_scores(numbers_list, analysis)
        
        # 返回Top15
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:15]]
    
    def predict_top20(self, numbers):
        """预测Top20"""
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        if len(numbers_list) < 30:
            return list(range(1, 21))
        
        analysis = self._comprehensive_analysis(numbers_list)
        scores = self._calculate_scores(numbers_list, analysis)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:20]]
    
    def _comprehensive_analysis(self, numbers):
        """综合统计分析"""
        recent_100 = numbers[-100:] if len(numbers) >= 100 else numbers
        recent_50 = numbers[-50:]
        recent_30 = numbers[-30:]
        recent_20 = numbers[-20:]
        recent_10 = numbers[-10:]
        
        analysis = {}
        
        # 1. 泊松分布分析 - 预测每个数字的出现频率
        analysis['poisson'] = self._poisson_analysis(recent_100)
        
        # 2. 正态分布分析 - 识别异常值和趋势
        analysis['normal'] = self._normal_distribution_analysis(recent_100)
        
        # 3. 卡方检验 - 验证分布均匀性
        analysis['chi_square'] = self._chi_square_test(recent_100)
        
        # 4. t检验 - 比较不同周期的差异
        analysis['t_test'] = self._t_test_analysis(recent_50, recent_20)
        
        # 5. 二项分布 - 奇偶预测
        analysis['binomial'] = self._binomial_analysis(recent_30)
        
        # 6. 趋势分析
        analysis['trend'] = self._trend_analysis(recent_20)
        
        # 7. 间隔分析
        analysis['gap'] = self._gap_analysis(recent_100)
        
        return analysis
    
    def _poisson_analysis(self, numbers):
        """泊松分布分析 - 预测数字出现频率"""
        # 计算每个数字的历史出现次数
        freq = Counter(numbers)
        total_periods = len(numbers)
        
        # 计算期望频率 (lambda)
        expected_freq = total_periods / 49  # 假设49个数字均匀分布
        
        # 为每个数字计算泊松概率
        poisson_scores = {}
        for num in range(1, 50):
            observed = freq.get(num, 0)
            
            # 如果观察值低于期望，说明该数字"欠债"，应该有更高概率出现
            if observed < expected_freq:
                # 计算出现的概率（基于泊松分布）
                prob = poisson.pmf(observed, expected_freq)
                # 欠债越多，分数越高
                debt_ratio = (expected_freq - observed) / expected_freq
                poisson_scores[num] = debt_ratio * 100
            else:
                # 已经超额出现，降低分数
                excess_ratio = (observed - expected_freq) / expected_freq
                poisson_scores[num] = max(0, 50 - excess_ratio * 50)
        
        return poisson_scores
    
    def _normal_distribution_analysis(self, numbers):
        """正态分布分析 - 识别异常值和趋势"""
        # 计算均值和标准差
        mean = np.mean(numbers)
        std = np.std(numbers)
        
        # 为每个数字计算概率密度
        normal_scores = {}
        for num in range(1, 50):
            # 计算z-score
            z_score = abs(num - mean) / (std + 1e-10)
            
            # 正态分布概率密度
            pdf = norm.pdf(num, mean, std)
            
            # 如果数字接近均值，给更高分数
            # 如果是异常值（z-score > 2），也给较高分数（认为会回归）
            if z_score > 2:
                normal_scores[num] = 70 + (z_score - 2) * 10  # 异常值回归机会
            else:
                normal_scores[num] = pdf * 1000  # 正常范围
        
        return {
            'scores': normal_scores,
            'mean': mean,
            'std': std,
            'trend': 'high' if mean > 30 else 'low' if mean < 20 else 'mid'
        }
    
    def _chi_square_test(self, numbers):
        """卡方检验 - 验证分布的均匀性"""
        # 统计每个数字的出现次数
        freq = Counter(numbers)
        observed = [freq.get(i, 0) for i in range(1, 50)]
        
        # 期望频率（均匀分布）
        expected_freq = len(numbers) / 49
        expected = [expected_freq] * 49
        
        # 卡方检验
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        # 根据偏差计算分数
        chi_scores = {}
        for num in range(1, 50):
            obs = freq.get(num, 0)
            # 偏差越大，说明该数字不符合均匀分布
            deviation = abs(obs - expected_freq)
            
            # 如果低于期望，给高分（认为会回补）
            if obs < expected_freq:
                chi_scores[num] = 80 + deviation * 5
            else:
                chi_scores[num] = max(0, 50 - deviation * 3)
        
        return {
            'scores': chi_scores,
            'chi2_stat': chi2_stat,
            'p_value': p_value,
            'is_uniform': p_value > self.alpha
        }
    
    def _t_test_analysis(self, period1, period2):
        """t检验 - 比较两个时期的显著性差异"""
        # 对每个数字，比较在两个时期的出现频率
        freq1 = Counter(period1)
        freq2 = Counter(period2)
        
        t_scores = {}
        for num in range(1, 50):
            f1 = freq1.get(num, 0)
            f2 = freq2.get(num, 0)
            
            # 如果在最近时期出现较少，给更高分
            diff = f1 - f2
            
            if diff > 0:
                # 早期出现多，最近少 -> 可能会回补
                t_scores[num] = 60 + diff * 10
            elif diff < 0:
                # 最近出现多 -> 降低优先级
                t_scores[num] = max(0, 40 + diff * 5)
            else:
                t_scores[num] = 50
        
        return t_scores
    
    def _binomial_analysis(self, numbers):
        """二项分布分析 - 奇偶模式"""
        # 统计奇偶出现次数
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        even_count = len(numbers) - odd_count
        
        # 预测下一个更可能是奇数还是偶数
        odd_prob = odd_count / len(numbers)
        
        # 二项分布：如果奇数太多，下个可能是偶数
        binom_scores = {}
        for num in range(1, 50):
            if num % 2 == 1:  # 奇数
                # 如果奇数已经很多，降低分数
                if odd_prob > 0.6:
                    binom_scores[num] = 30
                elif odd_prob < 0.4:
                    binom_scores[num] = 70
                else:
                    binom_scores[num] = 50
            else:  # 偶数
                if odd_prob > 0.6:
                    binom_scores[num] = 70
                elif odd_prob < 0.4:
                    binom_scores[num] = 30
                else:
                    binom_scores[num] = 50
        
        return {
            'scores': binom_scores,
            'odd_prob': odd_prob,
            'even_prob': 1 - odd_prob
        }
    
    def _trend_analysis(self, numbers):
        """趋势分析 - 使用线性回归"""
        if len(numbers) < 3:
            return {'slope': 0, 'trend': 'stable'}
        
        # 线性回归
        x = np.arange(len(numbers))
        y = np.array(numbers)
        
        # 计算斜率
        slope, intercept = np.polyfit(x, y, 1)
        
        # 预测下一个值
        next_pred = slope * len(numbers) + intercept
        
        return {
            'slope': slope,
            'intercept': intercept,
            'predicted_next': next_pred,
            'trend': 'increasing' if slope > 1 else 'decreasing' if slope < -1 else 'stable'
        }
    
    def _gap_analysis(self, numbers):
        """间隔分析 - 每个数字距离上次出现的期数"""
        last_seen = {}
        for i, num in enumerate(numbers):
            last_seen[num] = i
        
        current_idx = len(numbers)
        gap_scores = {}
        
        for num in range(1, 50):
            if num in last_seen:
                gap = current_idx - last_seen[num]
                
                # 最佳间隔：5-20期
                if 5 <= gap <= 20:
                    gap_scores[num] = 100
                elif 20 < gap <= 40:
                    gap_scores[num] = 80
                elif gap > 40:
                    gap_scores[num] = 90  # 长期未出现，回补机会
                else:
                    gap_scores[num] = 40  # 刚刚出现过
            else:
                gap_scores[num] = 95  # 从未出现，很大回补机会
        
        return gap_scores
    
    def _calculate_scores(self, numbers, analysis):
        """综合计算分数"""
        scores = defaultdict(float)
        
        # 优化权重配置
        weights = {
            'poisson': 0.30,      # 泊松分布 30% (提高)
            'normal': 0.10,       # 正态分布 10%
            'chi_square': 0.20,   # 卡方检验 20% (提高)
            't_test': 0.05,       # t检验 5%
            'binomial': 0.05,     # 二项分布 5%
            'gap': 0.25,          # 间隔分析 25% (提高)
            'frequency': 0.05     # 基础频率 5%
        }
        
        # 1. 泊松分布得分
        for num, score in analysis['poisson'].items():
            scores[num] += score * weights['poisson']
        
        # 2. 正态分布得分
        for num, score in analysis['normal']['scores'].items():
            scores[num] += score * weights['normal']
        
        # 3. 卡方检验得分
        for num, score in analysis['chi_square']['scores'].items():
            scores[num] += score * weights['chi_square']
        
        # 4. t检验得分
        for num, score in analysis['t_test'].items():
            scores[num] += score * weights['t_test']
        
        # 5. 二项分布得分
        for num, score in analysis['binomial']['scores'].items():
            scores[num] += score * weights['binomial']
        
        # 6. 间隔分析得分
        for num, score in analysis['gap'].items():
            scores[num] += score * weights['gap']
        
        # 7. 基础频率 - 增加最近频率权重
        recent_50 = numbers[-50:]
        recent_30 = numbers[-30:]
        recent_20 = numbers[-20:]
        
        freq_50 = Counter(recent_50)
        freq_30 = Counter(recent_30)
        freq_20 = Counter(recent_20)
        
        for num in range(1, 50):
            f50 = freq_50.get(num, 0)
            f30 = freq_30.get(num, 0)
            f20 = freq_20.get(num, 0)
            
            # 综合频率评分
            if f50 >= 2 and f30 >= 1 and f20 == 0:
                # 早期出现，最近未出现 - 高分
                scores[num] += 90 * weights['frequency']
            elif f20 >= 2:
                # 最近频繁出现 - 低分
                scores[num] += 20 * weights['frequency']
            elif f30 == 0 and f50 <= 2:
                # 中期未出现 - 中高分
                scores[num] += 70 * weights['frequency']
            else:
                scores[num] += 50 * weights['frequency']
        
        # 惩罚最近5期出现的数字
        recent_5 = set(numbers[-5:])
        for num in recent_5:
            scores[num] *= 0.2
        
        # 额外加分：周期回补
        for period in [3, 5, 7, 10]:
            if len(numbers) > period:
                scores[numbers[-period]] += 8
        
        return scores
    
    def get_analysis_report(self, numbers):
        """生成分析报告"""
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        if len(numbers_list) < 30:
            return "数据不足，至少需要30期"
        
        analysis = self._comprehensive_analysis(numbers_list)
        
        report = []
        report.append("=" * 70)
        report.append("统计分布分析报告")
        report.append("=" * 70)
        
        # 1. 正态分布分析
        report.append("\n【正态分布分析】")
        report.append(f"  均值: {analysis['normal']['mean']:.2f}")
        report.append(f"  标准差: {analysis['normal']['std']:.2f}")
        report.append(f"  趋势: {analysis['normal']['trend']}")
        
        # 2. 卡方检验
        report.append("\n【卡方检验 - 均匀性检验】")
        report.append(f"  卡方统计量: {analysis['chi_square']['chi2_stat']:.2f}")
        report.append(f"  p值: {analysis['chi_square']['p_value']:.4f}")
        report.append(f"  结论: {'分布均匀' if analysis['chi_square']['is_uniform'] else '分布不均匀'}")
        
        # 3. 二项分布
        report.append("\n【二项分布 - 奇偶分析】")
        report.append(f"  奇数概率: {analysis['binomial']['odd_prob']:.2%}")
        report.append(f"  偶数概率: {analysis['binomial']['even_prob']:.2%}")
        
        # 4. 趋势分析
        report.append("\n【趋势分析】")
        report.append(f"  斜率: {analysis['trend']['slope']:.2f}")
        report.append(f"  趋势: {analysis['trend']['trend']}")
        report.append(f"  预测下一值: {analysis['trend']['predicted_next']:.1f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def validate():
    """验证预测器性能"""
    print("=" * 80)
    print("Top15统计分布预测器 - 验证测试")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    predictor = Top15StatisticalPredictor()
    
    # 100期回测
    test_periods = 100
    hits_15 = 0
    hits_20 = 0
    total = 0
    
    print("\n开始100期回测验证...")
    print(f"{'期数':<8} {'实际':<6} {'Top15':<8} {'Top20':<8}")
    print("-" * 80)
    
    for i in range(len(numbers) - test_periods, len(numbers)):
        actual = numbers[i]
        history = numbers[:i]
        
        if len(history) < 30:
            continue
        
        # 预测
        top15 = predictor.predict(history)
        top20 = predictor.predict_top20(history)
        
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
        
        if i % 10 == 0 or hit_15:  # 只显示部分结果
            print(f"第{i+1:<5}期 {actual:<6} {status_15:<8} {status_20:<8}")
    
    # 统计结果
    print("-" * 80)
    print(f"\n验证完成!")
    print(f"测试期数: {total} 期")
    print(f"Top15命中: {hits_15} 期")
    print(f"Top15成功率: {hits_15/total*100:.1f}%")
    print(f"Top20命中: {hits_20} 期")
    print(f"Top20成功率: {hits_20/total*100:.1f}%")
    
    # 显示分析报告
    print("\n" + predictor.get_analysis_report(numbers))
    
    # 预测下一期
    print("\n【下一期预测】")
    top20 = predictor.predict_top20(numbers)
    print(f"Top20推荐: {top20}")
    

if __name__ == '__main__':
    validate()
