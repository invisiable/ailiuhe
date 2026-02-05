"""
改进的 Top15 预测器
目标：提高准确率至50%+，降低连续失败次数

改进策略：
1. 多窗口分析（10期、20期、30期、50期）
2. 冷热号平衡策略
3. 区域动态平衡
4. 自适应权重调整
5. 排除最近出现的数字
"""

import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class ImprovedTop15Predictor:
    """改进的 Top15 预测器"""
    
    def __init__(self):
        # 定义五行分类
        self.element_numbers = {
            '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
            '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
            '水': [13, 14, 21, 22, 29, 30, 43, 44],
            '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
            '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
        }
        
        # 定义区域
        self.zones = {
            '极小': range(1, 11),
            '小': range(11, 21),
            '中': range(21, 31),
            '大': range(31, 41),
            '极大': range(41, 50)
        }
    
    def analyze_multi_window(self, numbers):
        """多窗口分析"""
        analysis = {}
        
        # 不同窗口期
        windows = {
            'recent_5': numbers[-5:],
            'recent_10': numbers[-10:],
            'recent_20': numbers[-20:] if len(numbers) >= 20 else numbers,
            'recent_30': numbers[-30:] if len(numbers) >= 30 else numbers,
            'recent_50': numbers[-50:] if len(numbers) >= 50 else numbers,
        }
        
        for name, window in windows.items():
            freq = Counter(window)
            analysis[name] = {
                'data': window,
                'freq': freq,
                'mean': np.mean(window),
                'std': np.std(window),
                'extreme_ratio': sum(1 for n in window if n <= 10 or n >= 40) / len(window)
            }
        
        return analysis
    
    def calculate_cold_hot_numbers(self, numbers):
        """计算冷热号"""
        # 最近30期频率
        recent_30 = numbers[-30:]
        freq = Counter(recent_30)
        
        # 热号：出现3次及以上
        hot_numbers = {num for num, count in freq.items() if count >= 3}
        
        # 温号：出现1-2次
        warm_numbers = {num for num, count in freq.items() if 1 <= count < 3}
        
        # 冷号：未出现
        all_numbers = set(range(1, 50))
        cold_numbers = all_numbers - set(freq.keys())
        
        return {
            'hot': hot_numbers,
            'warm': warm_numbers,
            'cold': cold_numbers
        }
    
    def calculate_gap_scores(self, numbers):
        """计算间隔分数 - 优化版"""
        # 记录每个数字最后出现位置
        last_seen = {}
        for i, n in enumerate(numbers):
            last_seen[n] = i
        
        current_pos = len(numbers)
        gap_scores = {}
        
        for n in range(1, 50):
            if n in last_seen:
                gap = current_pos - last_seen[n]
                # 间隔分数：最佳间隔为3-8期
                if 3 <= gap <= 8:
                    score = 2.0
                elif 9 <= gap <= 15:
                    score = 1.5
                elif gap >= 16:
                    score = 1.0 + (gap - 16) * 0.05  # 缓慢递增
                else:  # gap < 3
                    score = 0.3  # 刚出现过的降权
            else:
                # 从未出现或很久没出现
                score = 1.2
            
            gap_scores[n] = score
        
        return gap_scores
    
    def method_balanced_selection(self, numbers):
        """方法1：平衡选号策略（权重35%）"""
        analysis = self.analyze_multi_window(numbers)
        cold_hot = self.calculate_cold_hot_numbers(numbers)
        recent_5 = set(analysis['recent_5']['data'])
        
        scores = {}
        
        for n in range(1, 50):
            score = 0.0
            
            # 排除最近5期出现的数字（重要）
            if n in recent_5:
                score = 0.1
                scores[n] = score
                continue
            
            # 冷热号平衡
            if n in cold_hot['cold']:
                score += 1.5  # 冷号加分
            elif n in cold_hot['warm']:
                score += 1.0  # 温号正常
            else:  # hot
                score += 0.6  # 热号降权
            
            # 多窗口频率综合
            freq_10 = analysis['recent_10']['freq'].get(n, 0)
            freq_20 = analysis['recent_20']['freq'].get(n, 0)
            freq_30 = analysis['recent_30']['freq'].get(n, 0)
            
            # 适度出现加分，频繁出现降权
            if freq_30 == 1:
                score += 0.8
            elif freq_30 == 2:
                score += 0.5
            elif freq_30 >= 3:
                score += 0.2  # 高频降权
            
            scores[n] = score
        
        # 返回分数最高的20个
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:20]]
    
    def method_zone_balanced(self, numbers):
        """方法2：区域平衡策略（权重30%）"""
        analysis = self.analyze_multi_window(numbers)
        recent_5 = set(analysis['recent_5']['data'])
        
        # 统计最近30期各区域出现次数
        recent_30 = analysis['recent_30']['data']
        zone_counts = {zone: 0 for zone in self.zones.keys()}
        
        for n in recent_30:
            for zone_name, zone_range in self.zones.items():
                if n in zone_range:
                    zone_counts[zone_name] += 1
                    break
        
        # 动态分配各区域名额（总共15个名额）
        total_count = sum(zone_counts.values())
        zone_quotas = {}
        
        for zone, count in zone_counts.items():
            # 反比例分配：出现少的区域多选
            ratio = 1.0 - (count / total_count)
            quota = max(2, int(ratio * 15 / len(self.zones) * 1.5))
            zone_quotas[zone] = quota
        
        # 从每个区域选数字
        result = []
        gap_scores = self.calculate_gap_scores(numbers)
        
        for zone_name, zone_range in self.zones.items():
            quota = zone_quotas[zone_name]
            
            # 该区域的候选数字（排除最近5期）
            candidates = [
                (n, gap_scores.get(n, 0))
                for n in zone_range
                if n not in recent_5
            ]
            
            # 按间隔分数排序
            candidates.sort(key=lambda x: x[1], reverse=True)
            result.extend([n for n, _ in candidates[:quota]])
        
        return result[:20]
    
    def method_adaptive_frequency(self, numbers):
        """方法3：自适应频率策略（权重20%）"""
        analysis = self.analyze_multi_window(numbers)
        recent_5 = set(analysis['recent_5']['data'])
        
        # 计算每个数字在不同窗口的"活跃度"
        scores = {}
        
        for n in range(1, 50):
            if n in recent_5:
                scores[n] = 0.1
                continue
            
            # 多窗口活跃度
            activity = 0.0
            
            # 10期内出现过得分
            if n in analysis['recent_10']['freq']:
                activity += 1.0
            
            # 20期内出现过得分
            if n in analysis['recent_20']['freq']:
                activity += 0.8
            
            # 30期内未出现加分
            if n not in analysis['recent_30']['freq']:
                activity += 1.5  # 久未出现
            
            # 50期内出现次数适中
            freq_50 = analysis['recent_50']['freq'].get(n, 0)
            if 1 <= freq_50 <= 2:
                activity += 1.0
            elif freq_50 == 0:
                activity += 0.8
            
            scores[n] = activity
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:20]]
    
    def method_pattern_recognition(self, numbers):
        """方法4：模式识别策略（权重15%）"""
        recent_5 = set(numbers[-5:])
        recent_30 = numbers[-30:]
        
        scores = {}
        
        for n in range(1, 50):
            if n in recent_5:
                scores[n] = 0.1
                continue
            
            score = 0.0
            
            # 检查是否有周期性
            positions = [i for i, num in enumerate(recent_30) if num == n]
            
            if len(positions) >= 2:
                # 计算间隔
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                
                # 如果间隔稳定，加分
                if len(gaps) > 0:
                    avg_gap = np.mean(gaps)
                    std_gap = np.std(gaps) if len(gaps) > 1 else 0
                    
                    # 间隔标准差小说明有规律
                    if std_gap < 2.0:
                        score += 1.5
                    else:
                        score += 0.5
            
            # 未出现的数字也给一定机会
            if len(positions) == 0:
                score += 1.0
            
            scores[n] = score
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:20]]
    
    def predict(self, numbers):
        """综合预测Top15"""
        # 运行所有方法
        method_results = [
            (self.method_balanced_selection(numbers), 0.35),
            (self.method_zone_balanced(numbers), 0.30),
            (self.method_adaptive_frequency(numbers), 0.20),
            (self.method_pattern_recognition(numbers), 0.15)
        ]
        
        # 综合评分
        final_scores = {}
        
        for candidates, weight in method_results:
            for rank, num in enumerate(candidates):
                # 排名越前分数越高
                score = weight * (1.0 - rank / len(candidates))
                final_scores[num] = final_scores.get(num, 0) + score
        
        # 排序获取Top15
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_results[:15]]
    
    def get_analysis(self, numbers):
        """获取详细分析"""
        top15 = self.predict(numbers)
        analysis = self.analyze_multi_window(numbers)
        cold_hot = self.calculate_cold_hot_numbers(numbers)
        
        # 分析预测结果分布
        zones = {
            '极小值区(1-10)': [n for n in top15 if 1 <= n <= 10],
            '小值区(11-20)': [n for n in top15 if 11 <= n <= 20],
            '中值区(21-30)': [n for n in top15 if 21 <= n <= 30],
            '大值区(31-40)': [n for n in top15 if 31 <= n <= 40],
            '极大值区(41-49)': [n for n in top15 if 41 <= n <= 49]
        }
        
        # 分析冷热分布
        cold_count = sum(1 for n in top15 if n in cold_hot['cold'])
        warm_count = sum(1 for n in top15 if n in cold_hot['warm'])
        hot_count = sum(1 for n in top15 if n in cold_hot['hot'])
        
        return {
            'top15': top15,
            'zones': zones,
            'cold_hot_distribution': {
                '冷号': cold_count,
                '温号': warm_count,
                '热号': hot_count
            },
            'extreme_ratio': analysis['recent_10']['extreme_ratio'] * 100,
            'trend': '极端值趋势' if analysis['recent_10']['extreme_ratio'] > 0.5 else '均衡趋势'
        }


def main():
    """主函数 - 测试改进版预测器"""
    from datetime import datetime
    
    print("=" * 80)
    print("改进版 Top15 预测器")
    print("=" * 80)
    
    # 显示预测时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n预测时间: {current_time}")
    print("读取最新数据...")
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    print(f"数据加载完成")
    print(f"基于历史数据: {len(df)}期")
    print(f"最近10期: {numbers[-10:].tolist()}")
    
    # 创建预测器
    predictor = ImprovedTop15Predictor()
    
    # 获取分析
    analysis = predictor.get_analysis(numbers)
    
    print(f"\n当前趋势分析:")
    print(f"  趋势判断: {analysis['trend']}")
    print(f"  极端值占比: {analysis['extreme_ratio']:.1f}% (最近10期)")
    
    print("\n" + "=" * 80)
    print("下一期 Top 15 预测")
    print("=" * 80)
    
    print(f"\n预测号码 (按优先级排序):")
    top15 = analysis['top15']
    
    # 分5-5-5显示
    print(f"  优先级1 (Top 1-5):   {top15[:5]}")
    print(f"  优先级2 (Top 6-10):  {top15[5:10]}")
    print(f"  优先级3 (Top 11-15): {top15[10:15]}")
    
    # 分组显示
    print(f"\n按区域分布:")
    for zone, nums in analysis['zones'].items():
        if nums:
            print(f"  {zone}: {nums}")
    
    print(f"\n冷热号分布:")
    for category, count in analysis['cold_hot_distribution'].items():
        print(f"  {category}: {count}个")
    
    print("\n" + "=" * 80)
    print("改进要点")
    print("=" * 80)
    print("1. 多窗口分析（5/10/20/30/50期）")
    print("2. 排除最近5期出现的数字")
    print("3. 冷热号平衡配置")
    print("4. 区域动态平衡")
    print("5. 自适应权重调整")
    print("\n预期目标：")
    print("  - Top15 命中率: 50%+")
    print("  - 最长连续失败: ≤5期")
    print("  - 稳定性: 前后期表现一致")


if __name__ == '__main__':
    main()
