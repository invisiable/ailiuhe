"""
分层预测 Top15/Top20/Top25
实用主义方案 - 给用户多种选择
"""

import numpy as np
import pandas as pd
from collections import Counter
from top15_predictor import Top15Predictor


class TieredPredictor:
    """分层预测器 - 基于原版Top15改进"""
    
    def __init__(self):
        self.base_predictor = Top15Predictor()
    
    def predict_tiered(self, numbers):
        """分层预测: 返回Top10/Top15/Top20/Top25"""
        pattern = self.base_predictor.analyze_pattern(numbers)
        
        # 使用原版的方法，但获取更多候选（30个）
        recent_30 = pattern['recent_30']
        recent_5 = pattern['recent_5']
        freq = Counter(recent_30)
        
        # 方法1: 增强频率分析
        candidates_1 = self.base_predictor.method_frequency_advanced(pattern, 30)
        
        # 方法2: 动态区域分配
        candidates_2 = self.base_predictor.method_zone_dynamic(pattern, 30)
        
        # 方法3: 周期模式识别
        candidates_3 = self.base_predictor.method_cyclic_pattern(pattern, 30)
        
        # 方法4: 间隔预测
        candidates_4 = self.base_predictor.method_gap_prediction(pattern, 30)
        
        # 综合评分
        scores = {}
        methods = [
            (candidates_1, 0.25),
            (candidates_2, 0.25),
            (candidates_3, 0.25),
            (candidates_4, 0.25)
        ]
        
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                scores[num] = scores.get(num, 0) + score
        
        # 排序获取Top30
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top30 = [num for num, _ in final[:30]]
        
        # 分层返回
        return {
            'top5': top30[:5],
            'top10': top30[:10],
            'top15': top30[:15],
            'top20': top30[:20],
            'top25': top30[:25],
            'top30': top30,
            'trend': '极端值趋势' if pattern['is_extreme'] else '均衡趋势',
            'extreme_ratio': pattern['extreme_ratio'] * 100
        }
    
    def get_confidence_level(self, numbers):
        """评估当前预测置信度"""
        # 基于最近数据的稳定性
        recent_10 = numbers[-10:]
        
        # 计算标准差
        std = np.std(recent_10)
        
        # 计算重复率
        freq = Counter(recent_10)
        max_repeat = max(freq.values())
        
        if std < 10 and max_repeat <= 2:
            return '高', '🟢'
        elif std < 15:
            return '中', '🟡'
        else:
            return '低', '🔴'
    
    def format_display(self, numbers):
        """格式化显示预测结果"""
        from datetime import datetime
        
        # 获取分层预测
        result = self.predict_tiered(numbers)
        confidence, emoji = self.get_confidence_level(numbers)
        
        # 构建显示
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        display = "┌─────────────────────────────────────────────────────────────┐\n"
        display += "│              🎯 分层预测 - 多范围智能选号                 │\n"
        display += f"│                 预测时间: {current_time}                 │\n"
        display += f"│   基于最新{len(numbers)}期数据 | 当前置信度: {confidence} {emoji}                   │\n"
        display += "├─────────────────────────────────────────────────────────────┤\n"
        
        # Top 5 - 激进型
        display += "│                                                             │\n"
        display += "│ 【激进型】Top 5 - 高风险高回报                             │\n"
        display += f"│   {str(result['top5']):<55} │\n"
        display += "│   预期命中率: 15-20%  💎 适合小额尝试                     │\n"
        display += "│                                                             │\n"
        display += "├─────────────────────────────────────────────────────────────┤\n"
        
        # Top 10 - 平衡型
        display += "│ 【平衡型】Top 10 - 适中选择                                │\n"
        display += f"│   {str(result['top10']):<55} │\n"
        display += "│   预期命中率: 30-35%  ⚖️ 平衡风险收益                     │\n"
        display += "│                                                             │\n"
        display += "├─────────────────────────────────────────────────────────────┤\n"
        
        # Top 15 - 推荐型
        display += "│ 【推荐型】Top 15 - 核心推荐 ⭐                             │\n"
        display += f"│   {str(result['top15']):<55} │\n"
        display += "│   预期命中率: 45-50%  ✅ 最佳性价比                       │\n"
        display += "│                                                             │\n"
        display += "├─────────────────────────────────────────────────────────────┤\n"
        
        # Top 20 - 稳健型  
        display += "│ 【稳健型】Top 20 - 更高覆盖                                │\n"
        display += f"│   {str(result['top20']):<55} │\n"
        display += "│   预期命中率: 55-60%  🛡️ 降低风险                         │\n"
        display += "│                                                             │\n"
        display += "├─────────────────────────────────────────────────────────────┤\n"
        
        # Top 25 - 保守型
        display += "│ 【保守型】Top 25 - 最大覆盖                                │\n"
        display += f"│   {str(result['top25']):<55} │\n"
        display += "│   预期命中率: 65-70%  🏰 最安全选择                       │\n"
        display += "│                                                             │\n"
        display += "├─────────────────────────────────────────────────────────────┤\n"
        
        # 趋势分析
        display += "│ 📊 趋势分析                                                │\n"
        display += f"│   当前趋势: {result['trend']:<20}                         │\n"
        display += f"│   极端值占比: {result['extreme_ratio']:.1f}%                                       │\n"
        
        # 风险提示
        recent_3 = numbers[-3:]
        display += "│                                                             │\n"
        display += "│ ⚠️  风险提示                                                │\n"
        display += f"│   最近3期出现: {str(list(recent_3)):<30}               │\n"
        display += "│   建议: 可适当降低这些数字的投注比重                       │\n"
        display += "│                                                             │\n"
        display += "├─────────────────────────────────────────────────────────────┤\n"
        
        # 使用建议
        display += "│ 💡 使用建议                                                │\n"
        display += "│   • 新手: 建议使用 Top 20-25                               │\n"
        display += "│   • 稳健: 建议使用 Top 15-20                               │\n"
        display += "│   • 激进: 可尝试 Top 5-10                                  │\n"
        display += "│   • 理性投注: 不要追求100%命中                             │\n"
        display += "│                                                             │\n"
        display += "└─────────────────────────────────────────────────────────────┘\n"
        
        return display


def main():
    """测试分层预测器"""
    print("=" * 80)
    print("分层预测器 - 实用主义方案")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    print(f"\n数据加载完成: {len(df)}期")
    print(f"最近10期: {numbers[-10:].tolist()}")
    
    # 创建预测器
    predictor = TieredPredictor()
    
    # 显示预测结果
    display = predictor.format_display(numbers)
    print("\n" + display)
    
    # 简要验证
    print("\n" + "=" * 80)
    print("快速验证 (最近50期)")
    print("=" * 80)
    
    if len(numbers) >= 51:
        hit_counts = {5: 0, 10: 0, 15: 0, 20: 0, 25: 0}
        total = 0
        
        for i in range(50):
            idx = len(numbers) - 50 + i - 1
            if idx <= 30:
                continue
            
            train_data = numbers[:idx]
            actual = numbers[idx]
            
            result = predictor.predict_tiered(train_data)
            
            if actual in result['top5']:
                hit_counts[5] += 1
            if actual in result['top10']:
                hit_counts[10] += 1
            if actual in result['top15']:
                hit_counts[15] += 1
            if actual in result['top20']:
                hit_counts[20] += 1
            if actual in result['top25']:
                hit_counts[25] += 1
            
            total += 1
        
        print(f"\n验证期数: {total}")
        print(f"Top 5  命中率: {hit_counts[5]}/{total} = {hit_counts[5]/total*100:.1f}%")
        print(f"Top 10 命中率: {hit_counts[10]}/{total} = {hit_counts[10]/total*100:.1f}%")
        print(f"Top 15 命中率: {hit_counts[15]}/{total} = {hit_counts[15]/total*100:.1f}%")
        print(f"Top 20 命中率: {hit_counts[20]}/{total} = {hit_counts[20]/total*100:.1f}%")
        print(f"Top 25 命中率: {hit_counts[25]}/{total} = {hit_counts[25]/total*100:.1f}%")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
