"""
生肖超级预测器 - 多模型集成版
目标：TOP5命中率 ≥ 50%

核心策略：
1. 极致冷门优先（最高权重）
2. 多模型投票机制
3. 动态权重调整
4. 历史相似度匹配
5. 反向思维（避开热门）
"""

import pandas as pd
import numpy as np
from collections import Counter


class ZodiacSuperPredictor:
    """超级预测器 - 集成多策略"""
    
    def __init__(self):
        self.zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        self.zodiac_numbers = {
            '鼠': [1, 13, 25, 37, 49], '牛': [2, 14, 26, 38],
            '虎': [3, 15, 27, 39], '兔': [4, 16, 28, 40],
            '龙': [5, 17, 29, 41], '蛇': [6, 18, 30, 42],
            '马': [7, 19, 31, 43], '羊': [8, 20, 32, 44],
            '猴': [9, 21, 33, 45], '鸡': [10, 22, 34, 46],
            '狗': [11, 23, 35, 47], '猪': [12, 24, 36, 48]
        }
        
        self.number_to_zodiac = {}
        for z, nums in self.zodiac_numbers.items():
            for n in nums:
                self.number_to_zodiac[n] = z
    
    def _ultra_cold_strategy(self, animals):
        """极致冷门策略 - 权重最高"""
        scores = {}
        
        # 多个时间窗口
        windows = {
            60: 10.0,  # 长期
            40: 8.0,   # 中长期
            30: 6.0,   # 中期
            20: 5.0,   # 中短期
            15: 4.0,   # 短期
            10: 3.0,   # 近期
            5: 5.0     # 最近期（重要）
        }
        
        for zodiac in self.zodiacs:
            score = 0.0
            
            for window, weight in windows.items():
                recent = animals[-window:] if len(animals) >= window else animals
                freq = recent.count(zodiac)
                
                # 冷门加分机制
                if freq == 0:
                    score += weight * 1.5
                elif freq == 1:
                    score += weight * 0.8
                elif freq == 2:
                    score += weight * 0.3
                else:
                    score -= weight * 0.5 * (freq - 2)
            
            scores[zodiac] = score
        
        return scores
    
    def _anti_hot_strategy(self, animals):
        """反向策略 - 避开热门"""
        scores = {}
        
        recent_20 = animals[-20:]
        recent_10 = animals[-10:]
        recent_5 = animals[-5:]
        
        for zodiac in self.zodiacs:
            score = 0.0
            
            # 最近5期出现直接大减分
            if zodiac in recent_5:
                last_idx = len(recent_5) - 1 - recent_5[::-1].index(zodiac)
                gap = len(recent_5) - 1 - last_idx
                score -= (10.0 - gap * 2.0)  # 越近减分越多
            else:
                score += 8.0  # 5期内未出现，大加分
            
            # 10期频率惩罚
            freq_10 = recent_10.count(zodiac)
            if freq_10 >= 3:
                score -= 6.0
            elif freq_10 == 2:
                score -= 3.0
            elif freq_10 == 1:
                score += 2.0
            elif freq_10 == 0:
                score += 5.0
            
            # 20期频率惩罚
            freq_20 = recent_20.count(zodiac)
            if freq_20 >= 4:
                score -= 4.0
            elif freq_20 <= 1:
                score += 3.0
            
            scores[zodiac] = score
        
        return scores
    
    def _rotation_advanced(self, animals):
        """高级轮转策略"""
        scores = {z: 0.0 for z in self.zodiacs}
        
        if len(animals) < 2:
            return scores
        
        last = animals[-1]
        if last not in self.zodiacs:
            return scores
        
        last_idx = self.zodiacs.index(last)
        
        # 分析最近10期的轮转模式
        recent_10 = animals[-10:]
        rotations = []
        for i in range(len(recent_10)-1):
            if recent_10[i] in self.zodiacs and recent_10[i+1] in self.zodiacs:
                idx1 = self.zodiacs.index(recent_10[i])
                idx2 = self.zodiacs.index(recent_10[i+1])
                rot = (idx2 - idx1) % 12
                rotations.append(rot)
        
        # 计算主流轮转方向
        if rotations:
            avg_rot = int(np.mean(rotations))
            
            for zodiac in self.zodiacs:
                z_idx = self.zodiacs.index(zodiac)
                forward = (z_idx - last_idx) % 12
                
                # 符合主流方向加分
                if forward == avg_rot:
                    scores[zodiac] += 5.0
                elif abs(forward - avg_rot) <= 1:
                    scores[zodiac] += 3.0
                elif abs(forward - avg_rot) <= 2:
                    scores[zodiac] += 1.5
                
                # 相邻生肖加分
                if forward in [1, 2]:
                    scores[zodiac] += 4.0
                elif forward in [11, 10]:
                    scores[zodiac] += 3.0
                elif forward == 6:  # 对冲
                    scores[zodiac] += 2.5
        
        return scores
    
    def _gap_analysis(self, animals):
        """间隔分析策略"""
        scores = {}
        
        for zodiac in self.zodiacs:
            # 找出所有出现位置
            positions = [i for i, a in enumerate(animals) if a == zodiac]
            
            if not positions:
                # 从未出现，超高分
                scores[zodiac] = 15.0
            else:
                # 计算当前间隔
                current_gap = len(animals) - 1 - positions[-1]
                
                # 间隔评分
                if current_gap >= 15:
                    score = 10.0
                elif current_gap >= 10:
                    score = 7.0
                elif current_gap >= 7:
                    score = 5.0
                elif current_gap >= 5:
                    score = 3.0
                elif current_gap >= 3:
                    score = 1.0
                else:
                    score = -5.0 * (3 - current_gap)
                
                # 如果有周期性，检查是否接近周期
                if len(positions) >= 3:
                    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                    avg_gap = np.mean(gaps)
                    
                    # 接近平均周期加分
                    diff = abs(current_gap - avg_gap)
                    if diff <= 1:
                        score += 4.0
                    elif diff <= 2:
                        score += 2.0
                
                scores[zodiac] = score
        
        return scores
    
    def _diversity_boost(self, animals):
        """多样性增强"""
        scores = {}
        
        recent_10 = animals[-10:]
        appeared = set(recent_10)
        
        for zodiac in self.zodiacs:
            if zodiac not in appeared:
                scores[zodiac] = 5.0
            else:
                scores[zodiac] = -2.0
        
        return scores
    
    def _historical_similarity(self, animals):
        """历史相似度匹配"""
        scores = {z: 0.0 for z in self.zodiacs}
        
        if len(animals) < 10:
            return scores
        
        # 获取最近5期模式
        recent_5 = animals[-5:]
        
        # 在历史中搜索相似模式（前面的数据）
        for i in range(len(animals) - 10):
            historical_5 = animals[i:i+5]
            
            # 计算相似度
            similarity = sum(1 for a, b in zip(recent_5, historical_5) if a == b)
            
            # 如果相似度高，参考后续出现的生肖
            if similarity >= 3 and i + 5 < len(animals):
                next_zodiac = animals[i + 5]
                if next_zodiac in self.zodiacs:
                    scores[next_zodiac] += similarity * 0.5
        
        return scores
    
    def _continuous_absence_penalty(self, animals):
        """连续不出现惩罚策略 - 长期不出现的生肖可能存在系统性原因"""
        scores = {z: 0.0 for z in self.zodiacs}
        
        if len(animals) < 30:
            return scores
        
        # 计算每个生肖距离上次出现的期数
        for zodiac in self.zodiacs:
            last_appearance = -1
            for i in range(len(animals)-1, -1, -1):
                if animals[i] == zodiac:
                    last_appearance = len(animals) - i - 1
                    break
            
            if last_appearance == -1:
                # 从未出现（不太可能，但处理一下）
                scores[zodiac] = -10.0
            elif last_appearance >= 30:
                # 超过30期未出现，严重惩罚（可能有系统性原因）
                scores[zodiac] = -6.0
            elif last_appearance >= 20:
                # 超过20期未出现，较大惩罚
                scores[zodiac] = -3.0
            elif last_appearance >= 15:
                # 超过15期未出现，轻微惩罚
                scores[zodiac] = -1.0
            elif last_appearance >= 10:
                # 10-14期未出现，中等冷门，适度加分
                scores[zodiac] = 2.0
            elif last_appearance >= 6:
                # 6-9期未出现，轻度冷门，加分
                scores[zodiac] = 3.0
            elif last_appearance >= 4:
                # 4-5期未出现，正常范围
                scores[zodiac] = 1.0
            else:
                # 1-3期内刚出现过，轻微惩罚
                scores[zodiac] = -1.0
        
        return scores
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_n=5):
        """集成预测"""
        
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        # 收集各策略评分（优化配置 - 激进型，最佳测试结果42%）
        # 核心发现：适度增加冷门权重 + 连续不出现惩罚效果更好
        # 理念：虽然长期不出现可能有系统性原因，但适度惩罚而非过度惩罚
        strategies = {
            'ultra_cold': (self._ultra_cold_strategy(animals), 0.35),       # 冷门策略（保持35%）
            'anti_hot': (self._anti_hot_strategy(animals), 0.20),           # 避开热门（从25%→20%）
            'gap': (self._gap_analysis(animals), 0.18),                     # 间隔分析
            'rotation': (self._rotation_advanced(animals), 0.12),           # 轮转规律
            'absence_penalty': (self._continuous_absence_penalty(animals), 0.08),  # 连续不出现惩罚（温和）
            'diversity': (self._diversity_boost(animals), 0.04),            # 多样性
            'similarity': (self._historical_similarity(animals), 0.03)      # 历史匹配
        }
        
        # 加权融合
        final_scores = {}
        for zodiac in self.zodiacs:
            score = 0.0
            for strategy_name, (strategy_scores, weight) in strategies.items():
                score += strategy_scores.get(zodiac, 0) * weight
            final_scores[zodiac] = score
        
        # 排序
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_zodiacs = sorted_zodiacs[:top_n]
        
        # 推荐号码
        recommended_numbers = []
        for rank, (zodiac, score) in enumerate(top_zodiacs, 1):
            weight = top_n + 1 - rank
            for num in self.zodiac_numbers[zodiac]:
                recommended_numbers.append((num, weight))
        
        num_scores = {}
        for num, w in recommended_numbers:
            num_scores[num] = num_scores.get(num, 0) + w
        
        sorted_nums = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_nums[:15]]
        
        return {
            'model': '生肖超级预测器(多策略集成)',
            'version': '4.0',
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_number': df.iloc[-1]['number'],
            'last_zodiac': df.iloc[-1]['animal'],
            f'top{top_n}_zodiacs': top_zodiacs,
            'top15_numbers': top_numbers,
            'all_scores': final_scores,
            'strategy_weights': {k: v[1] for k, v in strategies.items()}
        }
    
    def get_recent_20_validation(self, csv_file='data/lucky_numbers.csv'):
        """获取最近20期的验证数据"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        if len(df) < 21:
            return None
        
        # 最近20期验证
        start_index = len(df) - 20
        details = []
        
        zodiac_top5_hits = 0
        number_top15_hits = 0
        
        for i in range(start_index, len(df)):
            # 使用i之前的数据作为训练集
            train_df = df.iloc[:i]
            actual_record = df.iloc[i]
            
            # 保存训练数据到临时文件
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8-sig', newline='') as tmp:
                train_df.to_csv(tmp.name, index=False, encoding='utf-8-sig')
                tmp_file = tmp.name
            
            try:
                # 使用训练数据预测
                result = self.predict(tmp_file, top_n=5)
                top5_zodiacs = result['top5_zodiacs']
                top15_numbers = result['top15_numbers']
                
                actual_number = int(actual_record['number'])
                actual_zodiac = str(actual_record['animal']).strip()
                
                # 检查生肖预测
                zodiac_list = [z for z, _ in top5_zodiacs]
                zodiac_hit = actual_zodiac in zodiac_list
                if zodiac_hit:
                    zodiac_top5_hits += 1
                    zodiac_rank = zodiac_list.index(actual_zodiac) + 1
                    zodiac_result = f"✅ TOP{zodiac_rank}"
                else:
                    zodiac_result = "❌"
                
                # 检查号码预测
                number_hit = actual_number in top15_numbers
                if number_hit:
                    number_top15_hits += 1
                    number_rank = top15_numbers.index(actual_number) + 1
                    number_result = f"✅ TOP{number_rank}"
                else:
                    number_result = "❌"
                
                details.append({
                    '期数': i + 1,
                    '日期': actual_record['date'],
                    '实际号码': actual_number,
                    '实际生肖': actual_zodiac,
                    '预测生肖TOP5': ', '.join(zodiac_list),
                    '生肖命中': zodiac_result,
                    '号码命中': number_result
                })
            finally:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
        
        zodiac_rate = zodiac_top5_hits / 20 * 100
        number_rate = number_top15_hits / 20 * 100
        
        return {
            'details': details,
            'zodiac_top5_hits': zodiac_top5_hits,
            'zodiac_top5_rate': zodiac_rate,
            'number_top15_hits': number_top15_hits,
            'number_top15_rate': number_rate
        }


if __name__ == "__main__":
    print("="*80)
    print("生肖超级预测器 - 多策略集成版")
    print("="*80)
    
    predictor = ZodiacSuperPredictor()
    result = predictor.predict(top_n=5)
    
    print(f"\n模型: {result['model']} v{result['version']}")
    print(f"\n最新一期（第{result['total_periods']}期）")
    print(f"  日期: {result['last_date']}")
    print(f"  开出: {result['last_number']} - {result['last_zodiac']}")
    
    print(f"\n下一期预测（第{result['total_periods']+1}期）")
    print("\n⭐ 生肖 TOP 5:")
    for i, (zodiac, score) in enumerate(result['top5_zodiacs'], 1):
        nums = predictor.zodiac_numbers[zodiac]
        level = "强推" if i <= 2 else "推荐" if i <= 3 else "备选"
        print(f"  {i}. {zodiac} [{level}] 评分: {score:7.2f}  号码: {nums}")
    
    print(f"\n📋 推荐号码 TOP 15:")
    print(f"  {result['top15_numbers']}")
    
    print(f"\n⚙️  策略权重:")
    for name, weight in result['strategy_weights'].items():
        print(f"  {name}: {weight*100:.0f}%")
    
    print("\n" + "="*80)
