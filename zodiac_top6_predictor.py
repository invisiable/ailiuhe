"""
生肖TOP6预测模型
专注于预测最可能出现的6个生肖，并基于此推荐号码

特点：
1. 比TOP5多1个生肖选择，更高的覆盖率
2. 综合多维度分析：频率、轮转、冷热度、周期性
3. 优化评分算法，提升准确率
4. 基于6个生肖推荐TOP18号码
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime


class ZodiacTop6Predictor:
    """生肖TOP6预测器"""
    
    def __init__(self):
        # 12生肖列表
        self.zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        # 生肖对应的号码映射
        self.zodiac_numbers = {
            '鼠': [1, 13, 25, 37, 49],
            '牛': [2, 14, 26, 38],
            '虎': [3, 15, 27, 39],
            '兔': [4, 16, 28, 40],
            '龙': [5, 17, 29, 41],
            '蛇': [6, 18, 30, 42],
            '马': [7, 19, 31, 43],
            '羊': [8, 20, 32, 44],
            '猴': [9, 21, 33, 45],
            '鸡': [10, 22, 34, 46],
            '狗': [11, 23, 35, 47],
            '猪': [12, 24, 36, 48]
        }
        
        # 反向映射：号码到生肖
        self.number_to_zodiac = {}
        for zodiac, numbers in self.zodiac_numbers.items():
            for num in numbers:
                self.number_to_zodiac[num] = zodiac
        
        self.version = "1.0"
        self.model_name = "生肖TOP6预测模型"
    
    def _analyze_zodiac_pattern(self, animals):
        """分析生肖规律"""
        recent_50 = animals[-50:] if len(animals) >= 50 else animals
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        recent_5 = animals[-5:] if len(animals) >= 5 else animals
        recent_3 = animals[-3:] if len(animals) >= 3 else animals
        
        # 统计不同时间窗口的频率
        freq_50 = Counter(recent_50)
        freq_30 = Counter(recent_30)
        freq_20 = Counter(recent_20)
        freq_10 = Counter(recent_10)
        freq_5 = Counter(recent_5)
        
        # 检查是否有连续出现
        has_consecutive = len(recent_5) >= 2 and recent_5[-1] == recent_5[-2]
        
        # 检查循环模式（是否按十二生肖顺序）
        zodiac_indices = []
        for animal in recent_10:
            if animal.strip() in self.zodiacs:
                idx = self.zodiacs.index(animal.strip())
                zodiac_indices.append(idx)
        
        # 计算平均间隔
        if len(zodiac_indices) >= 2:
            intervals = [(zodiac_indices[i+1] - zodiac_indices[i]) % 12 for i in range(len(zodiac_indices)-1)]
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
        else:
            avg_interval = 0
        
        # 计算周期性模式
        zodiac_cycle_pattern = {}
        for i, zodiac in enumerate(self.zodiacs):
            positions = [idx for idx, animal in enumerate(recent_30) if animal.strip() == zodiac]
            if len(positions) >= 2:
                gaps = [positions[j+1] - positions[j] for j in range(len(positions)-1)]
                zodiac_cycle_pattern[zodiac] = np.mean(gaps) if gaps else 0
            else:
                zodiac_cycle_pattern[zodiac] = 0
        
        return {
            'recent_50': recent_50,
            'recent_30': recent_30,
            'recent_20': recent_20,
            'recent_10': recent_10,
            'recent_5': recent_5,
            'recent_3': recent_3,
            'freq_50': freq_50,
            'freq_30': freq_30,
            'freq_20': freq_20,
            'freq_10': freq_10,
            'freq_5': freq_5,
            'has_consecutive': has_consecutive,
            'last_zodiac': recent_5[-1].strip() if len(recent_5) > 0 else None,
            'avg_interval': avg_interval,
            'cycle_pattern': zodiac_cycle_pattern
        }
    
    def predict_zodiac_top6(self, csv_file='data/lucky_numbers.csv'):
        """
        预测TOP6最可能出现的生肖
        
        Args:
            csv_file: 数据文件路径
        
        Returns:
            list: [(生肖, 评分), ...] TOP6生肖及其评分
        """
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = df['animal'].values
        
        pattern = self._analyze_zodiac_pattern(animals)
        
        # 综合评分
        scores = {}
        
        for zodiac in self.zodiacs:
            score = 0.0
            
            # ===== 方法1: 多时间窗口频率分析（权重30%）=====
            freq_50 = pattern['freq_50'].get(zodiac, 0)
            freq_30 = pattern['freq_30'].get(zodiac, 0)
            freq_20 = pattern['freq_20'].get(zodiac, 0)
            freq_10 = pattern['freq_10'].get(zodiac, 0)
            
            # 长期冷门生肖（50期）
            if freq_50 <= 2:
                score += 4.0  # 长期冷门，强力推荐
            elif freq_50 <= 3:
                score += 2.5
            elif freq_50 <= 4:
                score += 1.0
            
            # 中期冷门生肖（30期）
            if freq_30 == 0:
                score += 3.5  # 30期内未出现
            elif freq_30 == 1:
                score += 2.5
            elif freq_30 == 2:
                score += 1.5
            
            # 短期冷门（20期）
            if freq_20 == 0:
                score += 2.5
            elif freq_20 == 1:
                score += 1.5
            
            # 近期冷门（10期）
            if freq_10 == 0:
                score += 1.5
            
            # ===== 方法2: 强化避重机制（权重35%）=====
            # 最近5期出现过的大幅降权
            if zodiac in pattern['recent_5']:
                last_appear_idx = len(pattern['recent_5']) - 1 - list(reversed(pattern['recent_5'])).index(zodiac)
                gap = len(pattern['recent_5']) - 1 - last_appear_idx
                
                if gap == 0:  # 上一期刚出现
                    score -= 4.5  # 大幅降权
                elif gap == 1:  # 倒数第2期
                    score -= 3.0
                elif gap == 2:  # 倒数第3期
                    score -= 2.0
                elif gap == 3:  # 倒数第4期
                    score -= 1.0
                else:  # 倒数第5期
                    score -= 0.5
            else:
                score += 3.0  # 最近5期未出现，加分
            
            # 连续出现惩罚
            if pattern['has_consecutive'] and pattern['last_zodiac'] == zodiac:
                score -= 3.0  # 避免连续
            
            # ===== 方法3: 生肖轮转与相邻规律（权重20%）=====
            last_zodiac = pattern['last_zodiac']
            if last_zodiac and last_zodiac in self.zodiacs:
                last_idx = self.zodiacs.index(last_zodiac)
                zodiac_idx = self.zodiacs.index(zodiac)
                
                # 计算顺序距离（考虑循环）
                forward_dist = (zodiac_idx - last_idx) % 12
                backward_dist = (last_idx - zodiac_idx) % 12
                
                # 相邻生肖（前后2-3个）加分
                if forward_dist in [1, 2]:  # 顺序相邻
                    score += 2.0
                elif forward_dist == 3:
                    score += 1.0
                elif backward_dist in [1, 2]:  # 逆序相邻
                    score += 1.5
                elif backward_dist == 3:
                    score += 0.5
                
                # 对冲生肖（相距6个）
                if forward_dist == 6:
                    score += 1.0
            
            # ===== 方法4: 周期性规律（权重10%）=====
            # 如果该生肖有明显周期，且接近周期点
            cycle = pattern['cycle_pattern'].get(zodiac, 0)
            if cycle > 0 and freq_30 > 0:
                # 计算距离上次出现的期数
                try:
                    positions = [idx for idx, animal in enumerate(pattern['recent_30']) 
                                if animal.strip() == zodiac]
                    if positions:
                        last_pos = positions[-1]
                        gap_since_last = len(pattern['recent_30']) - 1 - last_pos
                        
                        # 如果接近周期点（±2期）
                        if abs(gap_since_last - cycle) <= 2:
                            score += 2.0
                        elif abs(gap_since_last - cycle) <= 4:
                            score += 1.0
                except:
                    pass
            
            # ===== 方法5: 热度均衡（权重5%）=====
            # 保持12生肖出现均衡
            avg_freq_30 = len(pattern['recent_30']) / 12
            deviation = freq_30 - avg_freq_30
            
            if deviation < -1.5:  # 远低于平均
                score += 2.0
            elif deviation < -0.5:  # 低于平均
                score += 1.0
            elif deviation > 1.5:  # 远高于平均
                score -= 1.5
            elif deviation > 0.5:  # 高于平均
                score -= 0.5
            
            scores[zodiac] = score
        
        # 排序并返回TOP6
        sorted_zodiacs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_zodiacs[:6]
    
    def predict_numbers_by_zodiac(self, top_zodiacs, recent_numbers=None):
        """
        根据预测的TOP6生肖，推荐对应的号码
        
        Args:
            top_zodiacs: TOP6生肖列表 [(生肖, 评分), ...]
            recent_numbers: 最近的号码列表，用于避重
        
        Returns:
            list: TOP18号码
        """
        # 收集所有候选号码及其权重
        number_scores = {}
        
        for rank, (zodiac, zodiac_score) in enumerate(top_zodiacs, 1):
            # 获取该生肖对应的所有号码
            numbers = self.zodiac_numbers.get(zodiac, [])
            
            # 根据生肖排名给号码加权
            weight = 7 - rank  # TOP1权重6，TOP6权重1
            
            for num in numbers:
                if num not in number_scores:
                    number_scores[num] = 0
                # 累加权重和生肖评分
                number_scores[num] += weight * (1 + zodiac_score * 0.1)
        
        # 如果提供了最近号码，对最近出现的号码降权
        if recent_numbers is not None and len(recent_numbers) > 0:
            recent_5 = set(recent_numbers[-5:]) if len(recent_numbers) >= 5 else set(recent_numbers)
            recent_10 = set(recent_numbers[-10:]) if len(recent_numbers) >= 10 else set(recent_numbers)
            
            for num in number_scores:
                if num in recent_5:
                    number_scores[num] *= 0.3  # 最近5期出现，大幅降权
                elif num in recent_10:
                    number_scores[num] *= 0.6  # 最近10期出现，适度降权
        
        # 按评分排序
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回TOP18（6个生肖 × 平均3个号码）
        recommended = [num for num, score in sorted_numbers]
        
        # 如果不足18个，从所有号码中按规则补充
        if len(recommended) < 18:
            all_numbers = list(range(1, 50))
            for num in all_numbers:
                if num not in recommended:
                    # 优先补充中间范围的号码
                    if 15 <= num <= 35:
                        recommended.append(num)
                        if len(recommended) >= 18:
                            break
            
            # 如果还不够，继续补充
            for num in all_numbers:
                if num not in recommended:
                    recommended.append(num)
                    if len(recommended) >= 18:
                        break
        
        return recommended[:18]
    
    def predict(self, csv_file='data/lucky_numbers.csv'):
        """
        完整预测流程
        
        Args:
            csv_file: 数据文件路径
        
        Returns:
            dict: 预测信息字典，包含：
                - top6_zodiacs: TOP6生肖及评分
                - top18_numbers: 基于生肖的TOP18号码
                - last_date: 最新一期日期
                - last_number: 最新一期号码
                - last_zodiac: 最新一期生肖
                - total_periods: 总期数
                - model_info: 模型信息
        """
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        # 1. 预测TOP6生肖
        top6_zodiacs = self.predict_zodiac_top6(csv_file)
        
        # 2. 根据TOP6生肖推荐号码
        recent_numbers = df['number'].values
        top18_numbers = self.predict_numbers_by_zodiac(top6_zodiacs, recent_numbers)
        
        # 3. 获取最新信息
        last_row = df.iloc[-1]
        last_date = last_row['date']
        last_number = int(last_row['number'])
        last_zodiac = last_row['animal']
        total_periods = len(df)
        
        return {
            'top6_zodiacs': top6_zodiacs,
            'top18_numbers': top18_numbers,
            'last_date': last_date,
            'last_number': last_number,
            'last_zodiac': last_zodiac,
            'total_periods': total_periods,
            'model_info': {
                'name': self.model_name,
                'version': self.version,
                'description': '生肖TOP6预测模型 - 专注于6个最可能生肖的精准预测'
            }
        }
    
    def validate(self, csv_file='data/lucky_numbers.csv', test_periods=20):
        """
        验证模型准确率
        
        Args:
            csv_file: 数据文件路径
            test_periods: 测试期数
        
        Returns:
            dict: 验证结果，包含：
                - zodiac_top6_rate: 生肖TOP6命中率
                - number_top18_rate: 号码TOP18命中率
                - details: 每期详细结果
        """
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        if len(df) < test_periods + 30:
            test_periods = max(1, len(df) - 30)
        
        zodiac_hits = 0
        number_hits = 0
        details = []
        
        for i in range(test_periods):
            # 使用前N-i期数据进行预测
            test_idx = len(df) - test_periods + i
            train_df = df.iloc[:test_idx]
            actual_row = df.iloc[test_idx]
            
            # 保存临时数据
            temp_file = 'temp_validate.csv'
            train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
            
            # 预测
            try:
                result = self.predict(temp_file)
                
                # 检查生肖命中
                actual_zodiac = actual_row['animal'].strip()
                predicted_zodiacs = [z for z, s in result['top6_zodiacs']]
                zodiac_hit = actual_zodiac in predicted_zodiacs
                if zodiac_hit:
                    zodiac_hits += 1
                
                # 检查号码命中
                actual_number = int(actual_row['number'])
                number_hit = actual_number in result['top18_numbers']
                if number_hit:
                    number_hits += 1
                
                details.append({
                    '期号': test_idx + 1,
                    '日期': actual_row['date'],
                    '实际号码': actual_number,
                    '实际生肖': actual_zodiac,
                    '预测生肖TOP6': predicted_zodiacs,
                    '预测号码TOP18': result['top18_numbers'],
                    '生肖命中': '✓' if zodiac_hit else '✗',
                    '号码命中': '✓' if number_hit else '✗'
                })
            except Exception as e:
                print(f"第{test_idx+1}期预测失败: {e}")
                continue
        
        # 清理临时文件
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return {
            'test_periods': test_periods,
            'zodiac_top6_hits': zodiac_hits,
            'zodiac_top6_rate': zodiac_hits / test_periods * 100,
            'number_top18_hits': number_hits,
            'number_top18_rate': number_hits / test_periods * 100,
            'details': details
        }


if __name__ == '__main__':
    # 演示使用
    predictor = ZodiacTop6Predictor()
    
    print("=" * 80)
    print("🎯 生肖TOP6预测模型")
    print("=" * 80)
    
    # 预测
    result = predictor.predict()
    
    print(f"\n📅 最新一期（第{result['total_periods']}期）:")
    print(f"   日期: {result['last_date']}")
    print(f"   开出: {result['last_number']} - {result['last_zodiac']}")
    
    print(f"\n🔮 下一期预测（第{result['total_periods']+1}期）:\n")
    
    # 显示生肖预测
    print("⭐ 推荐生肖 TOP 6:")
    print("-" * 80)
    for i, (zodiac, score) in enumerate(result['top6_zodiacs'], 1):
        nums = predictor.zodiac_numbers[zodiac]
        emoji = "⭐⭐" if i <= 2 else "⭐" if i <= 4 else "✓"
        print(f"{emoji} {i}. {zodiac:2s} (评分: {score:6.2f})  对应号码: {nums}")
    
    # 显示号码推荐
    print(f"\n📋 推荐号码（基于TOP6生肖）:")
    print("-" * 80)
    top6 = result['top18_numbers'][:6]
    top12 = result['top18_numbers'][6:12]
    top18 = result['top18_numbers'][12:18]
    
    print(f"   TOP 1-6:   {top6}")
    print(f"   TOP 7-12:  {top12}")
    print(f"   TOP 13-18: {top18}")
    
    # 验证模型
    print(f"\n{'='*80}")
    print("📊 模型验证（最近20期）")
    print("=" * 80)
    
    validation = predictor.validate(test_periods=20)
    
    print(f"   生肖 TOP6 命中率: {validation['zodiac_top6_rate']:.1f}% "
          f"({validation['zodiac_top6_hits']}/{validation['test_periods']})")
    print(f"   号码 TOP18 命中率: {validation['number_top18_rate']:.1f}% "
          f"({validation['number_top18_hits']}/{validation['test_periods']})")
    
    # 使用建议
    print(f"\n{'='*80}")
    print("💡 使用建议")
    print("=" * 80)
    print("   1. ⭐⭐ 重点关注TOP2生肖（成功率最高）")
    print("   2. ⭐ 次要关注TOP3-4生肖")
    print("   3. ✓ TOP5-6作为备选")
    print("   4. 📋 号码推荐已按生肖排名加权，优先选择TOP1-6")
    print("=" * 80 + "\n")
