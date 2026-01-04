"""
生肖预测模型
基于生肖规律预测下一期最可能出现的生肖
"""

import pandas as pd
import numpy as np
from collections import Counter, deque
from datetime import datetime


class ZodiacPredictor:
    """生肖预测器"""
    
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
        self.model_name = "生肖预测模型"
    
    def _analyze_zodiac_pattern(self, animals):
        """分析生肖规律"""
        recent_30 = animals[-30:] if len(animals) >= 30 else animals
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        recent_5 = animals[-5:] if len(animals) >= 5 else animals
        recent_3 = animals[-3:] if len(animals) >= 3 else animals
        
        # 统计频率
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
            intervals = [zodiac_indices[i+1] - zodiac_indices[i] for i in range(len(zodiac_indices)-1)]
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
        else:
            avg_interval = 0
        
        return {
            'recent_30': recent_30,
            'recent_20': recent_20,
            'recent_10': recent_10,
            'recent_5': recent_5,
            'recent_3': recent_3,
            'freq_30': freq_30,
            'freq_20': freq_20,
            'freq_10': freq_10,
            'freq_5': freq_5,
            'has_consecutive': has_consecutive,
            'last_zodiac': recent_5[-1].strip() if len(recent_5) > 0 else None,
            'avg_interval': avg_interval
        }
    
    def predict_zodiac_top5(self, csv_file='data/lucky_numbers.csv'):
        """
        预测TOP5最可能出现的生肖
        返回：[(生肖, 评分), ...]
        """
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = df['animal'].values
        
        pattern = self._analyze_zodiac_pattern(animals)
        
        # 综合评分
        scores = {}
        
        for zodiac in self.zodiacs:
            score = 0.0
            
            # 方法1: 频率分析（权重35%）
            # 最近30期出现少的生肖，可能要出现
            freq_30 = pattern['freq_30'].get(zodiac, 0)
            freq_20 = pattern['freq_20'].get(zodiac, 0)
            freq_10 = pattern['freq_10'].get(zodiac, 0)
            
            # 冷门生肖加分
            if freq_30 == 0:
                score += 3.5  # 30期内未出现
            elif freq_30 == 1:
                score += 2.5
            elif freq_30 == 2:
                score += 1.5
            
            if freq_20 == 0:
                score += 2.0  # 20期内未出现
            elif freq_20 == 1:
                score += 1.0
            
            # 方法2: 避重机制（权重30%）
            # 最近5期出现过的降权
            if zodiac in pattern['recent_5']:
                last_appear = len(pattern['recent_5']) - list(pattern['recent_5']).index(zodiac) - 1
                if last_appear == 0:  # 上一期刚出现
                    score -= 3.0
                elif last_appear == 1:  # 倒数第2期
                    score -= 2.0
                elif last_appear == 2:  # 倒数第3期
                    score -= 1.0
            else:
                score += 2.0  # 最近5期未出现
            
            # 方法3: 生肖轮转规律（权重20%）
            # 根据十二生肖的自然顺序
            last_zodiac = pattern['last_zodiac']
            if last_zodiac and last_zodiac in self.zodiacs:
                last_idx = self.zodiacs.index(last_zodiac)
                zodiac_idx = self.zodiacs.index(zodiac)
                
                # 计算距离（考虑循环）
                forward_dist = (zodiac_idx - last_idx) % 12
                
                # 相邻生肖（前后2个）加分
                if forward_dist in [1, 2, 11, 10]:
                    score += 1.5
                elif forward_dist in [3, 4, 9, 8]:
                    score += 0.5
            
            # 方法4: 热度均衡（权重15%）
            # 保持12生肖出现均衡
            avg_freq_30 = len(pattern['recent_30']) / 12
            if freq_30 < avg_freq_30 * 0.6:
                score += 1.5  # 低于平均
            elif freq_30 > avg_freq_30 * 1.4:
                score -= 1.0  # 高于平均
            
            scores[zodiac] = score
        
        # 排序并返回TOP5
        sorted_zodiacs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_zodiacs[:5]
    
    def predict_numbers_by_zodiac(self, top_zodiacs):
        """
        根据预测的生肖，推荐对应的号码
        返回：TOP15号码
        """
        recommended_numbers = []
        
        for zodiac, score in top_zodiacs:
            # 获取该生肖对应的所有号码
            numbers = self.zodiac_numbers.get(zodiac, [])
            
            # 按一定规则选择号码（优先选择中间范围的）
            for num in numbers:
                if num not in recommended_numbers:
                    recommended_numbers.append(num)
        
        # 如果不足15个，从所有号码中补充
        if len(recommended_numbers) < 15:
            for num in range(1, 50):
                if num not in recommended_numbers:
                    recommended_numbers.append(num)
                if len(recommended_numbers) >= 15:
                    break
        
        return recommended_numbers[:15]
    
    def predict(self, csv_file='data/lucky_numbers.csv'):
        """
        完整预测流程
        返回：预测信息字典
        """
        # 1. 预测TOP5生肖
        top5_zodiacs = self.predict_zodiac_top5(csv_file)
        
        # 2. 根据生肖推荐号码
        top15_numbers = self.predict_numbers_by_zodiac(top5_zodiacs)
        
        # 3. 获取历史信息
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        last_record = df.iloc[-1]
        
        return {
            'top5_zodiacs': top5_zodiacs,
            'top15_numbers': top15_numbers,
            'last_number': int(last_record['number']),
            'last_zodiac': last_record['animal'],
            'last_date': last_record['date'],
            'total_periods': len(df)
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
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8-sig', newline='') as tmp:
                train_df.to_csv(tmp.name, index=False, encoding='utf-8-sig')
                tmp_file = tmp.name
            
            try:
                # 使用训练数据预测
                top5_zodiacs = self.predict_zodiac_top5(tmp_file)
                top15_numbers = self.predict_numbers_by_zodiac(top5_zodiacs)
                
                actual_number = int(actual_record['number'])
                actual_zodiac = actual_record['animal']
                
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
                import os
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
    
    def validate_recent_100_periods(self, csv_file='data/lucky_numbers.csv'):
        """验证最近100期的生肖预测成功率"""
        
        print("=" * 80)
        print("生肖预测模型 - 最近100期验证")
        print("=" * 80)
        print(f"\n模型说明:")
        print("  - 基于生肖规律预测")
        print("  - 综合考虑：频率、避重、轮转规律、热度均衡")
        print("  - 预测TOP5生肖，并推荐对应号码")
        
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        total_records = len(df)
        
        if total_records < 101:
            print(f"\n错误：数据不足100期（当前只有{total_records}期）")
            return
        
        print(f"\n数据信息:")
        print(f"  总记录数: {total_records}")
        print(f"  验证期数: 100期")
        print(f"  验证范围: 第{total_records-100+1}期 到 第{total_records}期")
        
        # 统计结果
        zodiac_top1_hits = 0  # TOP1生肖命中
        zodiac_top3_hits = 0  # TOP3生肖命中
        zodiac_top5_hits = 0  # TOP5生肖命中
        
        number_top5_hits = 0   # 推荐号码TOP5命中
        number_top10_hits = 0  # 推荐号码TOP10命中
        number_top15_hits = 0  # 推荐号码TOP15命中
        
        total_tests = 0
        details = []
        
        print(f"\n{'='*80}")
        print(f"开始验证...")
        print(f"{'='*80}\n")
        
        for i in range(100):
            current_idx = total_records - 100 + i
            train_data = df.iloc[:current_idx + 1]
            
            if current_idx + 1 < total_records:
                next_actual_num = int(df.iloc[current_idx + 1]['number'])
                next_actual_zodiac = df.iloc[current_idx + 1]['animal'].strip()
                next_date = df.iloc[current_idx + 1]['date']
                period_num = current_idx + 2
            else:
                break
            
            # 保存临时训练数据
            temp_file = 'data/temp_zodiac_train.csv'
            train_data.to_csv(temp_file, index=False, encoding='utf-8-sig')
            
            # 预测TOP5生肖
            top5_zodiacs = self.predict_zodiac_top5(temp_file)
            predicted_zodiacs = [z for z, s in top5_zodiacs]
            
            # 根据生肖推荐号码
            recommended_numbers = self.predict_numbers_by_zodiac(top5_zodiacs)
            
            # 检查生肖命中
            hit_zodiac_level = None
            if next_actual_zodiac in predicted_zodiacs:
                rank = predicted_zodiacs.index(next_actual_zodiac) + 1
                if rank == 1:
                    zodiac_top1_hits += 1
                    zodiac_top3_hits += 1
                    zodiac_top5_hits += 1
                    hit_zodiac_level = f"✅ 生肖TOP1 (#{rank})"
                elif rank <= 3:
                    zodiac_top3_hits += 1
                    zodiac_top5_hits += 1
                    hit_zodiac_level = f"✓ 生肖TOP3 (#{rank})"
                else:
                    zodiac_top5_hits += 1
                    hit_zodiac_level = f"○ 生肖TOP5 (#{rank})"
            else:
                hit_zodiac_level = "✗ 生肖未命中"
            
            # 检查号码命中
            hit_number_level = None
            top5_nums = recommended_numbers[:5]
            top10_nums = recommended_numbers[:10]
            top15_nums = recommended_numbers[:15]
            
            if next_actual_num in top5_nums:
                number_top5_hits += 1
                number_top10_hits += 1
                number_top15_hits += 1
                num_rank = top5_nums.index(next_actual_num) + 1
                hit_number_level = f"✅ 号码TOP5 (#{num_rank})"
            elif next_actual_num in top10_nums:
                number_top10_hits += 1
                number_top15_hits += 1
                num_rank = top10_nums.index(next_actual_num) + 1
                hit_number_level = f"✓ 号码TOP10 (#{num_rank})"
            elif next_actual_num in top15_nums:
                number_top15_hits += 1
                num_rank = top15_nums.index(next_actual_num) + 1
                hit_number_level = f"○ 号码TOP15 (#{num_rank})"
            else:
                hit_number_level = "✗ 号码未命中"
            
            total_tests += 1
            
            detail = {
                '期数': period_num,
                '日期': next_date,
                '实际号码': next_actual_num,
                '实际生肖': next_actual_zodiac,
                '预测生肖TOP5': str(predicted_zodiacs),
                '生肖命中': hit_zodiac_level,
                '号码命中': hit_number_level,
                '推荐号码TOP15': str(recommended_numbers)
            }
            details.append(detail)
            
            if (i + 1) % 20 == 0:
                current_zodiac_rate = (zodiac_top5_hits / total_tests) * 100
                print(f"已验证 {i+1}/100 期，当前生肖TOP5成功率: {current_zodiac_rate:.2f}%")
        
        # 计算成功率
        zodiac_top1_rate = (zodiac_top1_hits / total_tests) * 100
        zodiac_top3_rate = (zodiac_top3_hits / total_tests) * 100
        zodiac_top5_rate = (zodiac_top5_hits / total_tests) * 100
        
        number_top5_rate = (number_top5_hits / total_tests) * 100
        number_top10_rate = (number_top10_hits / total_tests) * 100
        number_top15_rate = (number_top15_hits / total_tests) * 100
        
        print(f"\n{'='*80}")
        print("验证结果统计")
        print(f"{'='*80}\n")
        
        print(f"总验证期数: {total_tests}\n")
        
        print("【生肖预测成功率】")
        print(f"  ⭐ 生肖 TOP 1: {zodiac_top1_rate:.2f}% ({zodiac_top1_hits}/{total_tests})")
        print(f"  ✓  生肖 TOP 3: {zodiac_top3_rate:.2f}% ({zodiac_top3_hits}/{total_tests})")
        print(f"  ○  生肖 TOP 5: {zodiac_top5_rate:.2f}% ({zodiac_top5_hits}/{total_tests})")
        
        print(f"\n【号码推荐成功率】（基于生肖）")
        print(f"     号码 TOP 5:  {number_top5_rate:.2f}% ({number_top5_hits}/{total_tests})")
        print(f"     号码 TOP 10: {number_top10_rate:.2f}% ({number_top10_hits}/{total_tests})")
        print(f"     号码 TOP 15: {number_top15_rate:.2f}% ({number_top15_hits}/{total_tests})")
        
        # 分段统计
        print(f"\n{'='*80}")
        print("分段成功率分析（每25期）")
        print(f"{'='*80}\n")
        
        for segment in range(4):
            start = segment * 25
            end = start + 25
            segment_details = details[start:end] if end <= len(details) else details[start:]
            
            seg_zodiac_top5 = sum(1 for d in segment_details if '生肖TOP' in d['生肖命中'] or '生肖TOP' in d['生肖命中'])
            seg_number_top15 = sum(1 for d in segment_details if '号码TOP' in d['号码命中'])
            
            seg_len = len(segment_details)
            start_period = segment_details[0]['期数']
            end_period = segment_details[-1]['期数']
            
            print(f"第{segment+1}段（第{start_period}-{end_period}期）:")
            print(f"  生肖TOP5: {seg_zodiac_top5/seg_len*100:.1f}% ({seg_zodiac_top5}/{seg_len})")
            print(f"  号码TOP15: {seg_number_top15/seg_len*100:.1f}% ({seg_number_top15}/{seg_len})\n")
        
        # 保存结果
        result_file = 'zodiac_validation_100periods_results.csv'
        result_df = pd.DataFrame(details)
        result_df.to_csv(result_file, index=False, encoding='utf-8-sig')
        
        print(f"详细结果已保存至: {result_file}")
        
        # 结论
        print(f"\n{'='*80}")
        print("结论")
        print(f"{'='*80}\n")
        
        print(f"生肖预测评价:")
        if zodiac_top5_rate >= 50:
            print(f"  生肖TOP5成功率: {zodiac_top5_rate:.2f}% - ✅ 优秀")
        elif zodiac_top5_rate >= 40:
            print(f"  生肖TOP5成功率: {zodiac_top5_rate:.2f}% - ✓ 良好")
        else:
            print(f"  生肖TOP5成功率: {zodiac_top5_rate:.2f}% - ○ 一般")
        
        print(f"\n号码推荐评价:")
        if number_top15_rate >= 30:
            print(f"  号码TOP15成功率: {number_top15_rate:.2f}% - ✅ 优秀")
        elif number_top15_rate >= 20:
            print(f"  号码TOP15成功率: {number_top15_rate:.2f}% - ✓ 良好")
        else:
            print(f"  号码TOP15成功率: {number_top15_rate:.2f}% - ○ 一般")
        
        return {
            'total_tests': total_tests,
            'zodiac_top1_hits': zodiac_top1_hits,
            'zodiac_top3_hits': zodiac_top3_hits,
            'zodiac_top5_hits': zodiac_top5_hits,
            'zodiac_top5_rate': zodiac_top5_rate,
            'number_top15_hits': number_top15_hits,
            'number_top15_rate': number_top15_rate,
            'details': details
        }


def main():
    """主函数"""
    predictor = ZodiacPredictor()
    
    # 生成下一期预测
    print("=" * 80)
    print("生肖预测模型 - 下一期预测")
    print("=" * 80)
    
    result = predictor.predict()
    
    print(f"\n最新一期信息:")
    print(f"  期数: {result['total_periods']}")
    print(f"  日期: {result['last_date']}")
    print(f"  开出: {result['last_number']} ({result['last_zodiac']})")
    
    print(f"\n下一期预测（第{result['total_periods']+1}期）:")
    print(f"\n⭐ 推荐生肖 TOP 5:")
    for i, (zodiac, score) in enumerate(result['top5_zodiacs'], 1):
        nums = ', '.join(map(str, predictor.zodiac_numbers[zodiac]))
        print(f"  {i}. {zodiac} (评分: {score:.2f}) - 对应号码: {nums}")
    
    print(f"\n📋 推荐号码 TOP 15:")
    top5 = result['top15_numbers'][:5]
    top10 = result['top15_numbers'][5:10]
    top15 = result['top15_numbers'][10:15]
    print(f"  TOP 1-5:   {top5}")
    print(f"  TOP 6-10:  {top10}")
    print(f"  TOP 11-15: {top15}")
    
    print(f"\n{'='*80}")
    print("开始100期验证...")
    print(f"{'='*80}\n")
    
    # 验证模型
    predictor.validate_recent_100_periods()


if __name__ == '__main__':
    main()
