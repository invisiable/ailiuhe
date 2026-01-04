"""
生肖预测 v12.0 - 平衡激进性与稳定性
继承v10.0的稳定基础，添加精准爆发检测
目标：最近50期成功率>50%
"""

import pandas as pd
from collections import Counter
from zodiac_simple_smart import ZodiacSimpleSmart

class ZodiacBalancedSmart(ZodiacSimpleSmart):
    """v12.0: 平衡版本 - 稳定基础 + 精准爆发捕捉"""
    
    def __init__(self):
        super().__init__()
        self.version = "v12.0 平衡智能选择器"
        # 从父类获取生肖列表
        self.all_zodiacs = self.zodiacs
        
    def predict_top5(self, recent_periods=100):
        """生成TOP5预测，平衡稳定性与爆发捕捉"""
        df = pd.read_csv('data/lucky_numbers.csv')
        df = df.tail(recent_periods)
        animals = df['animal'].tolist()
        
        # 1. 获取v10.0基础预测
        base_result = self.predict_from_history(animals, top_n=5, debug=False)
        base_scores = {zodiac: 0 for zodiac in self.all_zodiacs}
        # 归一化分数到0-100范围
        max_score = max(score for _, score in base_result['all_scores'])
        min_score = min(score for _, score in base_result['all_scores'])
        score_range = max_score - min_score if max_score > min_score else 1
        
        for zodiac, score in base_result['all_scores']:
            normalized = ((score - min_score) / score_range) * 100
            base_scores[zodiac] = normalized
        
        # 2. 检测是否有爆发
        burst_info = self._detect_precise_burst(df)
        
        if burst_info['has_burst']:
            # 检测到爆发：在v10.0基础上增强爆发生肖
            print(f"【爆发增强】检测到爆发生肖: {burst_info['burst_zodiacs']}")
            for z, info in burst_info['details'].items():
                print(f"  {z}: 前30期{info['prev_count']}次 → 最近10期{info['recent_count']}次 (强度+{info['strength']})")
            
            # 爆发生肖加成
            for zodiac in burst_info['burst_zodiacs']:
                strength = burst_info['details'][zodiac]['strength']
                # 根据强度给予不同加分
                if strength >= 2:
                    base_scores[zodiac] += 60  # 强爆发
                else:
                    base_scores[zodiac] += 45  # 普通爆发
            
            # 重新排序
            top5 = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            return [zodiac for zodiac, _ in top5]
        else:
            # 无爆发：直接使用v10.0预测
            return base_result['top5']
    
    def _detect_precise_burst(self, df):
        """精准爆发检测 - 平衡的条件避免误判"""
        recent_10 = df.tail(10)
        prev_30 = df.tail(40).head(30)  # 改用前30期对比
        
        recent_counter = Counter()
        prev_counter = Counter()
        
        for _, row in recent_10.iterrows():
            recent_counter[row['animal']] += 1
        
        for _, row in prev_30.iterrows():
            prev_counter[row['animal']] += 1
        
        burst_zodiacs = []
        details = {}
        
        for zodiac in self.all_zodiacs:
            prev_count = prev_counter.get(zodiac, 0)
            recent_count = recent_counter.get(zodiac, 0)
            
            # 调整爆发条件：
            # 1. 前30期出现≤1次（非常冷门）
            # 2. 最近10期出现≥2次（明确上升）
            # 3. 增幅≥1（显著增长）
            if prev_count <= 1 and recent_count >= 2:
                burst_zodiacs.append(zodiac)
                details[zodiac] = {
                    'prev_count': prev_count,
                    'recent_count': recent_count,
                    'strength': recent_count - prev_count
                }
        
        return {
            'has_burst': len(burst_zodiacs) > 0,
            'burst_zodiacs': burst_zodiacs,
            'details': details,
            'recent_counter': recent_counter,
            'prev_counter': prev_counter
        }
    
    def _balanced_burst_strategy(self, df, burst_info):
        """平衡的爆发策略 - 融合v10.0稳定基础"""
        scores = {zodiac: 0 for zodiac in self.all_zodiacs}
        
        recent_10 = df.tail(10)
        recent_20 = df.tail(20)
        recent_50 = df.tail(50)
        
        recent_counter = burst_info['recent_counter']
        
        # 1. 爆发加成（50分基础 + 每次出现25分）- 更温和
        for zodiac in burst_info['burst_zodiacs']:
            count = recent_counter.get(zodiac, 0)
            scores[zodiac] += 50 + count * 25
        
        # 2. 最近10期热门（权重40%）- 融合v10.0逻辑
        for zodiac, count in recent_counter.items():
            if count >= 2:  # 出现2+次
                scores[zodiac] += count * 18
        
        # 3. 最近20期稳定性（权重30%）
        counter_20 = Counter()
        for _, row in recent_20.iterrows():
            counter_20[row['animal']] += 1
        
        for zodiac, count in counter_20.items():
            if count >= 2:  # 降低门槛
                scores[zodiac] += count * 10
        
        # 4. 最近50期基础分（权重20%）- 保持长期稳定性
        counter_50 = Counter()
        for _, row in recent_50.iterrows():
            counter_50[row['animal']] += 1
        
        for zodiac, count in counter_50.items():
            if count >= 4:  # 降低门槛
                scores[zodiac] += count * 4
        
        # 5. 多样性保护（权重10%）- 避免过度集中
        current_top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_zodiacs = [z for z, _ in current_top]
        
        for zodiac in self.all_zodiacs:
            if zodiac not in top_zodiacs:
                # 给非热门一些机会
                if counter_50.get(zodiac, 0) >= 3:
                    scores[zodiac] += 12
        
        return scores
    
    def get_strategy_info(self, recent_periods=100):
        """返回当前策略信息"""
        df = pd.read_csv('data/lucky_numbers.csv')
        df = df.tail(recent_periods)
        
        burst_info = self._detect_precise_burst(df)
        
        if burst_info['has_burst']:
            return {
                'version': self.version,
                'scenario': 'burst_enhanced',
                'strategy': '爆发增强模式',
                'burst_zodiacs': burst_info['burst_zodiacs'],
                'details': burst_info['details']
            }
        else:
            # 无爆发时调用父类场景检测
            animals = df['animal'].tolist()
            scenario = self._detect_scenario(animals)
            model_map = {
                'normal': 'v5.0平衡模型',
                'extreme_hot': '热门感知模型',
                'extreme_cold': '极致冷门模型'
            }
            return {
                'version': self.version,
                'scenario': scenario,
                'strategy': model_map.get(scenario, 'v5.0平衡模型')
            }


if __name__ == '__main__':
    print("="*100)
    print("生肖预测 v12.0 - 平衡激进性与稳定性")
    print("="*100)
    
    predictor = ZodiacBalancedSmart()
    
    # 获取当前策略信息
    info = predictor.get_strategy_info()
    print(f"\n当前版本: {info['version']}")
    print(f"识别场景: {info['scenario']}")
    print(f"使用策略: {info['strategy']}")
    
    if info['scenario'] == 'burst_detected':
        print(f"\n检测到爆发生肖: {info['burst_zodiacs']}")
        for zodiac, detail in info['details'].items():
            print(f"  {zodiac}: 前40期{detail['prev_count']}次 → 最近10期{detail['recent_count']}次 (强度+{detail['strength']})")
    
    # 生成预测
    top5 = predictor.predict_top5()
    print(f"\n预测下一期TOP5生肖: {', '.join(top5)}")
    print("\n" + "="*100)
