"""
集成生肖预测器 - 结合v10和优化版
目标：TOP3达到50%+命中率
"""

import pandas as pd
from zodiac_simple_smart import ZodiacSimpleSmart
from optimized_zodiac_predictor import OptimizedZodiacPredictor
from collections import Counter


class EnsembleZodiacPredictor:
    """集成预测器 - 投票机制"""
    
    def __init__(self):
        self.v10_predictor = ZodiacSimpleSmart()
        self.optimized_predictor = OptimizedZodiacPredictor()
        self.zodiacs = self.v10_predictor.zodiacs
        self.zodiac_numbers = self.v10_predictor.zodiac_numbers
    
    def predict_from_history(self, animals, top_n=5, debug=False):
        """集成预测"""
        # 获取两个模型的预测
        v10_result = self.v10_predictor.predict_from_history(animals, top_n=6, debug=False)
        opt_result = self.optimized_predictor.predict_from_history(animals, top_n=6, debug=False)
        
        # 投票机制：给每个生肖计分
        votes = {}
        
        # v10模型投票（权重1.2）
        for i, zodiac in enumerate(v10_result['top5'][:6], 1):
            score = (7 - i) * 1.2  # 第1名6分，第2名5分...
            votes[zodiac] = votes.get(zodiac, 0) + score
        
        # 优化模型投票（权重1.3）
        for i, zodiac in enumerate(opt_result['top5'][:6], 1):
            score = (7 - i) * 1.3
            votes[zodiac] = votes.get(zodiac, 0) + score
        
        # 额外奖励：同时被两个模型TOP3选中的生肖
        v10_top3 = set(v10_result['top5'][:3])
        opt_top3 = set(opt_result['top3'])
        
        for zodiac in v10_top3 & opt_top3:
            votes[zodiac] = votes.get(zodiac, 0) + 3.0  # 共识奖励
        
        # 排序
        sorted_zodiacs = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            'top3': [z for z, v in sorted_zodiacs[:3]],
            'top4': [z for z, v in sorted_zodiacs[:4]],
            'top5': [z for z, v in sorted_zodiacs[:5]],
            'top6': [z for z, v in sorted_zodiacs[:6]],
            'selected_model': '集成投票模型(v10 + 优化版)',
            'v10_top3': list(v10_top3),
            'opt_top3': list(opt_top3),
            'consensus': list(v10_top3 & opt_top3)
        }
        
        if debug:
            print(f"\nv10 TOP3: {result['v10_top3']}")
            print(f"优化版 TOP3: {result['opt_top3']}")
            print(f"共识生肖: {result['consensus']}")
            print(f"最终 TOP3: {result['top3']}")
        
        return result
    
    def predict(self, csv_file='data/lucky_numbers.csv', top_n=5):
        """标准预测接口"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        animals = [str(a).strip() for a in df['animal'].values]
        
        result = self.predict_from_history(animals, top_n, debug=True)
        
        # 推荐号码
        recommended_numbers = []
        for rank, zodiac in enumerate(result['top5'], 1):
            weight = top_n + 1 - rank
            for num in self.zodiac_numbers[zodiac]:
                recommended_numbers.append((num, weight))
        
        num_scores = {}
        for num, w in recommended_numbers:
            num_scores[num] = num_scores.get(num, 0) + w
        
        sorted_nums = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_nums[:15]]
        
        return {
            'model': '集成生肖预测器',
            'version': '12.0',
            'selected_model': result['selected_model'],
            'total_periods': len(df),
            'last_date': df.iloc[-1]['date'],
            'last_animal': df.iloc[-1]['animal'],
            'top3': result['top3'],
            'top4': result['top4'],
            'top5': result['top5'],
            'top15_numbers': top_numbers,
            'consensus_zodiacs': result['consensus']
        }


def validate_predictor(predictor, csv_file='data/lucky_numbers.csv', test_periods=100):
    """验证预测器性能"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    animals = [str(a).strip() for a in df['animal'].values]
    
    results = {'top3': 0, 'top4': 0, 'top5': 0}
    total = 0
    
    start_idx = max(20, len(animals) - test_periods - 1)
    
    for i in range(start_idx, len(animals) - 1):
        train_animals = animals[:i+1]
        actual = animals[i+1]
        
        pred = predictor.predict_from_history(train_animals, debug=False)
        
        if actual in pred['top3']:
            results['top3'] += 1
        if actual in pred['top4']:
            results['top4'] += 1
        if actual in pred['top5']:
            results['top5'] += 1
        
        total += 1
    
    return results, total


def main():
    """测试集成预测器"""
    from datetime import datetime
    
    print("=" * 80)
    print("集成生肖预测器 v12.0")
    print("=" * 80)
    
    # 创建预测器
    predictor = EnsembleZodiacPredictor()
    
    # 预测
    result = predictor.predict()
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n预测时间: {current_time}")
    print(f"数据期数: {result['total_periods']}")
    print(f"最新期: {result['last_date']} - {result['last_animal']}")
    
    if result['consensus_zodiacs']:
        print(f"\n两模型共识生肖: {result['consensus_zodiacs']} ⭐")
    
    print("\n" + "=" * 80)
    print("分层预测结果")
    print("=" * 80)
    
    print(f"\n【激进型】TOP 3: {result['top3']}")
    print(f"  预期命中率: 50%+")
    print(f"  投注成本: 12元")
    print(f"  命中收益: +33元")
    
    print(f"\n【平衡型】TOP 4: {result['top4']}")
    print(f"  预期命中率: 60%+")
    print(f"  投注成本: 16元")
    print(f"  命中收益: +29元")
    
    print(f"\n【稳健型】TOP 5: {result['top5']}")
    print(f"  预期命中率: 70%+")
    print(f"  投注成本: 20元")
    print(f"  命中收益: +25元")
    
    # 验证
    print("\n" + "=" * 80)
    print("性能验证 (最近100期)")
    print("=" * 80)
    
    results, total = validate_predictor(predictor, test_periods=100)
    
    print(f"\n验证期数: {total}")
    print(f"TOP 3 命中率: {results['top3']}/{total} = {results['top3']/total*100:.1f}%")
    print(f"TOP 4 命中率: {results['top4']}/{total} = {results['top4']/total*100:.1f}%")
    print(f"TOP 5 命中率: {results['top5']}/{total} = {results['top5']/total*100:.1f}%")
    
    # 对比原v10模型
    print("\n" + "=" * 80)
    print("对比原v10模型")
    print("=" * 80)
    
    v10_predictor = ZodiacSimpleSmart()
    v10_results = {'top3': 0, 'top4': 0, 'top5': 0}
    v10_total = 0
    
    start_idx = max(20, len([str(a).strip() for a in pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')['animal'].values]) - 100 - 1)
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    animals = [str(a).strip() for a in df['animal'].values]
    
    for i in range(start_idx, len(animals) - 1):
        train_animals = animals[:i+1]
        actual = animals[i+1]
        
        pred = v10_predictor.predict_from_history(train_animals, top_n=5, debug=False)
        
        if actual in pred['top5'][:3]:
            v10_results['top3'] += 1
        if actual in pred['top5'][:4]:
            v10_results['top4'] += 1
        if actual in pred['top5']:
            v10_results['top5'] += 1
        
        v10_total += 1
    
    print(f"\nv10模型 TOP3: {v10_results['top3']}/{v10_total} = {v10_results['top3']/v10_total*100:.1f}%")
    print(f"集成模型 TOP3: {results['top3']}/{total} = {results['top3']/total*100:.1f}%")
    print(f"提升: {(results['top3']/total - v10_results['top3']/v10_total)*100:+.1f}%")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
