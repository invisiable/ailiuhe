"""
验证GUI中的最优智能投注策略实现
对比generate_smart_dynamic_300periods_detail.py的结果
"""

import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor

# 最优策略配置（与GUI完全一致 - v3.1激进组合）
config = {
    'name': '最优智能动态倍投策略 v3.1',
    'lookback': 12,
    'good_thresh': 0.35,
    'bad_thresh': 0.20,
    'boost_mult': 1.5,  # v3.1激进组合
    'reduce_mult': 0.5,  # v3.1激进组合
    'max_multiplier': 10,
    'base_bet': 15,
    'win_reward': 47
}

# Fibonacci数列
fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# 智能动态倍投策略类（从GUI复制）
class SmartDynamicStrategy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fib_index = 0
        self.recent_results = []
        self.total_bet = 0
        self.total_win = 0
        self.balance = 0
        self.min_balance = 0
        self.max_drawdown = 0
    
    def get_base_multiplier(self):
        if self.fib_index >= len(fib_sequence):
            return min(fib_sequence[-1], self.cfg['max_multiplier'])
        return min(fib_sequence[self.fib_index], self.cfg['max_multiplier'])
    
    def get_recent_rate(self):
        if len(self.recent_results) == 0:
            return 0.33
        return sum(self.recent_results) / len(self.recent_results)
    
    def process_period(self, hit):
        # 获取基础倍数
        base_mult = self.get_base_multiplier()
        
        # 根据最近命中率计算动态倍数（使用投注前的历史数据）
        if len(self.recent_results) >= self.cfg['lookback']:
            rate = self.get_recent_rate()
            if rate >= self.cfg['good_thresh']:
                multiplier = min(base_mult * self.cfg['boost_mult'], self.cfg['max_multiplier'])
            elif rate <= self.cfg['bad_thresh']:
                multiplier = max(base_mult * self.cfg['reduce_mult'], 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        # 计算投注和收益
        bet = self.cfg['base_bet'] * multiplier
        self.total_bet += bet
        
        if hit:
            win = self.cfg['win_reward'] * multiplier
            self.total_win += win
            profit = win - bet
            self.balance += profit
            self.fib_index = 0
        else:
            profit = -bet
            self.balance += profit
            self.fib_index += 1
            
            if self.balance < self.min_balance:
                self.min_balance = self.balance
                self.max_drawdown = abs(self.min_balance)
        
        # 添加结果到历史（在投注和结算之后）
        self.recent_results.append(1 if hit else 0)
        if len(self.recent_results) > self.cfg['lookback']:
            self.recent_results.pop(0)
        
        return {
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'recent_rate': self.get_recent_rate()
        }

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f"数据总期数: {len(df)}")

# 300期回测
test_periods = min(300, len(df) - 50)
start_idx = len(df) - test_periods
print(f"回测期数: {test_periods}")
print(f"回测范围: 第{start_idx+1}期 到 第{len(df)}期")

# 初始化预测器
predictor = PreciseTop15Predictor()

# 初始化策略
strategy = SmartDynamicStrategy(config)

# 回测统计
results = []
hit_10x_count = 0

for i in range(start_idx, len(df)):
    period_num = i - start_idx + 1
    
    # 预测
    train_data = df.iloc[:i]['number'].values
    predictions = predictor.predict(train_data)
    actual = df.iloc[i]['number']
    date = df.iloc[i]['date']
    
    # 判断命中
    hit = actual in predictions
    
    # 更新预测器性能跟踪
    predictor.update_performance(predictions, actual)
    
    # 处理这一期
    result = strategy.process_period(hit)
    
    # 记录
    hit_limit = result['multiplier'] >= 10
    if hit_limit:
        hit_10x_count += 1
    
    results.append({
        'period': period_num,
        'date': date,
        'actual': actual,
        'hit': hit,
        'multiplier': result['multiplier'],
        'bet': result['bet'],
        'profit': result['profit'],
        'cumulative_profit': strategy.balance,
        'hit_limit': hit_limit,
        'fib_index': strategy.fib_index
    })

# 统计结果
total_cost = strategy.total_bet
total_profit = strategy.balance
roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
hits = sum(1 for r in results if r['hit'])
hit_rate = hits / len(results) if results else 0

print(f"\n{'='*80}")
print(f"【GUI策略验证结果】")
print(f"{'='*80}")
print(f"\n策略参数:")
print(f"  基础投注: {config['base_bet']}元")
print(f"  中奖奖励: {config['win_reward']}元")
print(f"  回看期数: {config['lookback']}期")
print(f"  增强阈值: 命中率>={config['good_thresh']:.0%} → 倍数×{config['boost_mult']}")
print(f"  降低阈值: 命中率<={config['bad_thresh']:.0%} → 倍数×{config['reduce_mult']}")
print(f"  最大倍数: {config['max_multiplier']}倍")

print(f"\n投注结果:")
print(f"  命中次数: {hits}/{len(results)}")
print(f"  命中率: {hit_rate*100:.2f}%")
print(f"  总投入: {total_cost:.0f}元")
print(f"  总收益: {strategy.total_win:.0f}元")
print(f"  净利润: {total_profit:+.0f}元")
print(f"  ROI: {roi:.2f}%")
print(f"  最大回撤: {strategy.max_drawdown:.0f}元")
print(f"  触及10倍上限次数: {hit_10x_count}次")

print(f"\n{'='*80}")
print(f"【对比预期结果】")
print(f"{'='*80}")
expected = {
    'hits': 102,  # v3.1验证值
    'hit_rate': 34.00,
    'total_bet': 10822,  # v3.1激进组合
    'total_win': 12290,
    'profit': 1468,
    'roi': 13.56,
    'drawdown': 692,
    'hit_10x': 7
}

print(f"\n指标对比:")
print(f"  命中次数: {hits} vs {expected['hits']} {'✓' if hits == expected['hits'] else '✗'}")
print(f"  命中率: {hit_rate*100:.2f}% vs {expected['hit_rate']:.2f}% {'✓' if abs(hit_rate*100 - expected['hit_rate']) < 0.01 else '✗'}")
print(f"  总投入: {total_cost:.0f} vs {expected['total_bet']} {'✓' if total_cost == expected['total_bet'] else '✗'}")
print(f"  总收益: {strategy.total_win:.0f} vs {expected['total_win']} {'✓' if abs(strategy.total_win - expected['total_win']) < 1 else '✗'}")
print(f"  净利润: {total_profit:.0f} vs {expected['profit']} {'✓' if abs(total_profit - expected['profit']) < 1 else '✗'}")
print(f"  ROI: {roi:.2f}% vs {expected['roi']:.2f}% {'✓' if abs(roi - expected['roi']) < 0.01 else '✗'}")
print(f"  最大回撤: {strategy.max_drawdown:.0f} vs {expected['drawdown']} {'✓' if abs(strategy.max_drawdown - expected['drawdown']) < 1 else '✗'}")
print(f"  触及10x: {hit_10x_count} vs {expected['hit_10x']} {'✓' if hit_10x_count == expected['hit_10x'] else '✗'}")

if all([
    hits == expected['hits'],
    abs(total_cost - expected['total_bet']) < 1,
    abs(strategy.total_win - expected['total_win']) < 1,
    abs(total_profit - expected['profit']) < 1,
    abs(roi - expected['roi']) < 0.01,
    abs(strategy.max_drawdown - expected['drawdown']) < 1,
    hit_10x_count == expected['hit_10x']
]):
    print(f"\n✅ 验证通过！GUI实现与预期结果完全一致！")
else:
    print(f"\n❌ 验证失败！存在差异，请检查实现。")

# 显示触及10x的期数详情
print(f"\n{'='*80}")
print(f"触及10倍上限的{hit_10x_count}期详情:")
print(f"{'='*80}")
for r in results:
    if r['hit_limit']:
        print(f"第{r['period']}期 ({r['date']}): 开奖{r['actual']}, 倍数{r['multiplier']:.2f}, "
              f"{'命中' if r['hit'] else '未中'}, 盈亏{r['profit']:+.0f}元, 累计{r['cumulative_profit']:+.0f}元")
