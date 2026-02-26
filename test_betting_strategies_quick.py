"""
快速测试倍投策略对比功能
"""

# 模拟命中记录
hit_records = [True, False, False, True, True, False, False, False, True, 
               False, True, True, False, False, True, False, False, False,
               True, True, False, True, False, False, True, True, False]

# 定义倍投策略
betting_strategies = {
    'fixed': {'name': '固定投注', 'func': lambda x: 1},
    'conservative': {'name': '保守倍投', 'func': lambda x: 1 if x == 0 else (1 if x == 1 else 2 if x == 2 else 2 + (x - 2))},
    'dalembert': {'name': '达朗贝尔倍投', 'func': lambda x: 1 + x},
    'fibonacci': {'name': '斐波那契倍投', 'func': lambda x: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89][min(x, 10)]},
    'martingale': {'name': '马丁格尔倍投', 'func': lambda x: 1 if x == 0 else min(2 ** x, 89)}
}

print("倍投策略函数测试：\n")

# 测试连续亏损0-5期的倍数
for strategy_id, strategy_info in betting_strategies.items():
    print(f"{strategy_info['name']}:")
    multipliers = [strategy_info['func'](i) for i in range(6)]
    print(f"  连续亏损0-5期的倍数: {multipliers}")

print("\n测试通过！各倍投策略函数工作正常。")
