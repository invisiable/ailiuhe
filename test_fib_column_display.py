"""
快速测试GUI中Fib索引列的显示
"""

# 模拟暂停策略的历史数据
pause_history = [
    {'period': 1, 'date': '2025/5/4', 'actual': 10, 'predictions_str': '[3, 23, 20, 18, 1]...', 
     'multiplier': 1.0, 'bet': 15, 'hit': False, 'profit': -15, 'cumulative_profit': -15,
     'paused': False, 'pause_remaining': 0, 'fib_index': 1, 'result': 'LOSS'},
    
    {'period': 2, 'date': '2025/5/5', 'actual': 49, 'predictions_str': '[3, 38, 23, 20, 1]...', 
     'multiplier': 1.0, 'bet': 15, 'hit': False, 'profit': -15, 'cumulative_profit': -30,
     'paused': False, 'pause_remaining': 0, 'fib_index': 2, 'result': 'LOSS'},
    
    {'period': 3, 'date': '2025/5/6', 'actual': 2, 'predictions_str': '[3, 28, 38, 23, 18]...', 
     'multiplier': 2.0, 'bet': 30, 'hit': False, 'profit': -30, 'cumulative_profit': -60,
     'paused': False, 'pause_remaining': 0, 'fib_index': 3, 'result': 'LOSS'},
    
    {'period': 4, 'date': '2025/5/7', 'actual': 20, 'predictions_str': '[3, 6, 1, 20, 47]...', 
     'multiplier': 3.0, 'bet': 45, 'hit': True, 'profit': 96, 'cumulative_profit': 36,
     'paused': False, 'pause_remaining': 1, 'fib_index': 0, 'result': 'WIN'},
    
    {'period': 5, 'date': '2025/5/8', 'actual': 15, 'predictions_str': '[3, 15, 38, 23, 1]...', 
     'multiplier': 0, 'bet': 0, 'hit': True, 'profit': 0, 'cumulative_profit': 36,
     'paused': True, 'pause_remaining': 0, 'fib_index': 0, 'result': 'SKIP'},
    
    {'period': 6, 'date': '2025/5/9', 'actual': 7, 'predictions_str': '[3, 38, 7, 23, 1]...', 
     'multiplier': 1.0, 'bet': 15, 'hit': True, 'profit': 32, 'cumulative_profit': 68,
     'paused': False, 'pause_remaining': 1, 'fib_index': 0, 'result': 'WIN'},
     
    {'period': 7, 'date': '2025/5/10', 'actual': 30, 'predictions_str': '[3, 38, 23, 20, 1]...', 
     'multiplier': 0, 'bet': 0, 'hit': False, 'profit': 0, 'cumulative_profit': 68,
     'paused': True, 'pause_remaining': 0, 'fib_index': 0, 'result': 'SKIP'},
]

print("=" * 124)
print("GUI暂停策略详情表 - Fib索引列测试")
print("=" * 124)
print()

# 模拟GUI输出格式
print(f"{'期号':<8}{'日期':<12}{'开奖':<6}{'预测TOP15':<25}{'倍数':<8}{'投注':<8}{'命中':<6}{'盈亏':<10}{'累计':<12}{'暂停':<6}{'余停':<6}{'Fib':<4}")
print("-" * 124)

cumulative_pause = 0
for entry in pause_history:
    period = entry.get('period', 0)
    date = entry.get('date', '')
    actual = entry.get('actual', 'N/A')
    pred_str = entry.get('predictions_str') or str(entry.get('predictions', [])[:5]) + "..."
    multiplier = entry.get('multiplier', 0)
    bet_amount = entry.get('bet', 0)
    result = entry.get('result', 'SKIP')
    profit = entry.get('profit', 0)
    cumulative_pause += profit
    paused_flag = "暂停" if entry.get('paused') else ""
    pause_remaining = entry.get('pause_remaining', 0)
    fib_idx = entry.get('fib_index', 0)
    hit_mark = '✓' if entry.get('hit') else ('-' if result == 'SKIP' else '✗')
    
    print(
        f"{period:<8}{date:<12}{actual:<6}{pred_str:<25}"
        f"{multiplier:<8.2f}{bet_amount:<8.0f}{hit_mark:<6}"
        f"{profit:+10.0f}  {cumulative_pause:+12.0f}  {paused_flag:<6}{pause_remaining:<6}{fib_idx:<4}"
    )

print()
print("=" * 124)
print("说明：")
print("=" * 124)
print()
print("【Fib列说明】")
print("  • Fib = Fibonacci索引值（当前在Fibonacci序列中的位置）")
print("  • 索引0对应倍数1，索引2对应倍数2，索引4对应倍数5，依此类推")
print("  • 命中后自动重置为0（从最小倍数重新开始）")
print("  • 未中后索引+1（下期倍数增加）")
print("  • 暂停期间索引保持不变")
print()
print("【示例解读】")
print("  第1期：Fib=1（未中）→ 第2期：Fib=2（未中）→ 第3期：Fib=3")
print("  第4期：命中 → Fib重置为0，触发暂停")
print("  第5期：暂停期，Fib保持为0")
print("  第6期：恢复投注，Fib=0（倍数1），命中后重置为0")
print("  第7期：暂停期，Fib保持为0")
print()
print("✅ Fib索引列已成功添加到详情表中！")
print()
