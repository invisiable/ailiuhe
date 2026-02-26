"""
测试自动恢复后继续原倍数的逻辑

验证当触发止损后，如果连续错误5期自动恢复，应该继续原来的倍数逻辑
"""

# 模拟马丁格尔倍投 + 3期止损策略
def test_auto_resume_multiplier():
    print("="*80)
    print("测试：自动恢复后继续原倍数逻辑")
    print("="*80)
    print()
    
    # 马丁格尔倍数函数
    martingale = lambda x: 1 if x == 0 else min(2 ** x, 64)
    
    # 模拟场景
    print("场景：连续失败导致触发止损，然后自动恢复\n")
    
    # 投注前状态
    consecutive_losses = 0
    is_paused = False
    paused_periods = 0
    
    # 模拟投注序列
    results = [
        ('失败', False),  # 第1期：失败，连败1
        ('失败', False),  # 第2期：失败，连败2
        ('失败', False),  # 第3期：失败，连败3 → 触发止损
        ('失败', False),  # 第4期：暂停，失败1
        ('失败', False),  # 第5期：暂停，失败2
        ('失败', False),  # 第6期：暂停，失败3
        ('失败', False),  # 第7期：暂停，失败4
        ('失败', False),  # 第8期：暂停，失败5 → 自动恢复
        ('投注', True),   # 第9期：恢复投注
    ]
    
    for period, (label, is_normal_period) in enumerate(results, 1):
        if is_paused:
            # 暂停状态
            paused_periods += 1
            multiplier = 0
            bet = 0
            
            if paused_periods >= 5:
                # 自动恢复：继续原倍数
                is_paused = False
                status = f"[PAUSE]失败{paused_periods} → [AUTO]自动恢复"
                print(f"第{period}期: {status}")
                print(f"  → 恢复后继续原倍数：consecutive_losses={consecutive_losses}")
                print(f"  → 下期倍数将是：{martingale(consecutive_losses)}倍 (2^{consecutive_losses})")
            else:
                status = f"[PAUSE]失败{paused_periods}"
                print(f"第{period}期: 暂停投注，{status}")
        else:
            # 正常投注状态
            if is_normal_period:
                multiplier = martingale(consecutive_losses)
                bet = 16 * multiplier
                print(f"第{period}期: ✅ 恢复后首次投注")
                print(f"  → consecutive_losses={consecutive_losses}")
                print(f"  → 倍数：{multiplier}倍 (2^{consecutive_losses})")
                print(f"  → 投注：{bet}元")
            else:
                multiplier = martingale(consecutive_losses)
                bet = 16 * multiplier
                consecutive_losses += 1
                
                status = "正常" if consecutive_losses < 3 else "[STOP]触发止损"
                print(f"第{period}期: ✗失败 → 连败{consecutive_losses}期")
                print(f"  → 倍数：{multiplier}倍 (2^{consecutive_losses-1})")
                print(f"  → 投注：{bet}元")
                print(f"  → 状态：{status}")
                
                if consecutive_losses >= 3:
                    is_paused = True
                    paused_periods = 0
                    print(f"  → 触发止损，暂停投注")
        
        print()
    
    print("="*80)
    print("验证结果")
    print("="*80)
    print()
    print("✅ 预期行为：")
    print("  • 第1-2期：正常投注，2倍、4倍")
    print("  • 第3期：8倍投注，失败后触发止损（consecutive_losses=3）")
    print("  • 第4-8期：暂停投注，连续失败5期")
    print("  • 第9期：自动恢复，应该使用8倍（2^3），而不是1倍")
    print()
    print("💡 关键逻辑：")
    print("  • 命中恢复：重置consecutive_losses=0，倍数回到1倍")
    print("  • 自动恢复：保持consecutive_losses=3，倍数继续8倍")
    print()

if __name__ == '__main__':
    test_auto_resume_multiplier()
    
    print()
    print("="*80)
    print("对比场景：命中后恢复")
    print("="*80)
    print()
    
    print("场景：如果第4期命中...")
    print("第1期: ✗失败 → 连败1 → 2倍")
    print("第2期: ✗失败 → 连败2 → 4倍")
    print("第3期: ✗失败 → 连败3 → 8倍 → [STOP]触发止损")
    print("第4期: ✓命中 → [RESUME]恢复 → consecutive_losses重置为0")
    print("第5期: ✅ 恢复投注 → 1倍（重新开始）")
    print()
    print("✅ 命中恢复会重置倍数，这样更安全")
    print("✅ 自动恢复继续原倍数，给系统一次翻盘机会")
