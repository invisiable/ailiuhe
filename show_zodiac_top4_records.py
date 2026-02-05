"""
生肖TOP4动态投注详细记录查看器
方便查看和分析每期的投注情况
"""

import ast
import pandas as pd
import sys


def show_detailed_records(csv_file='zodiac_top4_dynamic_betting_100periods.csv', 
                         show_all=False, 
                         show_wins_only=False,
                         show_losses_only=False,
                         show_high_multiplier=False):
    """
    显示详细的投注记录
    
    Args:
        csv_file: CSV文件路径
        show_all: 显示所有期数
        show_wins_only: 仅显示命中的期数
        show_losses_only: 仅显示失败的期数
        show_high_multiplier: 仅显示高倍投注（>=4倍）
    """
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"文件不存在: {csv_file}")
        print("请先运行: python validate_zodiac_top4_dynamic_betting.py")
        return
    has_date = 'date' in df.columns
    has_global_period = 'global_period' in df.columns
    has_actual_animal = 'actual_animal' in df.columns
    
    hit_column = 'is_hit_animal' if 'is_hit_animal' in df.columns else 'is_hit'
    number_hit_column = 'is_hit_number' if 'is_hit_number' in df.columns else ('is_hit' if 'is_hit' in df.columns else None)

    # 应用过滤
    if show_wins_only:
        df = df[df[hit_column] == True]
        title = "命中记录"
    elif show_losses_only:
        df = df[df[hit_column] == False]
        title = "失败记录"
    elif show_high_multiplier:
        df = df[df['multiplier'] >= 4.0]
        title = "高倍投注记录（>=4倍）"
    else:
        title = "完整投注记录"
    
    print(f"\n{'='*100}")
    print(f"生肖TOP4动态投注 - {title}")
    print(f"{'='*100}\n")
    
    # 打印表头
    header = f"{'期数':>4} "
    if has_global_period:
        header += f"{'真实期':>6} "
    if has_date:
        header += f"{'日期':<12} "
    header += f"{'TOP4生肖':^20} {'投注数':>6} {'实际':>4} "
    if has_actual_animal:
        header += f"{'生肖':^4} "
    header += f"{'肖中':^4} "
    if number_hit_column:
        header += f"{'号中':^4} "
    header += f"{'倍数':>6} {'投注额':>8} {'盈亏':>8} {'累计':>10} {'连胜':>4} {'连败':>4}"
    print(header)
    print(f"{'-'*100}")
    
    # 打印每期记录
    for _, row in df.iterrows():
        period = int(row['period'])
        global_period = int(row['global_period']) if has_global_period else None
        date_str = str(row['date']) if has_date else ''
        top4 = ast.literal_eval(row['top4_zodiacs'])  # 转换字符串为列表
        top4_str = ''.join(top4)
        bet_count = int(row['bet_count'])
        actual = int(row['actual'])
        actual_animal = str(row['actual_animal']) if has_actual_animal else ''
        hit_animal = bool(row[hit_column])
        hit_number = bool(row[number_hit_column]) if number_hit_column else hit_animal
        result_icon_animal = "✅" if hit_animal else "❌"
        result_icon_number = "✅" if hit_number else "❌"
        multiplier = float(row['multiplier'])
        bet_amount = float(row['bet_amount'])
        profit = float(row['profit'])
        cumulative = float(row['cumulative_profit'])
        consecutive_wins = int(row['consecutive_wins'])
        consecutive_losses = int(row['consecutive_losses'])
        
        # 根据是否命中设置颜色（使用ANSI转义码）
        if hit_number:
            profit_str = f"+{profit:.1f}"
            cumulative_str = f"+{cumulative:.1f}"
        else:
            profit_str = f"{profit:.1f}"
            cumulative_str = f"{cumulative:+.1f}"
        
        row_output = f"{period:>4} "
        if has_global_period:
            row_output += f"{global_period:>6} "
        if has_date:
            row_output += f"{date_str:<12} "
        row_output += f"{top4_str:^20} {bet_count:>6} {actual:>4} "
        if has_actual_animal:
            row_output += f"{actual_animal:^4} "
        row_output += f"{result_icon_animal:^4} "
        if number_hit_column:
            row_output += f"{result_icon_number:^4} "
        row_output += (f"{multiplier:>5.1f}x {bet_amount:>7.1f}元 {profit_str:>7}元 "
                       f"{cumulative_str:>9}元 {consecutive_wins:>4} {consecutive_losses:>4}")
        print(row_output)
        
        # 如果不显示全部，只显示部分
        if not show_all and period >= 20:
            remaining = len(df) - period
            if remaining > 0 and period == 20:
                print(f"... (省略中间{remaining}期记录，使用 --all 参数查看全部)")
            if period > 20 and period < len(df) - 10:
                continue
    
    # 打印统计汇总
    print(f"\n{'-'*100}")
    total_periods = len(df)
    total_hits = df[hit_column].sum()
    hit_rate = total_hits / total_periods * 100
    number_hits = df[number_hit_column].sum() if number_hit_column else total_hits
    number_hit_rate = number_hits / total_periods * 100
    total_cost = df['bet_amount'].sum()
    total_profit = df['profit'].sum()
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
    
    print(f"\n📊 统计汇总:")
    print(f"   期数: {total_periods}期")
    print(f"   命中(生肖): {total_hits}次 ({hit_rate:.2f}%)")
    if number_hit_column:
        print(f"   命中(号码): {number_hits}次 ({number_hit_rate:.2f}%)")
    print(f"   投注: {total_cost:.2f}元")
    print(f"   盈亏: {total_profit:+.2f}元")
    print(f"   ROI: {roi:+.2f}%")
    
    # 倍数统计
    print(f"\n📈 倍数分布:")
    multiplier_counts = df['multiplier'].value_counts().sort_index()
    for mult, count in multiplier_counts.items():
        pct = count / total_periods * 100
        print(f"   {mult:.1f}倍: {count:>3}期 ({pct:>5.2f}%)")
    
    # 连胜连败统计
    max_wins = df['consecutive_wins'].max()
    max_losses = df['consecutive_losses'].max()
    print(f"\n🎯 连续记录:")
    print(f"   最大连胜: {max_wins}期")
    print(f"   最大连败: {max_losses}期")
    
    print(f"\n{'='*100}\n")


def show_profit_curve(csv_file='zodiac_top4_dynamic_betting_100periods.csv'):
    """显示累计盈亏曲线（文本版）"""
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"文件不存在: {csv_file}")
        return
    
    print(f"\n{'='*100}")
    print(f"累计盈亏曲线")
    print(f"{'='*100}\n")
    
    # 计算比例
    max_profit = df['cumulative_profit'].max()
    min_profit = df['cumulative_profit'].min()
    range_profit = max_profit - min_profit
    
    # 每10期显示一次
    step = max(1, len(df) // 20)
    
    for i in range(0, len(df), step):
        row = df.iloc[i]
        period = int(row['period'])
        cumulative = float(row['cumulative_profit'])
        
        # 计算柱状图长度（0-50个字符）
        if range_profit > 0:
            bar_length = int((cumulative - min_profit) / range_profit * 50)
        else:
            bar_length = 25
        
        bar = '█' * bar_length
        
        print(f"第{period:>3}期 {cumulative:>+8.1f}元 {bar}")
    
    print(f"\n{'='*100}\n")


def main():
    """主函数"""
    # 解析命令行参数
    args = sys.argv[1:]
    
    show_all = '--all' in args
    show_wins = '--wins' in args
    show_losses = '--losses' in args
    show_high = '--high' in args
    show_curve = '--curve' in args
    
    if '--help' in args or '-h' in args:
        print("""
生肖TOP4动态投注详细记录查看器

用法:
    python show_zodiac_top4_records.py [选项]

选项:
    --all       显示所有100期记录（默认显示前20期和后10期）
    --wins      仅显示命中的期数
    --losses    仅显示失败的期数
    --high      仅显示高倍投注（>=4倍）的期数
    --curve     显示累计盈亏曲线
    --help, -h  显示此帮助信息

示例:
    python show_zodiac_top4_records.py
    python show_zodiac_top4_records.py --all
    python show_zodiac_top4_records.py --wins
    python show_zodiac_top4_records.py --high
    python show_zodiac_top4_records.py --curve
        """)
        return
    
    # 显示详细记录
    if show_curve:
        show_profit_curve()
    else:
        show_detailed_records(
            show_all=show_all,
            show_wins_only=show_wins,
            show_losses_only=show_losses,
            show_high_multiplier=show_high
        )


if __name__ == '__main__':
    main()
