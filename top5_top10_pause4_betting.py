"""Fixed-stake TOP5+TOP10 reduced-15 betting with a four-miss pause."""

import argparse
import math
from pathlib import Path


def simulate_pause4_betting(rows, test_periods=300, stake_per_number=1):
    """Start active at the window boundary; resume after an observed hit."""
    unit = float(stake_per_number)
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError('每个号码投注额必须为有限正数')
    if not isinstance(test_periods, int) or test_periods <= 0 or len(rows) < test_periods:
        raise ValueError('回测期数必须为正整数，且历史预测数量不能少于回测期数')

    active = True
    miss_streak = 0
    total_stake = 0.0
    total_payout = 0.0
    total_bets = 0
    wins = 0
    missed_hits = 0
    pause_events = 0
    balance = 0.0
    peak = 0.0
    minimum_balance = 0.0
    max_drawdown = 0.0
    baseline_balance = 0.0
    baseline_peak = 0.0
    baseline_drawdown = 0.0
    details = []

    for sequence, source in enumerate(rows[-test_periods:], start=1):
        prediction = list(source['prediction'])
        if len(prediction) != 15 or len(set(prediction)) != 15 or not all(
            1 <= number <= 49 and int(number) == number for number in prediction
        ):
            raise ValueError('精简15每期必须包含15个不重复且在1到49之间的整数号码')

        bet = active
        before = miss_streak
        hit = source['actual_number'] in prediction
        stake = 15 * unit if bet else 0.0
        payout = 47 * unit if bet and hit else 0.0
        profit = payout - stake
        total_stake += stake
        total_payout += payout
        total_bets += int(bet)
        wins += int(bet and hit)
        missed_hits += int(not bet and hit)
        balance += profit
        peak = max(peak, balance)
        minimum_balance = min(minimum_balance, balance)
        drawdown = peak - balance
        max_drawdown = max(max_drawdown, drawdown)

        miss_streak = 0 if hit else miss_streak + 1
        if hit:
            active = True
        elif bet and miss_streak >= 4:
            active = False
            pause_events += 1

        baseline_balance += (47 if hit else 0) * unit - 15 * unit
        baseline_peak = max(baseline_peak, baseline_balance)
        baseline_drawdown = max(baseline_drawdown, baseline_peak - baseline_balance)
        details.append({
            'period': sequence,
            'date': str(source['date']),
            'prediction': prediction,
            'actual_number': source['actual_number'],
            'actual_animal': source['actual_animal'],
            'bet': bet,
            'hit': hit,
            'stake': stake,
            'payout': payout,
            'profit': profit,
            'cumulative_profit': balance,
            'miss_streak_before': before,
            'miss_streak_after': miss_streak,
            'next_bet': active,
            'drawdown': drawdown,
        })

    observed_hits = wins + missed_hits
    baseline_stake = test_periods * 15 * unit
    return {
        'periods': test_periods,
        'stake_per_number': unit,
        'rows': details,
        'total_bets': total_bets,
        'pause_periods': test_periods - total_bets,
        'pause_events': pause_events,
        'wins': wins,
        'losses': total_bets - wins,
        'missed_hits': missed_hits,
        'observed_hits': observed_hits,
        'total_stake': total_stake,
        'total_payout': total_payout,
        'net_profit': balance,
        'roi': balance / total_stake if total_stake else 0.0,
        'max_drawdown': max_drawdown,
        'minimum_balance': minimum_balance,
        'next_bet': active,
        'current_miss_streak': miss_streak,
        'baseline': {
            'total_stake': baseline_stake,
            'total_payout': observed_hits * 47 * unit,
            'net_profit': baseline_balance,
            'roi': baseline_balance / baseline_stake,
            'max_drawdown': baseline_drawdown,
        },
    }


def format_pause4_betting_report(report, include_details=True):
    """Format the same summary and settled rows for GUI and export."""
    unit = report['stake_per_number']
    rows = report['rows']
    baseline = report['baseline']
    lines = [
        f"TOP5+TOP10精简15｜最近{report['periods']}期｜连续4期不中后暂停",
        f"区间：{rows[0]['date']} 至 {rows[-1]['date']}",
        f"固定1倍，每号{unit:g}元，每期{15 * unit:g}元；命中返还{47 * unit:g}元，净收益{32 * unit:g}元。",
        '第4次未中仍投注，随后暂停；暂停继续观察，出现命中后下一期恢复，观察命中不计奖金。',
        '首期正常投注，连续未中计数与累计收益从0开始；预测使用当期之前的历史数据。',
        f"实际投注{report['total_bets']}期，暂停{report['pause_periods']}期，触发暂停{report['pause_events']}次。",
        f"实际中奖{report['wins']}期，投注未中{report['losses']}期，暂停观察命中{report['missed_hits']}期。",
        f"总投入：{report['total_stake']:.2f}元｜总返还：{report['total_payout']:.2f}元",
        f"净收益：{report['net_profit']:+.2f}元｜ROI：{report['roi'] * 100:+.2f}%",
        f"最大回撤：{report['max_drawdown']:.2f}元｜最低累计收益：{report['minimum_balance']:+.2f}元",
        f"不暂停对照：投入{baseline['total_stake']:.2f}元，返还{baseline['total_payout']:.2f}元，"
        f"净收益{baseline['net_profit']:+.2f}元，最大回撤{baseline['max_drawdown']:.2f}元。",
        f"与不暂停相比净收益差额：{report['net_profit'] - baseline['net_profit']:+.2f}元",
        f"期末连续未中{report['current_miss_streak']}期，下一期状态：{'投注' if report['next_bet'] else '暂停'}。",
        '历史回测仅作核算，不代表未来收益。',
    ]
    if include_details:
        lines.extend(['', '逐期投注情况：'])
        for row in rows:
            result = ('中奖' if row['hit'] else '未中') if row['bet'] else (
                '观察命中，不领奖' if row['hit'] else '观察未中'
            )
            lines.append(
                f"{row['period']:03d} {row['date']}｜{'投注' if row['bet'] else '暂停'}｜"
                f"预测={row['prediction']}｜实际={row['actual_number']}({row['actual_animal']})｜"
                f"{result}｜投入={row['stake']:.2f}｜返还={row['payout']:.2f}｜"
                f"本期净收益={row['profit']:+.2f}｜累计={row['cumulative_profit']:+.2f}｜"
                f"连续未中={row['miss_streak_after']}｜下一期={'投注' if row['next_bet'] else '暂停'}"
            )
    return '\n'.join(lines) + '\n'


def main():
    import pandas as pd

    from top5_top10_balanced15 import run_top5_top10_balanced15_audit

    parser = argparse.ArgumentParser(description='导出精简15连续4期不中暂停的收益与逐期明细')
    parser.add_argument('--data', default='data/lucky_numbers.csv')
    parser.add_argument('--periods', type=int, default=300)
    parser.add_argument('--stake-per-number', type=float, default=1)
    parser.add_argument('--output-dir', default='预测结果')
    args = parser.parse_args()
    source = pd.read_csv(args.data, encoding='utf-8-sig')
    audit = run_top5_top10_balanced15_audit(source, test_periods=args.periods)
    report = simulate_pause4_betting(audit['rows'], args.periods, args.stake_per_number)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'TOP5_TOP10精简15_连不中4期暂停_每号{args.stake_per_number:g}元_最近{args.periods}期'
    detail_path = output_dir / f'{stem}明细.csv'
    report_path = output_dir / f'{stem}报告.txt'
    export_rows = [
        {
            '序号': row['period'],
            '日期': row['date'],
            '状态': '投注' if row['bet'] else '暂停',
            '预测15号': ','.join(map(str, row['prediction'])),
            '实际号码': row['actual_number'],
            '实际生肖': row['actual_animal'],
            '预测命中': '是' if row['hit'] else '否',
            '实际中奖': '是' if row['bet'] and row['hit'] else '否',
            '本期投入': row['stake'],
            '本期返还': row['payout'],
            '本期净收益': row['profit'],
            '累计净收益': row['cumulative_profit'],
            '期前连续未中': row['miss_streak_before'],
            '期后连续未中': row['miss_streak_after'],
            '下一期状态': '投注' if row['next_bet'] else '暂停',
            '当前回撤': row['drawdown'],
        }
        for row in report['rows']
    ]
    pd.DataFrame(export_rows).to_csv(detail_path, index=False, encoding='utf-8-sig')
    report_path.write_text(format_pause4_betting_report(report), encoding='utf-8-sig')
    print(format_pause4_betting_report(report, include_details=False))
    print(f'明细文件：{detail_path}')
    print(f'完整报告：{report_path}')


if __name__ == '__main__':
    main()