"""Fixed 15-number TOP5 pool with balanced TOP10 consensus scoring."""

from top5_top10_balanced19 import run_top5_top10_balanced19_audit


TOP_N = 15


def run_top5_top10_balanced15_audit(df, test_periods=400):
    """Run a fresh forward-only audit with an independent fixed-15 entry point."""
    return run_top5_top10_balanced19_audit(
        df,
        test_periods=test_periods,
        target_count=TOP_N,
    )