"""
Summary report generator for token usage.
Produces both a human-readable console table and a machine-readable dict/JSON.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .logger import UsageLogger


def _fmt_tokens(n: int) -> str:
    """Format token count with commas."""
    return f"{n:,}"


def _fmt_cost(usd: float) -> str:
    if usd < 0.001:
        return f"${usd * 1000:.4f}m"  # millicents
    return f"${usd:.4f}"


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{100 * part / total:.1f}%"


class UsageReporter:
    """Generates token usage reports from a UsageLogger."""

    def __init__(self, logger: "UsageLogger") -> None:
        self._logger = logger

    # -------------------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------------------

    def print_report(self, title: str = "TOKEN USAGE REPORT") -> None:
        """Print a formatted summary to stdout."""
        logger = self._logger
        records = logger.records

        if not records:
            print("No API calls recorded.")
            return

        total_in = logger.total_input_tokens()
        total_out = logger.total_output_tokens()
        total_cache_create = logger.total_cache_creation_tokens()
        total_cache_read = logger.total_cache_read_tokens()
        total_pre = logger.total_pre_reduction_input_tokens()
        tokens_saved = logger.total_tokens_saved()
        total_cost = logger.total_cost_usd()
        num_calls = len(records)

        # Calculate effective savings including cache reads
        cache_savings_tok = total_cache_read  # cache reads are ~90% cheaper
        pct_reduction = _pct(tokens_saved, total_pre) if total_pre > 0 else "N/A"

        width = 62
        sep = "-" * width
        bar = "=" * width

        print(f"\n+{bar}+")
        print(f"|  {title:<{width - 2}}|")
        print(f"+{bar}+")
        print(f"|  {'API Calls:':<30} {num_calls:>28} |")
        print(f"|  {sep} |")
        print(f"|  {'INPUT TOKENS':<30} {'':>28} |")
        print(f"|  {'  Actual input (billed):':<30} {_fmt_tokens(total_in):>28} |")

        if total_pre > 0:
            print(f"|  {'  Pre-reduction estimate:':<30} {_fmt_tokens(total_pre):>28} |")
            print(f"|  {'  Tokens saved (compression):':<30} {_fmt_tokens(tokens_saved):>28} |")
            print(f"|  {'  Reduction %:':<30} {pct_reduction:>28} |")

        if total_cache_create > 0:
            print(f"|  {'  Cache writes (1x cost):':<30} {_fmt_tokens(total_cache_create):>28} |")
        if total_cache_read > 0:
            print(f"|  {'  Cache reads (0.1x cost):':<30} {_fmt_tokens(total_cache_read):>28} |")

        print(f"|  {sep} |")
        print(f"|  {'OUTPUT TOKENS':<30} {'':>28} |")
        print(f"|  {'  Output tokens:':<30} {_fmt_tokens(total_out):>28} |")
        print(f"|  {sep} |")
        print(f"|  {'COST ESTIMATE (USD)':<30} {'':>28} |")
        print(f"|  {'  Total estimated cost:':<30} {_fmt_cost(total_cost):>28} |")
        print(f"+{bar}+")

        # Top expensive calls
        top = logger.most_expensive_calls(5)
        print(f"|  {'MOST EXPENSIVE CALLS':<{width - 2}} |")
        for rec in top:
            label = rec.label or rec.call_id
            details = f"{_fmt_tokens(rec.total_tokens)} tok  {_fmt_cost(rec.cost_usd)}"
            strats = ",".join(rec.strategies_applied) if rec.strategies_applied else "none"
            print(f"|    {label:<22} {details:>24}  |")
            print(f"|    {'  strategies: ' + strats:<{width - 4}} |")

        print(f"+{bar}+\n")

    # -------------------------------------------------------------------------
    # Machine-readable output
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a structured summary dict (suitable for JSON serialization)."""
        logger = self._logger
        total_pre = logger.total_pre_reduction_input_tokens()
        total_in = logger.total_input_tokens()
        tokens_saved = logger.total_tokens_saved()

        pct_reduction = (
            round(100 * tokens_saved / total_pre, 2) if total_pre > 0 else 0.0
        )

        return {
            "summary": {
                "num_calls": len(logger.records),
                "total_input_tokens": total_in,
                "total_output_tokens": logger.total_output_tokens(),
                "total_cache_creation_tokens": logger.total_cache_creation_tokens(),
                "total_cache_read_tokens": logger.total_cache_read_tokens(),
                "pre_reduction_input_tokens": total_pre,
                "tokens_saved": tokens_saved,
                "pct_token_reduction": pct_reduction,
                "total_cost_usd": logger.total_cost_usd(),
            },
            "calls": logger.to_dict_list(),
        }

    def save_json(self, path: str) -> None:
        """Save report to a JSON file."""
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {path}")

    def print_per_call_table(self) -> None:
        """Print a per-call table to stdout."""
        records = self._logger.records
        if not records:
            return

        header = f"{'ID':<12} {'Label':<24} {'In':>8} {'Out':>8} {'Saved':>8} {'Cost':>10}  Strategies"
        print(header)
        print("-" * len(header))
        for r in records:
            label = (r.label or "")[:22]
            strats = ",".join(r.strategies_applied) or "-"
            print(
                f"{r.call_id:<12} {label:<24} "
                f"{_fmt_tokens(r.input_tokens):>8} {_fmt_tokens(r.output_tokens):>8} "
                f"{_fmt_tokens(r.tokens_saved):>8} {_fmt_cost(r.cost_usd):>10}  {strats}"
            )
