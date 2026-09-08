"""
Per-call token usage logger.
Stores a record for every API call made through the wrapper.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CallRecord:
    """Token usage record for a single API call."""
    call_id: str
    model: str
    timestamp: float

    # Actual token counts from API response
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    # Estimated pre-reduction token counts (for savings calculation)
    pre_reduction_input_tokens: int = 0

    # Applied strategies
    strategies_applied: list[str] = field(default_factory=list)

    # Cost estimates (USD)
    cost_usd: float = 0.0

    # Extra metadata
    label: Optional[str] = None  # e.g. benchmark prompt ID

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens_saved(self) -> int:
        """Tokens saved compared to pre-reduction estimate."""
        return max(0, self.pre_reduction_input_tokens - self.input_tokens)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "model": self.model,
            "timestamp": self.timestamp,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "pre_reduction_input_tokens": self.pre_reduction_input_tokens,
            "tokens_saved": self.tokens_saved,
            "strategies_applied": self.strategies_applied,
            "cost_usd": self.cost_usd,
            "label": self.label,
        }


class UsageLogger:
    """Thread-safe accumulator of per-call usage records."""

    def __init__(self) -> None:
        self._records: list[CallRecord] = []
        self._call_counter = 0
        self._lock = threading.Lock()

    def new_record(
        self,
        model: str,
        pre_reduction_input_tokens: int = 0,
        label: Optional[str] = None,
    ) -> CallRecord:
        """Create and register a new (empty) call record, returned for population."""
        with self._lock:
            self._call_counter += 1
            call_id = f"call_{self._call_counter:04d}"
        record = CallRecord(
            call_id=call_id,
            model=model,
            timestamp=time.time(),
            pre_reduction_input_tokens=pre_reduction_input_tokens,
            label=label,
        )
        with self._lock:
            self._records.append(record)
        return record

    @property
    def records(self) -> list[CallRecord]:
        with self._lock:
            return list(self._records)

    def total_input_tokens(self) -> int:
        with self._lock:
            return sum(r.input_tokens for r in self._records)

    def total_output_tokens(self) -> int:
        with self._lock:
            return sum(r.output_tokens for r in self._records)

    def total_cache_creation_tokens(self) -> int:
        with self._lock:
            return sum(r.cache_creation_input_tokens for r in self._records)

    def total_cache_read_tokens(self) -> int:
        with self._lock:
            return sum(r.cache_read_input_tokens for r in self._records)

    def total_pre_reduction_input_tokens(self) -> int:
        with self._lock:
            return sum(r.pre_reduction_input_tokens for r in self._records)

    def total_tokens_saved(self) -> int:
        with self._lock:
            return sum(r.tokens_saved for r in self._records)

    def total_cost_usd(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    def most_expensive_calls(self, n: int = 5) -> list[CallRecord]:
        with self._lock:
            return sorted(self._records, key=lambda r: r.cost_usd, reverse=True)[:n]

    def to_dict_list(self) -> list[dict]:
        with self._lock:
            return [r.to_dict() for r in self._records]

    def reset(self) -> None:
        with self._lock:
            self._records = []
            self._call_counter = 0
