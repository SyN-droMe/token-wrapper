"""
Utility helpers: token counting, text compression, hashing.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Fast heuristic token estimate without API calls.
    Claude tokenizer averages ~4 characters per token for English text.
    We use 3.8 to be slightly conservative (avoids underestimating).
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.8))


def messages_token_estimate(messages: list[dict]) -> int:
    """Estimate total tokens for a list of message dicts."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += estimate_tokens(block.get("text", ""))
    return total


# ---------------------------------------------------------------------------
# Text compression (heuristic, lossless-ish)
# ---------------------------------------------------------------------------

# Precompile patterns for performance
_MULTI_BLANK_LINES = re.compile(r'\n{3,}')
_LEADING_TRAILING_SPACES = re.compile(r'[ \t]+\n')
_REPEATED_SPACES = re.compile(r'  +')

# Filler phrases that add tokens but not meaning — safe to remove
_FILLER_PATTERNS = [
    re.compile(r'\bplease\b\s*', re.IGNORECASE),
    re.compile(r'\bkindly\b\s*', re.IGNORECASE),
    re.compile(r'\bif you (could|would|can|don\'t mind)\b[,.]?\s*', re.IGNORECASE),
    re.compile(r'\bfeel free to\b\s*', re.IGNORECASE),
    re.compile(r'\bi would (like|appreciate it if)\b\s+you\s+(could\s+)?', re.IGNORECASE),
    re.compile(r'\bcould you\b\s*', re.IGNORECASE),
    re.compile(r'\bwould you\b\s*', re.IGNORECASE),
    re.compile(r'\bI want you to\b\s*', re.IGNORECASE),
    re.compile(r'\bI need you to\b\s*', re.IGNORECASE),
    re.compile(r'\bcertain(ly)?\b\s*', re.IGNORECASE),
    re.compile(r'\bof course\b[,.]?\s*', re.IGNORECASE),
    re.compile(r'\bsure(ly)?\b[,.]?\s*', re.IGNORECASE),
]

# Verbose phrase → compact equivalent (order matters: longest first)
_VERBOSE_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bin order to\b', re.IGNORECASE), 'to'),
    (re.compile(r'\bdue to the fact that\b', re.IGNORECASE), 'because'),
    (re.compile(r'\bfor the purpose of\b', re.IGNORECASE), 'for'),
    (re.compile(r'\bat this point in time\b', re.IGNORECASE), 'now'),
    (re.compile(r'\bin the event that\b', re.IGNORECASE), 'if'),
    (re.compile(r'\bwith regard to\b', re.IGNORECASE), 'regarding'),
    (re.compile(r'\bwith respect to\b', re.IGNORECASE), 'regarding'),
    (re.compile(r'\bprior to\b', re.IGNORECASE), 'before'),
    (re.compile(r'\bsubsequent to\b', re.IGNORECASE), 'after'),
    (re.compile(r'\bin spite of\b', re.IGNORECASE), 'despite'),
    (re.compile(r'\bper se\b', re.IGNORECASE), ''),
    (re.compile(r'\bin terms of\b', re.IGNORECASE), 'in'),
    (re.compile(r'\ba (large|great|significant) number of\b', re.IGNORECASE), 'many'),
    (re.compile(r'\bthe majority of\b', re.IGNORECASE), 'most'),
    (re.compile(r'\bwith the exception of\b', re.IGNORECASE), 'except'),
    (re.compile(r'\bhas the ability to\b', re.IGNORECASE), 'can'),
    (re.compile(r'\bis able to\b', re.IGNORECASE), 'can'),
    (re.compile(r'\bwas able to\b', re.IGNORECASE), 'could'),
    (re.compile(r'\bmake sure that\b', re.IGNORECASE), 'ensure'),
    (re.compile(r'\bthe fact that\b', re.IGNORECASE), 'that'),
]


def compress_text(text: str, aggressive: bool = False) -> str:
    """
    Compress prompt text using heuristic rules.
    - Normalizes whitespace
    - Removes verbose filler phrases
    - Substitutes wordy constructs with concise equivalents
    - aggressive=True also strips filler polite openers
    """
    if not text or len(text) < 50:
        return text

    # Normalize whitespace
    text = _LEADING_TRAILING_SPACES.sub('\n', text)
    text = _MULTI_BLANK_LINES.sub('\n\n', text)
    text = _REPEATED_SPACES.sub(' ', text)
    text = text.strip()

    # Apply verbose replacements
    for pattern, replacement in _VERBOSE_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    # Apply filler removal (only in aggressive mode to avoid mangling code)
    if aggressive:
        for pattern in _FILLER_PATTERNS:
            text = pattern.sub('', text)

    # Clean up any double spaces introduced by substitutions
    text = _REPEATED_SPACES.sub(' ', text)
    text = text.strip()

    return text


def compress_messages(
    messages: list[dict],
    aggressive: bool = False,
) -> tuple[list[dict], int, int]:
    """
    Compress all text content in a messages list.
    Returns (compressed_messages, original_token_estimate, compressed_token_estimate).
    Code blocks (``` fenced) are preserved verbatim.
    """
    original_tokens = messages_token_estimate(messages)
    compressed = []

    for msg in messages:
        new_msg = dict(msg)
        content = msg.get("content", "")

        if isinstance(content, str):
            new_msg["content"] = _compress_preserving_code(content, aggressive)
        elif isinstance(content, list):
            new_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    new_block = dict(block)
                    new_block["text"] = _compress_preserving_code(block["text"], aggressive)
                    new_blocks.append(new_block)
                else:
                    new_blocks.append(block)
            new_msg["content"] = new_blocks

        compressed.append(new_msg)

    compressed_tokens = messages_token_estimate(compressed)
    return compressed, original_tokens, compressed_tokens


_CODE_INDICATORS = re.compile(
    r'```'                      # fenced code blocks
    r'|^\s*def\s+\w+\s*\('      # Python function definitions
    r'|^\s*class\s+\w+[\s(:]'   # Python class definitions
    r'|^\s*if\s+__name__\s*=='  # main guard
    r'|^\s*import\s+\w+'        # import statements
    r'|^\s*from\s+\w+\s+import' # from imports
    r'|^\s*return\s+'           # return statements
    r'|^\s*for\s+\w+\s+in\s+'  # for loops
    r'|^\s*while\s+\w+',       # while loops
    re.MULTILINE,
)


def _has_code(text: str) -> bool:
    """Return True if text contains code blocks or inline code patterns."""
    return bool(_CODE_INDICATORS.search(text))


def _compress_preserving_code(text: str, aggressive: bool) -> str:
    """
    Compress text while leaving fenced code blocks untouched.

    If the prompt contains code blocks, skip compression entirely —
    whitespace normalization can shift token boundaries in code-heavy
    prompts and actually *increase* real token count despite our
    heuristic estimating a decrease.
    """
    if _has_code(text):
        return text

    return compress_text(text, aggressive=aggressive)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hash_content(content: Any) -> str:
    """Stable SHA-256 hash of any string-serializable content."""
    if isinstance(content, str):
        raw = content
    else:
        import json
        raw = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Prices per million tokens as of July 2026 (USD)
# Source: https://www.anthropic.com/pricing
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-3-5-haiku-20241022": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-3-7-sonnet-20250219": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-sonnet-4-5": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-opus-4-5": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
}

_DEFAULT_PRICING = {
    "input": 3.00,
    "output": 15.00,
    "cache_write": 3.75,
    "cache_read": 0.30,
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate cost in USD for a single API call."""
    pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
    cost = (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
        + (cache_creation_tokens / 1_000_000) * pricing["cache_write"]
        + (cache_read_tokens / 1_000_000) * pricing["cache_read"]
    )
    return cost
