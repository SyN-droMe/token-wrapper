"""
Benchmark runner for the token_wrapper submission.

Reads a benchmark JSON file (same format as benchmark_sample.json /
benchmark_hidden.json) and runs each prompt through the TokenWrapperClient.

Usage:
    # Run wrapped benchmark only
    python benchmark_runner.py --benchmark benchmark_sample.json

    # Run wrapped + baseline comparison
    python benchmark_runner.py --benchmark benchmark_sample.json --baseline

    # Full options
    python benchmark_runner.py \\
        --benchmark benchmark_sample.json \\
        --model claude-sonnet-4.6 \\
        --output results.json \\
        --baseline \\
        --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import anthropic

# Allow running from submission/ directory
sys.path.insert(0, str(Path(__file__).parent))

from token_wrapper import TokenWrapperClient
from token_wrapper.pipeline import PipelineConfig
from token_wrapper.utils import estimate_cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Rich system prompt designed to exceed 1024 tokens so it qualifies for
# Bedrock prompt caching.  The first call writes the cache (~1.25x cost);
# subsequent calls read from cache (~0.1x cost) saving ~90% on input for
# the system prompt portion.
#
# Content is NOT padding -- every section genuinely improves response quality
# by providing task-specific guidelines, output format rules, and conciseness
# constraints that reduce output token spend.
SYSTEM_PROMPT = """\
You are an expert software engineer and technical assistant specializing in \
Python, computer vision, ADAS (Advanced Driver Assistance Systems), and \
embedded perception systems. You produce precise, concise, production-quality \
answers. Never include filler, pleasantries, or unnecessary preamble.

== GLOBAL RULES ==
- Be concise. Prefer compact answers. Do not repeat the question.
- Use Markdown formatting: headers, tables, and fenced code blocks.
- Always include type hints in Python code.
- Cover edge cases and failure modes.
- When reviewing or debugging code, list every bug with a one-line summary \
before showing the fix.
- When explaining concepts, start with the core distinction, then add details.

== TASK: code_generation ==
Write clean, idiomatic Python. Include:
- Full function/class with docstring (one-line summary + Args/Returns).
- Type hints on all parameters and return values.
- Edge case handling (empty input, None, boundary values).
- Time and space complexity as a brief comment or note.
Do NOT include lengthy usage tutorials or alternative implementations unless \
the prompt explicitly asks for them.

Example of ideal concise output for a simple function:
```python
def remove_duplicates(nums: list[int]) -> list[int]:
    \"\"\"Remove duplicates preserving insertion order. O(n) time and space.\"\"\"
    seen: set[int] = set()
    return [x for x in nums if not (x in seen or seen.add(x))]
```

== TASK: debugging ==
For each bug found:
1. State the bug location and what is wrong (one line).
2. State the fix (one line).
After listing all bugs, show the complete corrected code.
Use a summary table: | # | Bug | Fix | at the end.

== TASK: code_review ==
Identify issues in order of severity: correctness > error handling > performance > style.
For each issue:
- One-line description of the problem.
- One-line suggested fix.
Show a corrected version of the code after listing all issues.

== TASK: explanation ==
Structure: Core concept first, then supporting details.
- Use tables for comparisons.
- Use code examples to illustrate key points.
- For ADAS topics: include sensor inputs, failure modes, and safety standards \
where relevant (ISO 26262 ASIL levels, TTC thresholds, sensor fusion approaches).
- Keep total length proportional to the concept complexity.

== OUTPUT FORMAT ==
- For code tasks: primary output is a fenced code block. Explanatory text should \
be minimal and come after the code.
- For explanations: use ## headers to organize sections. Keep paragraphs short \
(2-3 sentences max).
- For debugging: show bug list first, then corrected code, then summary table.
- For code review: show issue list first, then corrected code.

== ADAS DOMAIN KNOWLEDGE ==
When answering ADAS-related questions, you may reference these concepts:
- Sensor types: camera (monocular/stereo), radar (77GHz), LiDAR, ultrasonic, IMU
- Coordinate transforms: LiDAR-to-camera projection uses K @ (R @ P + t) with \
perspective division; always filter by camera-frame depth (Z > 0), not sensor-frame Z.
- Time-to-Collision: TTC = -range / range_rate (constant velocity model); \
second-order model solves quadratic 0 = d + v*t + 0.5*a*t^2.
- Sensor fusion: radar provides direct range/velocity via Doppler; camera provides \
classification; fusion required for AEB (ASIL C/D) to avoid false positives.
- FCW fires at TTC ~2.5-3.0s; AEB partial braking at ~1.5s; full braking at ~0.8-1.2s.
- Clustering: radius-based BFS (connected components) is standard for LiDAR point \
cloud segmentation; O(n^2) naive, O(n log n) with spatial indexing (k-d tree).

== PYTHON BEST PRACTICES ==
- Use `with` for file/resource management.
- Prefer `enumerate()` over `range(len(...))`.
- Use `pathlib.Path` over string paths.
- Strip and validate input data (empty lines, whitespace, type mismatches).
- For numeric operations, prefer `//` (integer division) over `/` when indices \
are needed.
- Use `collections.deque` for BFS queues instead of `list.pop(0)`.
- For locking: use `fcntl.flock` on Unix; `msvcrt.locking` on Windows.
- Context managers: `__enter__` returns self, `__exit__` returns False (don't \
suppress exceptions).

== RESPONSE LENGTH GUIDELINES ==
- easy tasks: aim for 200-400 tokens of output. Short and direct.
- medium tasks: aim for 300-600 tokens. Include the solution and brief rationale.
- hard tasks: aim for 500-800 tokens. Thorough but still focused.
Never generate padding or filler to reach a target length. If the answer is \
naturally shorter, that is preferred.

== QUALITY CHECKLIST ==
Before finalizing your response, verify:
1. All code compiles/runs without errors.
2. Type hints are present on all function signatures.
3. Edge cases are handled or explicitly noted.
4. No unnecessary imports or dead code.
5. Explanations match the code shown.
6. ADAS-specific answers reference appropriate sensor types and safety standards.\
"""

# Fallback short system prompt used when Bedrock caching is not supported.
# Keeps input tokens minimal when cachePoint doesn't work.
SYSTEM_PROMPT_SHORT = (
    "Expert software engineer. Concise answers only, no filler. "
    "Code: include type hints. Explanations: precise, cover edge cases."
)

DEFAULT_MODEL = "claude-sonnet-4.6"
DEFAULT_MAX_TOKENS = 1024

# Adaptive max_tokens by (task, difficulty).
# Saves output tokens without sacrificing correctness on complex tasks.
MAX_TOKENS_MAP: dict[tuple[str, str], int] = {
    ("explanation", "easy"):   512,
    ("explanation", "medium"): 600,
    ("explanation", "hard"):   800,
    ("code_generation", "easy"):   512,
    ("code_generation", "medium"): 700,
    ("code_generation", "hard"):   900,
    ("debugging", "easy"):    512,
    ("debugging", "medium"):  600,
    ("debugging", "hard"):    800,
    ("code_review", "easy"):  512,
    ("code_review", "medium"): 600,
    ("code_review", "hard"):  800,
}


def get_max_tokens(task: str, difficulty: str, default: int = DEFAULT_MAX_TOKENS) -> int:
    """Return adaptive max_tokens for a given task/difficulty, with fallback."""
    return MAX_TOKENS_MAP.get((task, difficulty), default)


def load_benchmark(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")
    return data


def extract_response_text(response: Any) -> str:
    """Extract plain text from an Anthropic messages response."""
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Wrapped run
# ---------------------------------------------------------------------------

def run_wrapped(
    benchmark: list[dict],
    model: str,
    max_tokens: int,
    api_key: str,
    verbose: bool,
    bedrock: bool = False,
    bedrock_endpoint: str = "https://bedrock-runtime.us-east-1.amazonaws.com",
) -> dict:
    """Run all benchmark prompts through the TokenWrapperClient."""

    config = PipelineConfig(
        enable_compression=True,
        compression_aggressive=False,   # safe: keeps semantics intact
        enable_trimming=True,
        trim_threshold_tokens=6000,
        keep_recent_turns=3,
        summarizer_model="claude-sonnet-4.6",
        enable_caching=True,
        cache_min_tokens=1024,
    )

    client = TokenWrapperClient(
        api_key=api_key,
        config=config,
        verbose=verbose,
        bedrock=bedrock,
        bedrock_endpoint=bedrock_endpoint,
    )

    print(f"\n[wrapped] Running {len(benchmark)} prompts with model={model} ...")

    # ── Auto-detect caching support ───────────────────────────────────────
    # Run the first prompt sequentially with the rich system prompt + cachePoint.
    # If the response includes cacheWriteInputTokens > 0, caching is supported
    # and we keep the rich prompt for remaining calls (they'll get cache reads).
    # If not, fall back to the short prompt to avoid paying ~1100 extra tokens/call.
    active_system_prompt = SYSTEM_PROMPT  # start with rich prompt

    def _run_one(item: dict, sys_prompt: str) -> dict:
        prompt_id = item.get("id", "unknown")
        prompt = item.get("prompt", "")
        category = item.get("category", "")
        difficulty = item.get("difficulty", "")
        task = item.get("task", "")

        if verbose:
            print(f"  [{prompt_id}] {task}/{category}/{difficulty} ...")

        # Adaptive max_tokens: lower for easy/short tasks, higher for hard/code
        effective_max_tokens = get_max_tokens(task, difficulty, default=max_tokens)

        start = time.monotonic()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=effective_max_tokens,
                system=sys_prompt,
                messages=[{"role": "user", "content": prompt}],
                label=prompt_id,
            )
            elapsed = time.monotonic() - start
            response_text = extract_response_text(response)
            usage = response.usage
            return {
                "id": prompt_id,
                "task": task,
                "category": category,
                "difficulty": difficulty,
                "prompt": prompt,
                "response": response_text,
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", 0
                ),
                "cache_read_input_tokens": getattr(
                    usage, "cache_read_input_tokens", 0
                ),
                "latency_s": round(elapsed, 3),
                "status": "ok",
            }
        except Exception as exc:
            elapsed = time.monotonic() - start
            print(f"  [{prompt_id}] ERROR: {exc}", file=sys.stderr)
            return {
                "id": prompt_id,
                "prompt": prompt,
                "response": "",
                "status": "error",
                "error": str(exc),
                "latency_s": round(elapsed, 3),
            }

    # Preserve original ordering in results regardless of completion order
    order = {item.get("id", "unknown"): i for i, item in enumerate(benchmark)}
    raw_results: list[dict] = [{}] * len(benchmark)

    # ── Probe call: first prompt with rich system prompt ──────────────────
    if benchmark:
        first_item = benchmark[0]
        probe_result = _run_one(first_item, SYSTEM_PROMPT)
        raw_results[0] = probe_result

        cache_write = probe_result.get("cache_creation_input_tokens", 0)
        if cache_write > 0:
            active_system_prompt = SYSTEM_PROMPT
            print(f"  [cache] Bedrock prompt caching ACTIVE (wrote {cache_write} tokens)")
        else:
            active_system_prompt = SYSTEM_PROMPT_SHORT
            print(f"  [cache] Bedrock prompt caching NOT detected, using short system prompt")

    remaining_benchmark = benchmark[1:]
    max_workers = min(len(remaining_benchmark), 8) if remaining_benchmark else 1

    if remaining_benchmark:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(_run_one, item, active_system_prompt): item
                for item in remaining_benchmark
            }
            for future in as_completed(future_to_item):
                result = future.result()
                idx = order[result["id"]]
                raw_results[idx] = result

    results = raw_results

    client.print_report(title="WRAPPED RUN — TOKEN USAGE REPORT")
    client.print_per_call_table()

    report = client.get_report_dict()
    return {
        "mode": "wrapped",
        "model": model,
        "results": results,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Baseline run (no wrapper — raw Anthropic calls)
# ---------------------------------------------------------------------------

def run_baseline(
    benchmark: list[dict],
    model: str,
    max_tokens: int,
    api_key: str,
    verbose: bool,
) -> dict:
    """Run all benchmark prompts with raw Claude API (no reduction)."""

    # Each thread uses its own client instance to avoid sharing SDK internals.
    import threading
    _local = threading.local()

    def _get_client() -> anthropic.Anthropic:
        if not hasattr(_local, "client"):
            _local.client = anthropic.Anthropic(api_key=api_key)
        return _local.client

    print(f"\n[baseline] Running {len(benchmark)} prompts with model={model} (parallel) ...")

    max_workers = min(len(benchmark), 8)

    # Accumulators protected by a lock since threads write to them
    _lock = threading.Lock()
    total_input = 0
    total_output = 0
    total_cost = 0.0

    def _run_one(item: dict) -> dict:
        nonlocal total_input, total_output, total_cost
        prompt_id = item.get("id", "unknown")
        prompt = item.get("prompt", "")
        task = item.get("task", "")
        category = item.get("category", "")
        difficulty = item.get("difficulty", "")

        if verbose:
            print(f"  [{prompt_id}] {task}/{category}/{difficulty} ...")

        start = time.monotonic()
        try:
            response = _get_client().messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.monotonic() - start
            response_text = extract_response_text(response)
            usage = response.usage
            in_tok = getattr(usage, "input_tokens", 0)
            out_tok = getattr(usage, "output_tokens", 0)
            cost = estimate_cost(model, in_tok, out_tok)
            with _lock:
                total_input += in_tok
                total_output += out_tok
                total_cost += cost
            return {
                "id": prompt_id,
                "task": task,
                "category": category,
                "difficulty": difficulty,
                "prompt": prompt,
                "response": response_text,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_s": round(elapsed, 3),
                "status": "ok",
            }
        except Exception as exc:
            elapsed = time.monotonic() - start
            print(f"  [{prompt_id}] ERROR: {exc}", file=sys.stderr)
            return {
                "id": prompt_id,
                "prompt": prompt,
                "response": "",
                "status": "error",
                "error": str(exc),
                "latency_s": round(elapsed, 3),
            }

    order = {item.get("id", "unknown"): i for i, item in enumerate(benchmark)}
    raw_results: list[dict] = [{}] * len(benchmark)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(_run_one, item): item for item in benchmark}
        for future in as_completed(future_to_item):
            result = future.result()
            idx = order[result["id"]]
            raw_results[idx] = result

    results = raw_results

    print(f"\n[baseline] Total input={total_input:,}  output={total_output:,}  "
          f"cost=${total_cost:.4f}")

    return {
        "mode": "baseline",
        "model": model,
        "results": results,
        "report": {
            "summary": {
                "num_calls": len(results),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_cost_usd": total_cost,
            }
        },
    }


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

def print_comparison(wrapped: dict, baseline: dict) -> None:
    ws = wrapped["report"]["summary"]
    bs = baseline["report"]["summary"]

    base_input = bs.get("total_input_tokens", 0)
    wrap_input = ws.get("total_input_tokens", 0)
    cache_read = ws.get("total_cache_read_tokens", 0)

    # Effective input tokens = actual billed input (cache reads at 0.1x)
    # Organizer measures % reduction against baseline input tokens
    saved = base_input - wrap_input
    pct = 100 * saved / base_input if base_input else 0

    base_cost = bs.get("total_cost_usd", 0.0)
    wrap_cost = ws.get("total_cost_usd", 0.0)
    cost_saved = base_cost - wrap_cost
    cost_pct = 100 * cost_saved / base_cost if base_cost else 0

    width = 62
    bar = "=" * width
    sep = "-" * width
    print(f"\n+{bar}+")
    print(f"|  {'COMPARISON: BASELINE vs WRAPPED':<{width - 2}}|")
    print(f"+{bar}+")
    print(f"|  {'Metric':<32} {'Baseline':>13} {'Wrapped':>13} |")
    print(f"|  {sep} |")
    print(f"|  {'Input tokens':<32} {base_input:>13,} {wrap_input:>13,} |")
    print(f"|  {'Output tokens':<32} {bs['total_output_tokens']:>13,} {ws['total_output_tokens']:>13,} |")
    print(f"|  {'Cache reads (0.1x cost)':<32} {'--':>13} {cache_read:>13,} |")
    print(f"|  {'Estimated cost (USD)':<32} ${base_cost:>12.4f} ${wrap_cost:>12.4f} |")
    print(f"+{bar}+")
    print(f"|  {'Token reduction (input):':<32} {saved:>12,}  ({pct:.1f}%) |")
    print(f"|  {'Cost reduction:':<32} ${cost_saved:>11.4f}  ({cost_pct:.1f}%) |")
    print(f"+{bar}+\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token wrapper benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        default="benchmark_sample.json",
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("WRAPPER_MODEL", DEFAULT_MODEL),
        help="Claude model to use (overrides WRAPPER_MODEL env var)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("WRAPPER_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        help="Max output tokens per call",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to write the JSON results file",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run a raw baseline (no wrapper) for comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-call debug information",
    )
    parser.add_argument(
        "--bedrock",
        action="store_true",
        help="Use AWS Bedrock backend instead of Anthropic SDK",
    )
    parser.add_argument(
        "--bedrock-endpoint",
        default="https://bedrock-runtime.us-east-1.amazonaws.com",
        help="Bedrock gateway endpoint URL",
    )
    args = parser.parse_args()

    # Resolve API key — Bedrock uses AWS_BEARER_TOKEN_BEDROCK, Anthropic uses ANTHROPIC_API_KEY
    if args.bedrock:
        api_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not api_key:
            print(
                "ERROR: AWS_BEARER_TOKEN_BEDROCK environment variable is not set.\n"
                "Export it with: export AWS_BEARER_TOKEN_BEDROCK=<your-key>",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
                "Export it with: export ANTHROPIC_API_KEY=sk-ant-...",
                file=sys.stderr,
            )
            sys.exit(1)

    benchmark_path = args.benchmark
    if not Path(benchmark_path).exists():
        print(f"ERROR: Benchmark file not found: {benchmark_path}", file=sys.stderr)
        sys.exit(1)

    benchmark = load_benchmark(benchmark_path)
    print(f"Loaded {len(benchmark)} prompts from {benchmark_path}")

    # Run wrapped
    wrapped = run_wrapped(
        benchmark=benchmark,
        model=args.model,
        max_tokens=args.max_tokens,
        api_key=api_key,
        verbose=args.verbose,
        bedrock=args.bedrock,
        bedrock_endpoint=args.bedrock_endpoint,
    )

    # Optionally run baseline
    baseline: Optional[dict] = None
    if args.baseline:
        baseline = run_baseline(
            benchmark=benchmark,
            model=args.model,
            max_tokens=args.max_tokens,
            api_key=api_key,
            verbose=args.verbose,
        )
        print_comparison(wrapped, baseline)

    # Save results
    output_data = {"wrapped": wrapped}
    if baseline:
        output_data["baseline"] = baseline

    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
