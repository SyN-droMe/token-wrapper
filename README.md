# Token Wrapper — LLM Token Reduction Layer

A drop-in wrapper that **intercepts every LLM call**, applies a multi-strategy token reduction
pipeline, and produces detailed usage reports. Supports the Anthropic SDK and AWS Bedrock.

---

## Quick Start

**Anthropic:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
bash run.sh
```

**AWS Bedrock:**
```bash
export AWS_BEARER_TOKEN_BEDROCK=<your-key>
python benchmark_runner.py --bedrock --model claude-sonnet-4 --benchmark benchmark_sample.json
```

To also measure against an unoptimized baseline:
```bash
bash run.sh --baseline
```

---

## Requirements

- Python 3.9+
- `pip install -r requirements.txt` (`anthropic` + `boto3`)
- An Anthropic API key **or** a Bedrock bearer token

---

## Step-by-Step Install & Run

### 1. Get an API Key

**Anthropic:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**AWS Bedrock:**
```bash
export AWS_BEARER_TOKEN_BEDROCK=<your-bearer-token>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `anthropic` (Anthropic SDK) and `boto3` (Bedrock support).

### 3. Run the Benchmark

#### Linux / macOS

```bash
# Basic run — wrapped mode, outputs to results.json
bash run.sh

# With baseline comparison
bash run.sh --baseline

# Different model
bash run.sh --model claude-sonnet-4.6

# Benchmark JSON
bash run.sh --benchmark benchmark_sample.json --output results.json

# Verbose per-call debug output
bash run.sh --verbose
```

#### Windows (PowerShell)

```powershell
# Set API key
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Or for Bedrock
$env:AWS_BEARER_TOKEN_BEDROCK = "<your-bearer-token>"

# Install dependencies
pip install -r requirements.txt

# Basic run
python benchmark_runner.py --benchmark benchmark_sample.json --output results.json

# Bedrock
python benchmark_runner.py --bedrock --model claude-sonnet-4 --benchmark benchmark_sample.json --output results.json

# With baseline + verbose
python benchmark_runner.py --benchmark benchmark_sample.json --output results.json --baseline --verbose
```

#### Direct Python (all platforms)

```bash
# Anthropic
python benchmark_runner.py \
    --benchmark benchmark_sample.json \
    --model claude-sonnet-4.6 \
    --output results.json --baseline --verbose

# Bedrock
python benchmark_runner.py \
    --bedrock --model claude-sonnet-4 \
    --benchmark benchmark_sample.json \
    --output results.json --verbose
```

### 4. Read the Output

**Console output** — two tables are printed after the run:

1. A summary report with total tokens, cost, and most expensive calls
2. A per-call table with input/output tokens, savings, and strategies applied

**`results.json`** — machine-readable output containing:
- Every prompt's response text
- Per-call token counts (input, output, cache reads/writes)
- Summary report with % token reduction and total cost

---

## Using the Library in Your Own Code

```python
from token_wrapper import TokenWrapperClient
from token_wrapper.pipeline import PipelineConfig

config = PipelineConfig(
    enable_compression=True,
    enable_trimming=True,
    enable_caching=True,
)

client = TokenWrapperClient(api_key="sk-ant-...", config=config)

response = client.messages.create(
    model="claude-sonnet-4.6",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Explain recursion."}],
    label="my_call",  # optional label for reports
)

print(response.content[0].text)
client.print_report()
```

### Context manager usage (auto-prints report on exit):

```python
with TokenWrapperClient(api_key="...") as client:
    client.messages.create(...)
```

---

## Project Structure

```
token-wrapper/
├── token_wrapper/
│   ├── __init__.py         # Public API: TokenWrapperClient, UsageReporter, UsageLogger
│   ├── client.py           # Drop-in wrapper (Anthropic SDK or Bedrock)
│   ├── bedrock_client.py   # AWS Bedrock adapter
│   ├── pipeline.py         # Reduction pipeline (compression, trimming, caching)
│   ├── logger.py           # Per-call token usage accumulator (thread-safe)
│   ├── reporter.py         # Console and JSON report generator
│   └── utils.py            # Token counting, text compression, cost estimation
├── benchmark_runner.py     # CLI benchmark harness (parallel, Anthropic + Bedrock)
├── benchmark_sample.json   # Sample benchmark prompts
├── run.sh                  # Entry point script
├── requirements.txt
├── README.md
└── architecture.md
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required for Anthropic)* | Anthropic API key (`sk-ant-...`) |
| `AWS_BEARER_TOKEN_BEDROCK` | *(required for Bedrock)* | Bearer token for Bedrock gateway |
| `WRAPPER_MODEL` | `claude-sonnet-4.6` | Default model (overridden by `--model`) |
| `WRAPPER_MAX_TOKENS` | `1024` | Max output tokens per call (overridden by `--max-tokens`) |

---

## Notes on Correctness

The wrapper does **not** truncate or rewrite prompt meaning. Reduction strategies are:
- **Prompt caching** — zero semantic change; uses Anthropic's native feature
- **Text compression** — normalizes whitespace, replaces verbose constructs with concise
  equivalents (e.g. "in order to" → "to"). Prompts containing code (fenced or unfenced)
  are automatically skipped to avoid counterproductive token inflation
- **Context trimming** — only summarizes long multi-turn history (>6000 tokens) using
  the configured model; all relevant context is preserved in the summary
- **Output reduction** — concise system prompt and adaptive `max_tokens` per task/difficulty
  reduce output verbosity without sacrificing answer quality
