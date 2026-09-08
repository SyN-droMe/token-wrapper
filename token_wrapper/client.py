"""
TokenWrapperClient — drop-in replacement for anthropic.Anthropic().

Usage:
    from token_wrapper import TokenWrapperClient

    client = TokenWrapperClient(api_key="sk-ant-...")
    response = client.messages.create(
        model="claude-sonnet-4.6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    client.print_report()

The client intercepts every messages.create() call, applies the reduction
pipeline, logs token usage, and forwards the cleaned request to Claude.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterator, Optional

import anthropic

from .bedrock_client import BedrockAdapter
from .logger import UsageLogger
from .pipeline import PipelineConfig, ReductionPipeline
from .reporter import UsageReporter
from .utils import estimate_cost, hash_content, messages_token_estimate, estimate_tokens

log = logging.getLogger("token_wrapper.client")


class _MessagesProxy:
    """
    Proxy for anthropic_client.messages — intercepts .create() calls.
    Mirrors the interface of anthropic.resources.Messages.
    """

    def __init__(self, wrapper: "TokenWrapperClient") -> None:
        self._wrapper = wrapper
        # Local response cache: {prompt_hash: response}
        # Avoids redundant API calls for identical prompts (same model+system+messages).
        self._response_cache: dict[str, Any] = {}

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: Optional[str | list] = None,
        stream: bool = False,
        label: Optional[str] = None,  # extra kwarg — removed before API call
        **kwargs: Any,
    ) -> Any:
        """
        Intercept, reduce, and forward a messages.create() call.

        Extra keyword argument:
            label: str — optional human-readable name for this call in reports.
        """
        wrapper = self._wrapper

        # ── Pre-reduction token estimate ──────────────────────────────────
        pre_tokens = messages_token_estimate(messages)
        if isinstance(system, str):
            pre_tokens += estimate_tokens(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    pre_tokens += estimate_tokens(block.get("text", ""))

        # ── Register call record before sending ───────────────────────────
        record = wrapper.logger.new_record(
            model=model,
            pre_reduction_input_tokens=pre_tokens,
            label=label,
        )

        # ── Apply reduction pipeline ──────────────────────────────────────
        processed_messages, processed_system, strategies = wrapper.pipeline.process(
            messages=messages,
            system=system,
            model=model,
        )
        record.strategies_applied = strategies

        # ── Local response cache check ────────────────────────────────────
        # Hash the processed request to check for exact duplicates.
        # If found, return cached response with zero API cost.
        cache_key = None
        if wrapper.enable_response_cache and not stream:
            import json
            cache_payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": processed_messages,
                "system": processed_system if isinstance(processed_system, str)
                    else json.dumps(processed_system, sort_keys=True)
                    if processed_system else None,
            }
            cache_key = hash_content(cache_payload)

            if cache_key in self._response_cache:
                cached = self._response_cache[cache_key]
                record.input_tokens = 0
                record.output_tokens = 0
                record.cost_usd = 0.0
                strategies.append("response_cache_hit")
                log.debug("[%s] Response cache hit for %s", record.call_id, label or model)
                return cached

        # ── Build API call kwargs ─────────────────────────────────────────
        api_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": processed_messages,
            **kwargs,
        }
        if processed_system is not None:
            api_kwargs["system"] = processed_system
        if stream:
            api_kwargs["stream"] = True

        # ── Forward to Claude API ─────────────────────────────────────────
        try:
            if stream:
                return self._handle_stream(api_kwargs, record)

            response = wrapper._raw_client.messages.create(**api_kwargs)

            # ── Extract token usage from response ─────────────────────────
            usage = response.usage
            record.input_tokens = getattr(usage, "input_tokens", 0)
            record.output_tokens = getattr(usage, "output_tokens", 0)
            record.cache_creation_input_tokens = getattr(
                usage, "cache_creation_input_tokens", 0
            )
            record.cache_read_input_tokens = getattr(
                usage, "cache_read_input_tokens", 0
            )
            record.cost_usd = estimate_cost(
                model=model,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                cache_creation_tokens=record.cache_creation_input_tokens,
                cache_read_tokens=record.cache_read_input_tokens,
            )

            log.debug(
                "[%s] %s in=%d out=%d saved~%d cost=$%.6f strats=%s",
                record.call_id,
                label or model,
                record.input_tokens,
                record.output_tokens,
                record.tokens_saved,
                record.cost_usd,
                strategies,
            )

            # ── Store in local response cache ─────────────────────────────
            if cache_key is not None:
                self._response_cache[cache_key] = response

            return response

        except Exception as exc:
            log.error("API call failed for %s: %s", record.call_id, exc)
            raise

    def _handle_stream(self, api_kwargs: dict, record: Any) -> Iterator:
        """
        Wrap a streaming response to capture usage once the stream completes.
        Yields chunks transparently.
        """
        with self._wrapper._raw_client.messages.stream(**api_kwargs) as stream:
            for chunk in stream:
                yield chunk
            # After stream ends, usage is available in the final message
            try:
                final = stream.get_final_message()
                usage = final.usage
                record.input_tokens = getattr(usage, "input_tokens", 0)
                record.output_tokens = getattr(usage, "output_tokens", 0)
                record.cache_creation_input_tokens = getattr(
                    usage, "cache_creation_input_tokens", 0
                )
                record.cache_read_input_tokens = getattr(
                    usage, "cache_read_input_tokens", 0
                )
                record.cost_usd = estimate_cost(
                    model=api_kwargs["model"],
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cache_creation_tokens=record.cache_creation_input_tokens,
                    cache_read_tokens=record.cache_read_input_tokens,
                )
            except Exception:
                pass  # Non-fatal; usage just won't be recorded for this stream


class TokenWrapperClient:
    """
    Drop-in replacement for anthropic.Anthropic(), with optional AWS Bedrock backend.

    Supports the same interface as the official client for the
    messages.create() endpoint, with transparent token reduction.

    Args:
        api_key:            API key. For Anthropic: sk-ant-... (or ANTHROPIC_API_KEY).
                            For Bedrock: bearer token (or AWS_BEARER_TOKEN_BEDROCK).
        config:             PipelineConfig — controls which strategies are applied.
        verbose:            If True, logs per-call details to stderr.
        bedrock:            If True, use AWS Bedrock backend instead of Anthropic SDK.
        bedrock_endpoint:   Bedrock gateway URL.
        bedrock_region:     AWS region string.
        enable_response_cache: If True, cache API responses by prompt hash. Identical
                            prompts return cached responses with zero token cost.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
        verbose: bool = False,
        bedrock: bool = False,
        bedrock_endpoint: str = "https://bedrock-runtime.us-east-1.amazonaws.com",
        bedrock_region: str = "us-east-1",
        enable_response_cache: bool = True,
    ) -> None:
        if verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(levelname)s %(name)s: %(message)s",
            )

        if bedrock:
            self._raw_client = BedrockAdapter(
                api_key=api_key,
                endpoint_url=bedrock_endpoint,
                region_name=bedrock_region,
            )
        else:
            resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "No API key provided. Set ANTHROPIC_API_KEY or pass api_key=."
                )
            self._raw_client = anthropic.Anthropic(api_key=resolved_key)

        self.enable_response_cache = enable_response_cache
        self.logger = UsageLogger()
        self.pipeline = ReductionPipeline(
            config=config or PipelineConfig(),
            api_client=self._raw_client,
        )
        self.reporter = UsageReporter(self.logger)
        self.messages = _MessagesProxy(self)

    # ── Convenience report methods ────────────────────────────────────────

    def print_report(self, title: str = "TOKEN USAGE REPORT") -> None:
        """Print a human-readable summary table to stdout."""
        self.reporter.print_report(title=title)

    def print_per_call_table(self) -> None:
        """Print a per-call detail table to stdout."""
        self.reporter.print_per_call_table()

    def save_report(self, path: str) -> None:
        """Save a JSON report to `path`."""
        self.reporter.save_json(path)

    def get_report_dict(self) -> dict:
        """Return the report as a Python dict."""
        return self.reporter.to_dict()

    def reset(self) -> None:
        """Reset all accumulated usage data."""
        self.logger.reset()

    # ── Context manager support ───────────────────────────────────────────

    def __enter__(self) -> "TokenWrapperClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.print_report()
