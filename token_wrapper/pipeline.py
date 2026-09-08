"""
Token reduction pipeline.

Strategies applied in order:
  1. Prompt compression   — remove verbose filler, normalize whitespace
  2. Context trimming     — summarize old turns when history exceeds threshold
  3. Prompt caching       — inject cache_control on stable content blocks

Each strategy is independent and can be toggled via config.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .utils import (
    compress_messages,
    estimate_tokens,
    hash_content,
    messages_token_estimate,
)

log = logging.getLogger("token_wrapper.pipeline")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # Prompt compression
    enable_compression: bool = True
    compression_aggressive: bool = False  # removes polite filler words too

    # Context trimming / summarization
    enable_trimming: bool = True
    # When estimated input tokens exceed this, summarize old turns.
    # With sonnet-4.6 as the summarizer (same cost as main model), trimming
    # only saves money when old turns are substantially longer than the summary
    # call itself — threshold set high to ensure net positive savings.
    trim_threshold_tokens: int = 6000
    # How many recent turns to keep verbatim (not summarized)
    keep_recent_turns: int = 3
    # Model used for summarization (cheap model preferred)
    summarizer_model: str = "claude-sonnet-4.6"

    # Prompt caching (Anthropic cache_control)
    enable_caching: bool = True
    # Minimum token size for a block to be worth caching (API minimum is 1024)
    cache_min_tokens: int = 1024

    # Extra fields for future strategies
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class ReductionPipeline:
    """
    Applies a sequence of token reduction strategies to messages before
    they are sent to the Claude API.
    """

    def __init__(self, config: Optional[PipelineConfig] = None, api_client: Any = None) -> None:
        self.config = config or PipelineConfig()
        # api_client is the raw anthropic.Anthropic instance, needed for summarization
        self._api_client = api_client
        self._system_prompt_cache: dict[str, str] = {}  # hash → already cached marker
        self._summary_cache: dict[str, str] = {}  # hash(old_text) → summary text

    def process(
        self,
        messages: list[dict],
        system: Optional[str | list] = None,
        model: str = "claude-sonnet-4.6",
    ) -> tuple[list[dict], Optional[str | list], list[str]]:
        """
        Apply all enabled reduction strategies.

        Returns:
            (processed_messages, processed_system, strategies_applied)
        """
        strategies: list[str] = []
        messages = copy.deepcopy(messages)
        if system is not None:
            system = copy.deepcopy(system)

        # ── Step 1: Prompt compression ──────────────────────────────────────
        if self.config.enable_compression:
            messages, orig_tok, comp_tok = compress_messages(
                messages, aggressive=self.config.compression_aggressive
            )
            if orig_tok > comp_tok:
                strategies.append(
                    f"compression(saved~{orig_tok - comp_tok}tok)"
                )
                log.debug("Compression: %d -> %d tokens", orig_tok, comp_tok)

            # Also compress system prompt if it's a plain string
            if isinstance(system, str):
                from .utils import _compress_preserving_code
                compressed_sys = _compress_preserving_code(
                    system, self.config.compression_aggressive
                )
                if compressed_sys != system:
                    saved = estimate_tokens(system) - estimate_tokens(compressed_sys)
                    strategies.append(f"system_compression(saved~{saved}tok)")
                    system = compressed_sys

        # ── Step 2: Context trimming / summarization ─────────────────────────
        if self.config.enable_trimming and len(messages) > 0:
            total_est = messages_token_estimate(messages)
            if isinstance(system, str):
                total_est += estimate_tokens(system)

            if total_est > self.config.trim_threshold_tokens:
                messages, trimmed = self._trim_context(messages, model)
                if trimmed:
                    strategies.append("context_trim")

        # ── Step 3: Prompt caching ────────────────────────────────────────────
        from .bedrock_client import BedrockAdapter
        is_bedrock = isinstance(self._api_client, BedrockAdapter)

        if self.config.enable_caching:
            if is_bedrock:
                # Bedrock caching is handled at the adapter level via cachePoint
                # blocks on the system prompt. We just note it as a strategy.
                if getattr(self._api_client, 'enable_cache', False):
                    strategies.append("bedrock_cache_point")
            else:
                # Anthropic SDK: inject cache_control on system/messages
                system, messages, cache_applied = self._inject_cache_control(
                    system, messages
                )
                if cache_applied:
                    strategies.append("cache_control")

        return messages, system, strategies

    # -------------------------------------------------------------------------
    # Strategy implementations
    # -------------------------------------------------------------------------

    def _trim_context(
        self,
        messages: list[dict],
        model: str,
    ) -> tuple[list[dict], bool]:
        """
        Summarize conversation history older than `keep_recent_turns` turns
        when total context is large.
        """
        keep = self.config.keep_recent_turns * 2  # user + assistant pairs
        if len(messages) <= keep:
            return messages, False

        old_turns = messages[:-keep]
        recent_turns = messages[-keep:]

        # Build a text representation of old turns to summarize
        old_text_parts = []
        for msg in old_turns:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                )
            old_text_parts.append(f"{role.upper()}: {content}")

        old_text = "\n\n".join(old_text_parts)

        summary = self._summarize(old_text, model)
        if not summary:
            return messages, False

        summary_message = {
            "role": "user",
            "content": (
                "[CONTEXT SUMMARY — earlier conversation]\n"
                + summary
                + "\n[END SUMMARY — continuing from here]"
            ),
        }

        # Inject a synthetic assistant ack so the turn order stays valid
        ack_message = {
            "role": "assistant",
            "content": "Understood. Continuing with that context.",
        }

        trimmed_messages = [summary_message, ack_message] + list(recent_turns)
        log.debug(
            "Context trimmed: %d → %d messages", len(messages), len(trimmed_messages)
        )
        return trimmed_messages, True

    def _summarize(self, text: str, model: str) -> Optional[str]:
        """
        Call the summarizer model (haiku) to produce a condensed summary.
        Results are cached by content hash to avoid redundant API calls.
        Falls back gracefully if the call fails.
        """
        if self._api_client is None:
            log.warning("No API client available for summarization; skipping.")
            return None

        cache_key = hash_content(text)
        if cache_key in self._summary_cache:
            log.debug("Summary cache hit for key %s", cache_key)
            return self._summary_cache[cache_key]

        # When using a Bedrock adapter the haiku model ID won't exist; fall back
        # to whatever model the caller is already using (passed via model arg).
        from .bedrock_client import BedrockAdapter
        if isinstance(self._api_client, BedrockAdapter):
            summarizer_model = model
        else:
            summarizer_model = self.config.summarizer_model
        prompt = (
            "Summarize the following conversation history as concisely as possible, "
            "preserving all key facts, decisions, code, and context needed to continue "
            "the conversation. Be terse.\n\n"
            f"{text}"
        )

        try:
            resp = self._api_client.messages.create(
                model=summarizer_model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = resp.content[0].text
            self._summary_cache[cache_key] = summary
            return summary
        except Exception as exc:
            log.warning("Summarization failed: %s", exc)
            return None

    def _inject_cache_control(
        self,
        system: Optional[str | list],
        messages: list[dict],
    ) -> tuple[Optional[str | list], list[dict], bool]:
        """
        Inject Anthropic's cache_control: {"type": "ephemeral"} on:
        - The system prompt block (if large enough)
        - The last user message (for multi-turn conversation prefix caching)
        """
        applied = False

        # Cache the system prompt if it's large enough
        if isinstance(system, str):
            if estimate_tokens(system) >= self.config.cache_min_tokens:
                system = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                applied = True
                log.debug("Cache control applied to system prompt")
        elif isinstance(system, list):
            # Already a block list; tag the last text block
            for i in range(len(system) - 1, -1, -1):
                block = system[i]
                if isinstance(block, dict) and block.get("type") == "text":
                    if estimate_tokens(block.get("text", "")) >= self.config.cache_min_tokens:
                        system[i] = dict(block)
                        system[i]["cache_control"] = {"type": "ephemeral"}
                        applied = True
                    break

        # Cache the last user message turn (supports conversation prefix caching)
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user":
                content = msg.get("content", "")

                if isinstance(content, str):
                    tok_count = estimate_tokens(content)
                    if tok_count >= self.config.cache_min_tokens:
                        messages[i] = dict(msg)
                        messages[i]["content"] = [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                        applied = True
                        log.debug("Cache control applied to user message %d", i)
                elif isinstance(content, list) and content:
                    # Tag last text block in the content list
                    new_content = list(content)
                    for j in range(len(new_content) - 1, -1, -1):
                        block = new_content[j]
                        if isinstance(block, dict) and block.get("type") == "text":
                            if estimate_tokens(block.get("text", "")) >= self.config.cache_min_tokens:
                                new_content[j] = dict(block)
                                new_content[j]["cache_control"] = {"type": "ephemeral"}
                                applied = True
                            break
                    messages[i] = dict(msg)
                    messages[i]["content"] = new_content
                break

        return system, messages, applied
