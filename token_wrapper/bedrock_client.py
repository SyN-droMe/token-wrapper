"""
BedrockAdapter — wraps boto3 bedrock-runtime converse() to expose the same
interface as anthropic.Anthropic().messages.create(), so the rest of the
wrapper (pipeline, logger, reporter) works unchanged.

Usage:
    from token_wrapper.bedrock_client import BedrockAdapter

    adapter = BedrockAdapter(api_key="<your-key>", model_id="claude-sonnet-4")
    response = adapter.messages.create(
        model="claude-sonnet-4.6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.content[0].text)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import boto3

# Bedrock gateway defaults (replace with your own endpoint)
_DEFAULT_ENDPOINT = "https://bedrock-runtime.us-east-1.amazonaws.com"
_DEFAULT_REGION = "us-east-1"
_API_KEY_ENV = "AWS_BEARER_TOKEN_BEDROCK"


# ---------------------------------------------------------------------------
# Lightweight response/usage objects that mirror the Anthropic SDK shape
# so the rest of the wrapper can stay provider-agnostic.
# ---------------------------------------------------------------------------

@dataclass
class _ContentBlock:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class _Response:
    content: list[_ContentBlock]
    usage: _Usage
    model: str
    stop_reason: str = "end_turn"


# ---------------------------------------------------------------------------
# Message format conversion helpers
# ---------------------------------------------------------------------------

def _to_bedrock_messages(messages: list[dict]) -> list[dict]:
    """
    Convert Anthropic-style messages to Bedrock converse format.

    Anthropic: {"role": "user", "content": "text"}
               {"role": "user", "content": [{"type": "text", "text": "..."}]}
    Bedrock:   {"role": "user", "content": [{"text": "..."}]}
    """
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if isinstance(content, str):
            bedrock_content = [{"text": content}]
        elif isinstance(content, list):
            bedrock_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    bedrock_content.append({"text": block.get("text", "")})
                # cache_control and other Anthropic-specific keys are silently dropped
                # — Bedrock doesn't support them
        else:
            bedrock_content = [{"text": str(content)}]

        result.append({"role": role, "content": bedrock_content})
    return result


def _extract_system(
    system: Optional[str | list],
    enable_cache: bool = False,
) -> Optional[list[dict]]:
    """
    Convert Anthropic system prompt to Bedrock system format.

    If enable_cache is True, append a cachePoint block after the system text
    so Bedrock can cache the system prompt across calls (requires >= 1024 tokens).
    """
    if system is None:
        return None

    if isinstance(system, str):
        blocks: list[dict] = [{"text": system}]
    else:
        # Already a list of blocks (Anthropic format)
        blocks = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                blocks.append({"text": block.get("text", "")})

    if not blocks:
        return None

    # Append cachePoint to enable Bedrock prompt caching on system prompt
    if enable_cache:
        blocks.append({"cachePoint": {"type": "default"}})

    return blocks


# ---------------------------------------------------------------------------
# Proxy that mirrors anthropic.resources.Messages
# ---------------------------------------------------------------------------

class _BedrockMessagesProxy:
    def __init__(self, adapter: "BedrockAdapter") -> None:
        self._adapter = adapter

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: Optional[str | list] = None,
        label: Optional[str] = None,  # wrapper extra kwarg — ignored here
        **kwargs: Any,
    ) -> _Response:
        client = self._adapter._boto_client
        model_id = model  # use whatever model string is passed in

        converse_kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": _to_bedrock_messages(messages),
            "inferenceConfig": {"maxTokens": max_tokens},
        }

        # Enable cachePoint on system prompt if adapter has caching enabled
        enable_cache = self._adapter.enable_cache
        system_blocks = _extract_system(system, enable_cache=enable_cache)
        if system_blocks:
            converse_kwargs["system"] = system_blocks

        response = client.converse(**converse_kwargs)

        # Extract text content
        out_msg = response.get("output", {}).get("message", {})
        content_blocks = [
            _ContentBlock(text=b["text"])
            for b in out_msg.get("content", [])
            if "text" in b
        ]

        # Extract usage — Bedrock returns inputTokens / outputTokens
        # With prompt caching enabled, also look for cacheReadInputTokens
        # and cacheWriteInputTokens in the usage response.
        raw_usage = response.get("usage", {})
        usage = _Usage(
            input_tokens=raw_usage.get("inputTokens", 0),
            output_tokens=raw_usage.get("outputTokens", 0),
            cache_creation_input_tokens=raw_usage.get("cacheWriteInputTokens", 0),
            cache_read_input_tokens=raw_usage.get("cacheReadInputTokens", 0),
        )

        stop_reason = response.get("stopReason", "end_turn")
        return _Response(
            content=content_blocks,
            usage=usage,
            model=model_id,
            stop_reason=stop_reason,
        )


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------

class BedrockAdapter:
    """
    Drop-in replacement for anthropic.Anthropic() that uses boto3 bedrock-runtime
    under the hood. Exposes the same .messages.create() interface.

    Args:
        api_key:       Bearer token for the gateway (or set AWS_BEARER_TOKEN_BEDROCK).
        endpoint_url:  Bedrock gateway URL.
        region_name:   AWS region string.
        model_id:      Default model ID (can be overridden per-call via model=).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_url: str = _DEFAULT_ENDPOINT,
        region_name: str = _DEFAULT_REGION,
        model_id: str = "claude-sonnet-4.6",
        enable_cache: bool = True,
    ) -> None:
        resolved_key = api_key or os.environ.get(_API_KEY_ENV)
        if not resolved_key:
            raise ValueError(
                f"No API key provided. Pass api_key= or set {_API_KEY_ENV}."
            )

        # Set the env var boto3 reads for bearer token auth
        os.environ[_API_KEY_ENV] = resolved_key
        self._api_key = resolved_key

        # If using a custom gateway with api-key auth (not AWS SigV4),
        # supply dummy AWS credentials so boto3 doesn't abort at signing stage,
        # then inject the real api-key header before the request is sent.
        self._boto_client = boto3.client(
            service_name="bedrock-runtime",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id="placeholder",
            aws_secret_access_key="placeholder",
            aws_session_token=None,
        )

        # Inject "api-key" header on the prepared HTTP request just before it
        # is sent — "before-send" receives the AWSPreparedRequest with real headers.
        def _inject_api_key(request, **kwargs):
            request.headers["api-key"] = resolved_key

        self._boto_client.meta.events.register("before-send.bedrock-runtime.*", _inject_api_key)

        self.model_id = model_id
        self.enable_cache = enable_cache
        self.messages = _BedrockMessagesProxy(self)
