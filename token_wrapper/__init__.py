"""
token_wrapper — A drop-in token reduction wrapper for the Anthropic Claude API.

Usage:
    from token_wrapper import TokenWrapperClient

    client = TokenWrapperClient(api_key="sk-ant-...")
    response = client.messages.create(
        model="claude-sonnet-4.6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    client.print_report()
"""

from .client import TokenWrapperClient
from .reporter import UsageReporter
from .logger import UsageLogger

__all__ = ["TokenWrapperClient", "UsageReporter", "UsageLogger"]
__version__ = "1.0.0"
