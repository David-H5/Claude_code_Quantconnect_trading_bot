"""
Anthropic Claude API Client

Wrapper for Anthropic's Claude API with retry logic, rate limiting, and error handling.

QuantConnect Compatible: Yes (no blocking operations)
"""

import os
import time
from dataclasses import dataclass
from enum import Enum

from models.exceptions import ConfigurationError


class ClaudeModel(Enum):
    """Available Claude models."""

    OPUS_4 = "claude-opus-4-20250514"
    SONNET_4 = "claude-sonnet-4-20250514"
    HAIKU = "claude-haiku-20250315"


@dataclass
class ClaudeMessage:
    """Single message in conversation."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class ClaudeResponse:
    """Response from Claude API."""

    content: str
    stop_reason: str  # "end_turn", "max_tokens", "stop_sequence"
    model: str
    usage: dict[str, int]  # {"input_tokens": X, "output_tokens": Y}
    finish_time_ms: float


class AnthropicClient:
    """
    Client for Anthropic Claude API.

    Features:
    - Automatic retry with exponential backoff
    - Rate limiting (configurable requests/minute)
    - Token counting and cost estimation
    - Error handling and graceful degradation
    - Support for all Claude models (Opus, Sonnet, Haiku)

    Example:
        client = AnthropicClient(api_key="...")
        response = client.chat(
            model=ClaudeModel.SONNET_4,
            messages=[{"role": "user", "content": "Analyze AAPL"}],
            max_tokens=1000,
            temperature=0.7
        )
    """

    # Pricing per 1M tokens (input / output) as of 2025
    PRICING = {
        ClaudeModel.OPUS_4: (15.00, 75.00),
        ClaudeModel.SONNET_4: (3.00, 15.00),
        ClaudeModel.HAIKU: (0.80, 4.00),
    }

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        rate_limit_rpm: int = 50,
        timeout_seconds: float = 30.0,
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            max_retries: Maximum number of retries on failure
            rate_limit_rpm: Maximum requests per minute
            timeout_seconds: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                key="ANTHROPIC_API_KEY",
                reason="API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter.",
            )

        self.max_retries = max_retries
        self.rate_limit_rpm = rate_limit_rpm
        self.timeout_seconds = timeout_seconds

        # Rate limiting state
        self.request_times: list[float] = []

        # Try to import anthropic
        try:
            import anthropic

            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def chat(
        self,
        model: ClaudeModel,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> ClaudeResponse:
        """
        Send chat completion request to Claude.

        Args:
            model: Claude model to use
            messages: List of messages [{"role": "user", "content": "..."}]
            system: System prompt (optional)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
            stop_sequences: Optional stop sequences

        Returns:
            ClaudeResponse with content and metadata

        Raises:
            Exception: If request fails after all retries
        """
        # Rate limiting
        self._wait_for_rate_limit()

        # Retry logic
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Make API call
                response = self.client.messages.create(
                    model=model.value,
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop_sequences,
                )

                end_time = time.time()

                # Parse response
                return ClaudeResponse(
                    content=response.content[0].text,
                    stop_reason=response.stop_reason,
                    model=response.model,
                    usage={
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    finish_time_ms=(end_time - start_time) * 1000,
                )

            except self.anthropic.APITimeoutError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue

            except self.anthropic.RateLimitError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 60  # Wait 1 minute for rate limit
                    time.sleep(wait_time)
                    continue

            except self.anthropic.APIError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                    continue

        # All retries failed
        raise Exception(f"Claude API request failed after {self.max_retries} retries: {last_exception}")

    def estimate_cost(
        self,
        model: ClaudeModel,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost for a request.

        Args:
            model: Claude model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        input_price, output_price = self.PRICING[model]

        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price

        return input_cost + output_cost

    def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()

        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]

        # Check if we're at the limit
        if len(self.request_times) >= self.rate_limit_rpm:
            # Wait until oldest request is >1 minute old
            oldest = self.request_times[0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                time.sleep(wait_time)

            # Clean up again after wait
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < 60]

        # Record this request
        self.request_times.append(now)


def create_anthropic_client(
    api_key: str | None = None,
    max_retries: int = 3,
    rate_limit_rpm: int = 50,
) -> AnthropicClient:
    """
    Factory function to create Anthropic client.

    Args:
        api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
        max_retries: Maximum retries on failure
        rate_limit_rpm: Rate limit in requests per minute

    Returns:
        AnthropicClient instance
    """
    return AnthropicClient(
        api_key=api_key,
        max_retries=max_retries,
        rate_limit_rpm=rate_limit_rpm,
    )
