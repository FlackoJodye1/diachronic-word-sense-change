"""LLM client supporting OpenAI and Anthropic Claude APIs."""

import os
import logging
import time
from typing import Optional
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients with shared retry logic."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    @abstractmethod
    def _call(self, prompt: str, max_tokens: int) -> str:
        """Make a single API call and return the response text."""
        pass

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from a prompt with exponential backoff retry."""
        last_error: Exception = RuntimeError("No attempts made")
        for attempt in range(self.max_retries + 1):
            start_time = time.time()
            try:
                result = self._call(prompt, max_tokens)
                logger.debug(f"API call succeeded in {time.time() - start_time:.1f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                last_error = e
                if attempt < self.max_retries:
                    is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
                    wait_time = (2 ** attempt) * (10 if is_rate_limit else 1)
                    logger.warning(
                        f"API call failed after {elapsed:.1f}s "
                        f"(attempt {attempt+1}/{self.max_retries+1}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"API call failed after {self.max_retries+1} attempts: {e}")
        raise last_error


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        timeout_seconds: int = 120,
    ):
        super().__init__(max_retries=max_retries)
        self.model = model
        self.temperature = temperature
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key, timeout=timeout_seconds)

    def _call(self, prompt: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude API client."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        timeout_seconds: int = 120,
    ):
        super().__init__(max_retries=max_retries)
        self.model = model
        self.temperature = temperature
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        self.client = Anthropic(api_key=api_key, timeout=timeout_seconds)

    def _call(self, prompt: str, max_tokens: int) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


class LLMClient:
    """Factory class for creating LLM clients."""

    PROVIDERS = {
        "openai": OpenAIClient,
        "claude": ClaudeClient,
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        provider = provider.lower()
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Use: {list(self.PROVIDERS.keys())}")

        client_class = self.PROVIDERS[provider]
        kwargs = {"api_key": api_key, "temperature": temperature}
        if model:
            kwargs["model"] = model
        self._client = client_class(**kwargs)

        self.provider = provider
        logger.info(f"Initialized {provider} client")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from a prompt."""
        return self._client.generate(prompt, max_tokens=max_tokens)