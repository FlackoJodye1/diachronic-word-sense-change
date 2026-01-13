# -*- coding: utf-8 -*-
"""LLM client supporting OpenAI and Anthropic Claude APIs."""

import os
import logging
from typing import Optional
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from a prompt."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.7
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude API client."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        temperature: float = 0.7
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")

        self.client = Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text using Claude API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise


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
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM client.

        Args:
            provider: "openai" or "claude"
            model: Model name (uses provider default if not specified)
            temperature: Sampling temperature (0.0 = deterministic)
            api_key: API key (uses environment variable if not specified)
        """
        provider = provider.lower()
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Use: {list(self.PROVIDERS.keys())}")

        client_class = self.PROVIDERS[provider]
        if model:
            self._client = client_class(model=model, api_key=api_key, temperature=temperature)
        else:
            self._client = client_class(api_key=api_key, temperature=temperature)

        self.provider = provider
        logger.info(f"Initialized {provider} client")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from a prompt."""
        return self._client.generate(prompt, max_tokens=max_tokens)