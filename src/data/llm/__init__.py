"""LLM-based text generation for sense representations."""

from .client import LLMClient
from .text_generator import SenseTextGenerator

__all__ = ['LLMClient', 'SenseTextGenerator']
