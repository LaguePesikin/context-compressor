"""
context-compressor: Efficient context window management for LLMs.
"""

from .compressor import ContextCompressor, Message
from .tokenizer import TokenCounter
from .summarizers import (
    TruncateSummarizer,
    HeadTailSummarizer,
    LLMSummarizer,
    create_truncate_summarizer,
    create_head_tail_summarizer,
    create_llm_summarizer,
    default_truncate,
    default_head_tail,
)

__version__ = "0.1.1"

__all__ = [
    # Core classes
    "ContextCompressor",
    "Message",
    "TokenCounter",
    # Summarizers
    "TruncateSummarizer",
    "HeadTailSummarizer",
    "LLMSummarizer",
    # Factory functions
    "create_truncate_summarizer",
    "create_head_tail_summarizer",
    "create_llm_summarizer",
    # Default instances
    "default_truncate",
    "default_head_tail",
]
