"""
Context Compressor - Incremental context compression for LLMs.

Based on Factory.ai's approach to managing finite context windows
with persistent anchored summaries.
"""

__version__ = "0.1.0"

from .compressor import ContextCompressor
from .types import Message, AnchoredSummary, CompressionState
from .tokenizer import TokenCounter, SimpleTokenCounter

__all__ = [
    "ContextCompressor",
    "Message",
    "AnchoredSummary",
    "CompressionState",
    "TokenCounter",
    "SimpleTokenCounter",
]
