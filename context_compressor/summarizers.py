"""
Built-in summarizers for context compression.

This module provides common summarization strategies that can be used
with ContextCompressor out of the box.
"""

from typing import List, Dict, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class TruncateSummarizer:
    """
    Summarizer that truncates each message to a fixed character length.
    
    Example:
        >>> summarizer = TruncateSummarizer(max_chars=50)
        >>> summary = summarizer(messages, previous_summary)
    """
    
    def __init__(
        self, 
        max_chars: int = 50,
        ellipsis: str = "...",
        include_previous: bool = True,
        separator: str = "\n"
    ):
        """
        Initialize the truncate summarizer.
        Args:
            max_chars: Maximum characters to keep from each message
            ellipsis: String to append after truncation
            include_previous: Whether to include previous summary
            separator: String to join message summaries
        """
        self.max_chars = max_chars
        self.ellipsis = ellipsis
        self.include_previous = include_previous
        self.separator = separator
    
    def __call__(
        self, 
        messages_list: List[Dict[str, str]], 
        previous_summary: Optional[str] = None
    ) -> str:
        """
        Summarize messages by truncating each to fixed length.
        Args:
            messages_list: List of message dicts with 'role' and 'content'
            previous_summary: Optional previous summary to build upon
        Returns:
            Summary string with truncated messages
        """
        summary_parts = []
        
        if self.include_previous and previous_summary:
            summary_parts.append(f"[Previous: {previous_summary}]")
        for msg in messages_list:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            cleaned = content.replace("\n", " ").strip()
            if len(cleaned) > self.max_chars:
                truncated = cleaned[:self.max_chars] + self.ellipsis
            else:
                truncated = cleaned
            summary_parts.append(f"{role.upper()}: {truncated}")
        return self.separator.join(summary_parts)

class HeadTailSummarizer:
    """
    Summarizer that keeps first N and last M messages, with ellipsis in between.
    
    Example:
        >>> summarizer = HeadTailSummarizer(head_count=3, tail_count=2)
        >>> summary = summarizer(messages, previous_summary)
    """
    
    def __init__(
        self,
        head_count: int = 3,
        tail_count: int = 2,
        middle_placeholder: str = "\n[... {count} messages omitted ...]\n",
        include_previous: bool = True,
        separator: str = "\n"
    ):
        """
        Initialize the head-tail summarizer.
        Args:
            head_count: Number of messages to keep from the beginning
            tail_count: Number of messages to keep from the end
            middle_placeholder: Placeholder text for omitted messages
            include_previous: Whether to include previous summary
            separator: String to join messages
        """
        self.head_count = head_count
        self.tail_count = tail_count
        self.middle_placeholder = middle_placeholder
        self.include_previous = include_previous
        self.separator = separator
    
    def __call__(
        self, 
        messages_list: List[Dict[str, str]], 
        previous_summary: Optional[str] = None
    ) -> str:
        """
        Summarize messages by keeping head and tail.
        Args:
            messages_list: List of message dicts with 'role' and 'content'
            previous_summary: Optional previous summary to build upon
        Returns:
            Summary string with head and tail messages
        """
        summary_parts = []
        if self.include_previous and previous_summary:
            summary_parts.append(f"[Previous: {previous_summary}]")
        total_messages = len(messages_list)
        if total_messages <= (self.head_count + self.tail_count):
            for msg in messages_list:
                summary_parts.append(self._format_message(msg))
        else:
            for msg in messages_list[:self.head_count]:
                summary_parts.append(self._format_message(msg))
            omitted_count = total_messages - self.head_count - self.tail_count
            placeholder = self.middle_placeholder.format(count=omitted_count)
            summary_parts.append(placeholder)
            for msg in messages_list[-self.tail_count:]:
                summary_parts.append(self._format_message(msg))
        return self.separator.join(summary_parts)
    
    def _format_message(self, msg: Dict[str, str]) -> str:
        """Format a single message for display."""
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        return f"{role.upper()}: {content}"

class LLMSummarizer:
    """
    Summarizer that uses an LLM API to intelligently summarize messages.
    Supports OpenAI, Anthropic, and custom API clients.
    Example:
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> summarizer = LLMSummarizer(
        ...     client=client,
        ...     model="gpt-4o-mini",
        ...     max_tokens=300
        ... )
        >>> summary = summarizer(messages, previous_summary)
    """
    
    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        max_tokens: int = 300,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        include_previous: bool = True,
        api_type: str = "openai"
    ):
        """
        Initialize the LLM summarizer.
        
        Args:
            client: API client instance (e.g., OpenAI(), Anthropic())
            model: Model name to use for summarization
            max_tokens: Maximum tokens in the summary
            temperature: Sampling temperature (lower = more focused)
            system_prompt: Custom system prompt (uses default if None)
            include_previous: Whether to include previous summary in context
            api_type: Type of API ("openai", "anthropic", or "custom")
        """
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.include_previous = include_previous
        self.api_type = api_type.lower()
        
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that creates concise summaries of conversations. "
            "Capture the key points, decisions, and context while being extremely brief. "
            "Aim for clarity and completeness in under 200 words."
        )
    
    def __call__(
        self, 
        messages_list: List[Dict[str, str]], 
        previous_summary: Optional[str] = None
    ) -> str:
        """
        Summarize messages using an LLM API.
        
        Args:
            messages_list: List of message dicts with 'role' and 'content'
            previous_summary: Optional previous summary to build upon
            
        Returns:
            LLM-generated summary string
        """
        # Build context
        context = self._build_context(messages_list, previous_summary)
        
        # Call appropriate API
        if self.api_type == "openai":
            return self._call_openai(context)
        elif self.api_type == "anthropic":
            return self._call_anthropic(context)
        else:
            raise ValueError(f"Unsupported api_type: {self.api_type}")
    
    def _build_context(
        self, 
        messages_list: List[Dict[str, str]], 
        previous_summary: Optional[str]
    ) -> str:
        """Build context string from messages and previous summary."""
        parts = []
        
        if self.include_previous and previous_summary:
            parts.append(f"Previous context: {previous_summary}\n")
        
        parts.append("Conversation to summarize:")
        for msg in messages_list:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(parts)
    
    def _call_openai(self, context: str) -> str:
        """Call OpenAI API for summarization."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context},
                    {"role": "user", "content": "Please provide a concise summary."}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            # Fallback to simple truncation
            return context[:500] + "..."
    
    def _call_anthropic(self, context: str) -> str:
        """Call Anthropic API for summarization."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": f"{context}\n\nPlease provide a concise summary."}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return context[:500] + "..."


def create_truncate_summarizer(**kwargs) -> TruncateSummarizer:
    """
    Create a truncate summarizer with custom settings.
    
    Args:
        **kwargs: Arguments to pass to TruncateSummarizer
        
    Returns:
        Configured TruncateSummarizer instance
    """
    return TruncateSummarizer(**kwargs)


def create_head_tail_summarizer(**kwargs) -> HeadTailSummarizer:
    """
    Create a head-tail summarizer with custom settings.
    
    Args:
        **kwargs: Arguments to pass to HeadTailSummarizer
        
    Returns:
        Configured HeadTailSummarizer instance
    """
    return HeadTailSummarizer(**kwargs)


def create_llm_summarizer(client: Any, **kwargs) -> LLMSummarizer:
    """
    Create an LLM summarizer with custom settings.
    
    Args:
        client: API client instance
        **kwargs: Arguments to pass to LLMSummarizer
        
    Returns:
        Configured LLMSummarizer instance
    """
    return LLMSummarizer(client=client, **kwargs)


default_truncate = TruncateSummarizer(max_chars=50)
default_head_tail = HeadTailSummarizer(head_count=3, tail_count=2)