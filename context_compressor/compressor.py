from typing import List, Optional, Callable, Union
from .types import Message, AnchoredSummary, CompressionState
from .tokenizer import TokenCounter, SimpleTokenCounter


class ContextCompressor:
    """
    Implements incremental context compression with anchored summaries.
    
    Based on the algorithm from Factory.ai:
    https://factory.ai/news/compressing-context
    """
    
    def __init__(
        self,
        summarizer: Callable[[str, Optional[str]], str],
        t_max: int = 8000,
        t_retained: int = 6000,
        t_summary: int = 500,
        tokenizer: Optional[Union[TokenCounter, SimpleTokenCounter]] = None,
    ):
        """
        Initialize the context compressor.
        
        Args:
            summarizer: Function that takes (messages_text, optional_previous_summary) 
                       and returns a summary string
            t_max: Maximum token threshold before compression
            t_retained: Target tokens to retain after compression
            t_summary: Expected token count for summaries
            tokenizer: Custom tokenizer (defaults to SimpleTokenCounter)
        """
        self.summarizer = summarizer
        self.t_max = t_max
        self.t_retained = t_retained
        self.t_summary = t_summary
        self.tokenizer = tokenizer or SimpleTokenCounter()
        
        self.state = CompressionState(messages=[])
    
    def add_message(self, content: str, role: str = "user", metadata: dict = None) -> None:
        """
        Add a new message to the conversation.
        
        Args:
            content: Message content
            role: Message role (user, assistant, system)
            metadata: Optional metadata dict
        """
        token_count = self.tokenizer.count_tokens(content)
        message = Message(
            content=content,
            role=role,
            metadata=metadata or {},
            token_count=token_count
        )
        self.state.messages.append(message)
    
    def get_current_context(self, auto_compress: bool = True) -> List[Message]:
        """
        Get the current context, optionally applying compression if needed.
        Args:
            auto_compress: Whether to automatically compress if over threshold
        Returns:
            List of messages in the current compressed state
        """
        if auto_compress:
            self._compress_if_needed()
        return self._build_current_context()
    
    def _build_current_context(self) -> List[Message]:
        """Build the current context from summary + messages."""
        context = []
        
        if self.state.current_summary:
            summary_msg = Message(
                content=self.state.current_summary.summary,
                role="system",
                metadata={"type": "summary", "anchor_index": self.state.current_summary.anchor_index},
                token_count=self.state.current_summary.token_count
            )
            context.append(summary_msg)
        
        context.extend(self.state.get_messages_after_anchor())
        
        return context
    
    def _compress_if_needed(self) -> bool:
        """
        Check if compression is needed and perform it.
        Returns:
            True if compression was performed, False otherwise
        """
        current_tokens = self.state.total_token_count()
        
        if current_tokens <= self.t_max:
            return False  # No compression necessary
        
        # Trigger compression
        self._perform_compression()
        return True
    
    def _perform_compression(self) -> None:
        """Perform the compression algorithm as described in the paper."""
        anchor_idx = self.state.get_anchor_index()
        messages_after_anchor = self.state.get_messages_after_anchor()
        
        # Step 2.1: Find longest suffix that fits within T_retained - T_summary
        target_tokens = self.t_retained - self.t_summary
        suffix_start_idx = self._find_suffix_start(messages_after_anchor, target_tokens)
        
        if suffix_start_idx == 0:
            print("Warning: Cannot compress further without losing all context")
            return
        
        new_anchor_global_idx = anchor_idx + suffix_start_idx
        new_anchor_local_idx = suffix_start_idx - 1
        
        # Step 2.2: Create/update summary for the dropped prefix
        messages_to_summarize = messages_after_anchor[:suffix_start_idx]
        new_summary_text = self._create_or_update_summary(messages_to_summarize)
        
        new_summary_tokens = self.tokenizer.count_tokens(new_summary_text)
        
        # Calculate tokens saved
        old_tokens = sum(m.token_count or 0 for m in messages_to_summarize)
        tokens_saved = old_tokens - new_summary_tokens
        
        # Step 2.3 & 2.4: Persist the new summary
        self.state.current_summary = AnchoredSummary(
            summary=new_summary_text,
            anchor_index=new_anchor_global_idx,
            token_count=new_summary_tokens
        )
        
        self.state.compression_count += 1
        self.state.total_tokens_saved += max(0, tokens_saved)

    
    def _find_suffix_start(self, messages: List[Message], target_tokens: int) -> int:
        """
        Find the start index of the longest suffix that fits within target_tokens.
        Args:
            messages: List of messages to analyze
            target_tokens: Maximum tokens for the suffix
        Returns:
            Index where the suffix starts (0 means keep all messages)
        """
        if not messages:
            return 0
        
        cumulative_tokens = 0
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = messages[i].token_count or 0
            if cumulative_tokens + msg_tokens > target_tokens:
                return i + 1
            cumulative_tokens += msg_tokens
        
        return 0
    
    def _create_or_update_summary(self, new_messages: List[Message]) -> str:
        """
        Create a new summary or update existing summary with new messages.
        Args:
            new_messages: Messages to summarize
        Returns:
            Summary text
        """
        messages_text = self._format_messages_for_summary(new_messages)
        
        previous_summary = None
        if self.state.current_summary:
            previous_summary = self.state.current_summary.summary
        
        summary = self.summarizer(messages_text, previous_summary)
        
        summary_tokens = self.tokenizer.count_tokens(summary)

        if summary_tokens > self.t_summary:
            print(f"Warning: Summary is too long ({summary_tokens} tokens).")

        return summary
    
    def _format_messages_for_summary(self, messages: List[Message]) -> str:
        """Format messages into a readable text for summarization."""
        lines = []
        for msg in messages:
            lines.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(lines)
    
    def get_stats(self) -> dict:
        """Get compression statistics."""
        return {
            "total_messages": len(self.state.messages),
            "current_tokens": self.state.total_token_count(),
            "compression_count": self.state.compression_count,
            "total_tokens_saved": self.state.total_tokens_saved,
            "has_summary": self.state.current_summary is not None,
            "anchor_index": self.state.get_anchor_index(),
            "t_max": self.t_max,
            "t_retained": self.t_retained,
        }
    
    def reset(self) -> None:
        """Reset the compressor state."""
        self.state = CompressionState(messages=[])
