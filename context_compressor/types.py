from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

@dataclass
class Message:
    content: str
    role: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: Optional[int] = None

@dataclass
class AnchoredSummary:
    """Represents a summary anchored to a specific message index."""

    summary: str
    anchor_index: int  # Index of the last message covered by this summary
    token_count: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def __repr__(self):
        return f"AnchoredSummary(anchor={self.anchor_index}, tokens={self.token_count})"

@dataclass
class CompressionState:
    """Current state of the compressed conversation."""
    
    messages: List[Message]
    current_summary: Optional[AnchoredSummary] = None
    compression_count: int = 0
    total_tokens_saved: int = 0
    
    def get_anchor_index(self) -> int:
        """Get the current anchor index, or -1 if no summary exists."""
        return self.current_summary.anchor_index if self.current_summary else -1
    
    def get_messages_after_anchor(self) -> List[Message]:
        """Get all messages after the current anchor."""
        anchor_idx = self.get_anchor_index()
        return self.messages[anchor_idx + 1:] if anchor_idx >= 0 else self.messages
    
    def total_token_count(self) -> int:
        """Calculate total tokens in current compressed state."""
        summary_tokens = self.current_summary.token_count if self.current_summary else 0
        message_tokens = sum(m.token_count or 0 for m in self.get_messages_after_anchor())
        return summary_tokens + message_tokens