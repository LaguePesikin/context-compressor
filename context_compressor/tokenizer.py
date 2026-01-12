from typing import Optional, Union, List
import tiktoken

class TokenCounter:
    """Handle token counting for different models."""
    
    def __init__(self, model_name: str):
        self.model = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        else:
            return len(self.encoding.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        tokens = 0
        for message in messages:
            tokens += 4
            tokens += self.count_tokens(message.get('content', ''))
            tokens += self.count_tokens(message.get('role', ''))
        tokens += 2
        return tokens


class SimpleTokenCounter:
    """Simple word-based token counter (for testing without tiktoken)."""
    
    def count_tokens(self, text: str) -> int:
        """Approximate tokens by word count * 1.3."""
        if not text:
            return 0
        return int(len(text.split()) * 1.3)
    
    def count_message_tokens(self, messages: List[dict]) -> int:
        """Count tokens in messages."""
        return sum(self.count_tokens(m.get('content', '')) for m in messages)