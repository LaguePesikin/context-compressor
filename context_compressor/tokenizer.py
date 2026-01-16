from typing import Optional, Union, List
import tiktoken

class TokenCounter:
    """Handle token counting for different models."""
    
    def __init__(self, model_name: str, use_transformers: bool = False):
        """
        Initialize the token counter.
        
        Args:
            model_name: The name of the model to use for token counting
            use_transformers: Whether to use the transformers library for token counting
        """
        self.model = model_name
        self.use_transformers = use_transformers
        self.encoding = None
        self.tokenizer = None
        
        if use_transformers:
            self._init_transformers_tokenizer(model_name)
        else:
            self._init_tiktoken_encoding(model_name)

    def _init_tiktoken_encoding(self, model_name: str):
        """Initialize tiktoken encoding."""
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            print(f"Warning: Model '{model_name}' not recognized by tiktoken. Using cl100k_base encoding.")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def _init_transformers_tokenizer(self, model_name: str):
        """Initialize transformers tokenizer."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except ImportError:
            raise ImportError(
                "transformers library is required when use_transformers=True. "
                "Install it with: pip install transformers"
            )
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer for '{model_name}': {e}")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.use_transformers:
            return len(self.tokenizer.encode(text))
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