# context-compressor

## Intro

A simple but effective context compressor, supports incremental context compression for LLMs with persistent anchored summaries.

Based on the algorithm from [Factory.ai](https://factory.ai/news/compressing-context), this library efficiently manages finite context windows in extended conversations and multi-step workflows.

Features:
- Incremental Updates: Only summarize newly dropped messages
- Anchor Points: Each summary is linked to a specific message turn
- Efficient Compression: Dramatically reduces computation and cost

## Diagram

![](images/diagram.png)

## Installation

```bash
# Install from PyPI
pip install context-compressor-llm

# Install from source
git clone https://github.com/LaguePesikin/context-compressor
cd context-compressor
pip install -e .
```

## Quick Start

```python
from context_compressor import ContextCompressor, TokenCounter

# Define your summarizer function
def simple_summarizer(messages_list, previous_summary=None):
    """
    Args:
        messages_list: List of dicts like [{"role": "user", "content": "..."}]
        previous_summary: Optional previous summary to build upon
    Returns:
        A summary string
    """
    summary_parts = []
    
    if previous_summary:
        summary_parts.append(f"[Previous: {previous_summary}]")
    for msg in messages_list:
        role = msg["role"]
        content = msg["content"]
        # Take first 50 chars of each message
        snippet = content[:50].replace("\n", " ")
        summary_parts.append(f"{role.upper()}: {snippet}...")
    return "\n".join(summary_parts)

# Initialize compressor
compressor = ContextCompressor(
    summarizer=simple_summarizer,
    t_max=8000,      # Max tokens before compression
    t_retained=6000, # Tokens to keep after compression
    t_summary=500,   # Reserved tokens for summary
    tokenizer=TokenCounter(
        model_name="gpt-4o",
        use_transformers=False   # Will use default tiktoken encoding
    )
)

# Add messages to your conversation
for _ in range(30):
    compressor.add_message("Hello, how are you?", role="user")
    compressor.add_message("I'm doing well, thanks!", role="assistant")

# Get compressed context (auto-compresses if needed)
context = compressor.get_current_context()

# View statistics
stats = compressor.get_stats()
print(f"Compressions: {stats['compression_count']}")
print(f"Tokens saved: {stats['total_tokens_saved']}")
```

### Expected Output

```plaintext
Warning: Summary is too long (2813 tokens).
Compressions: 1
Tokens saved: 291
```

## Core Functionality

### `ContextCompressor`

**Parameters:**

- `summarizer`: Custom text summarization function that takes message text and optional previous summary, returns a new summary. View `examples/basic_usage.llm_summarizer_example` for a fundamental implementation.
- `t_max`: Maximum token threshold. Context compression is triggered when this limit is exceeded
- `t_retained`: Expected token count to retain after compression. The ratio `t_retained/t_max` determines the compression rate
- `t_summary`: Length of the context summary. This parameter takes effect through prompt engineering in your summarizer (if using LLM) and the `_compress` method
- `tokenizer`: Custom Tokenizer (you can set`tiktoken` or `transformers.AutoTokenizer` here). See `context_compressor.tokenizer.TokenCounter` for more details.


## Citation
Based on the approach described in: Factory.ai: [Compressing Context](https://factory.ai/news/compressing-context)
