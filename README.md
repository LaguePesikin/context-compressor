# context-compressor

## Intro

A simple but effective context compressor, supports incremental context compression for LLMs with persistent anchored summaries.

Based on the algorithm from [Factory.ai](https://factory.ai/news/compressing-context), this library efficiently manages finite context windows in extended conversations and multi-step workflows.

Features:
- Incremental Updates: Only summarize newly dropped messages
- Anchor Points: Each summary is linked to a specific message turn
- Efficient Compression: Dramatically reduces computation and cost


## Installation

```bash
git clone https://github.com/LaguePesikin/context-compressor
cd context-compressor
pip install -e .
# for developers run: pip install -e ".[dev]"
```
or 
```
pip install context-compressor
```


## Quick Start

```python
from context_compressor import ContextCompressor

# Define your summarizer function
def my_summarizer(messages_text, previous_summary=None):
    # Use your LLM of choice (OpenAI, Anthropic, etc.)
    if previous_summary:
        prompt = f"Update this summary with new info:\n{previous_summary}\n\nNew: {messages_text}"
    else:
        prompt = f"Summarize: {messages_text}"
    
    # Call your LLM here
    return your_llm_call(prompt)

# Initialize compressor
compressor = ContextCompressor(
    summarizer=my_summarizer,
    t_max=8000,      # Max tokens before compression
    t_retained=6000, # Tokens to keep after compression
    t_summary=500,   # Reserved tokens for summary
)

# Add messages to your conversation
compressor.add_message("Hello, how are you?", role="user")
compressor.add_message("I'm doing well, thanks!", role="assistant")

# Get compressed context (auto-compresses if needed)
context = compressor.get_current_context()

# View statistics
stats = compressor.get_stats()
print(f"Compressions: {stats['compression_count']}")
print(f"Tokens saved: {stats['total_tokens_saved']}")
```

## Core Functionality

### `ContextCompressor`

**Parameters:**

- `summarizer`: Custom text summarization function that takes message text and optional previous summary, returns a new summary
- `t_max`: Maximum token threshold. Context compression is triggered when this limit is exceeded
- `t_retained`: Expected token count to retain after compression. The ratio `t_retained/t_max` determines the compression rate
- `t_summary`: Length of the context summary. This parameter takes effect through prompt engineering in your summarizer (if using LLM) and the `_compress` method



## Citation
Based on the approach described in: Factory.ai: [Compressing Context](https://factory.ai/news/compressing-context)
