from context_compressor import ContextCompressor, TokenCounter
import requests
import os
import traceback

def simple_summarizer_example(messages_text: str, previous_summary: str = None) -> str:
    """
    Simple summarizer example.
    In production sceneio please replace this with an actual LLM call.
    """
    if previous_summary:
        return f"{previous_summary}\n\nAdditional context: {messages_text[:100]}..."
    else:
        return f"Summary: {messages_text[:100]}..."


def simple_summarizer_example_main():
    compressor = ContextCompressor(
        summarizer=simple_summarizer_example,
        t_max=1000,      # Trigger compression at 1000 tokens
        t_retained=800,  # Keep 800 tokens after compression
        t_summary=200,   # Reserve 200 tokens for summary
        tokenizer=TokenCounter("gpt-4o", use_transformers=False)
    )
    
    print("=== Starting Conversation ===\n")
    compressor.add_message("Hello, I need help with Python programming.", role="user")
    compressor.add_message("I'd be happy to help! What specific topic?", role="assistant")
    compressor.add_message("I want to learn about decorators.", role="user")
    compressor.add_message("Decorators are functions that modify other functions...", role="assistant")
    
    for i in range(20):
        compressor.add_message(
            f"This is message {i} with some content to increase token count. " * 20,
            role="user" if i % 2 == 0 else "assistant"
        )
    
    context = compressor.get_current_context()
    
    print(f"\n=== Current Context ({len(context)} messages) ===\n")
    for msg in context:
        print(f"[{msg.role}] {msg.content[:80]}...")
        print()
    
    stats = compressor.get_stats()
    print("\n=== Compression Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


def llm_summarizer_example():
    """Example with actual LLM integration."""

    T_SUMMARY = 150

    def llm_summarizer(messages_text: str, previous_summary: str = None) -> str:
        """Use LLM to create summaries."""

        if previous_summary:
            prompt = f"""Previous summary:\n\n{previous_summary}\n\nNew messages to incorporate:\n\n{messages_text}\n\nCreate an updated summary that combines the previous summary with the new information.\n\nIMPORTANT: Keep the summary under {T_SUMMARY} tokens. Be concise but include key points."""
        else:
            prompt = f"""Summarize the following conversation:\n\n{messages_text}\n\nProvide a concise summary of the key points.\n\nIMPORTANT: Keep the summary under {T_SUMMARY} tokens. Be concise but include key points."""
        try:
            response = requests.post(
                os.getenv("OPENAI_API_BASE") + "/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemini-3-flash-preview",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8192
                },
                timeout=15
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM call failed: {e}")
            traceback.print_exc()
            # return a simple summary of the messages
            return f"Summary: {messages_text[:100]}..."
        
    compressor = ContextCompressor(
        summarizer=llm_summarizer,
        t_max=600,
        t_retained=400,
        t_summary=T_SUMMARY,
    )
    
    test_conversation = [
        {"role": "user", "content": "Hello, I want to learn Python programming. Where should I start?"},
        {"role": "assistant", "content": "Hello! I'd be happy to help. I recommend starting with the basics: variables, data types (strings, lists, dictionaries), conditional statements, and loops. I suggest using Python 3.x. Do you have any prior programming experience?"},
        {"role": "user", "content": "I'm a complete beginner. I've never programmed before."},
        {"role": "assistant", "content": "No problem! Python is one of the best languages for beginners. Here's what I suggest: 1) Install Python 3.11 or newer on your computer; 2) Learn to use an IDE - I recommend VSCode or PyCharm; 3) Start practicing with variables and print statements. Would you like me to recommend some learning resources?"},
        {"role": "user", "content": "Yes, what books or video tutorials should I look at?"},
        {"role": "assistant", "content": "I recommend several resources: For books, 'Python Crash Course' by Eric Matthes is excellent. For videos, check out Corey Schafer's Python tutorials on YouTube. The most important thing is hands-on practice - write code to verify each concept you learn. You can also try simple problems on LeetCode to practice."},
        {"role": "user", "content": "I've now learned the basic syntax. What Python data structures should I learn next?"},
        {"role": "assistant", "content": "Great progress! The core Python data structures include: 1) Lists - mutable ordered sequences; 2) Tuples - immutable sequences; 3) Dictionaries - key-value pair mappings; 4) Sets - unordered collections of unique elements. These are the most commonly used. I recommend practicing them through actual projects."},
        {"role": "user", "content": "What's the difference between lists and tuples? When should I use each one?"},
        {"role": "assistant", "content": "The key difference is mutability: lists can be modified (add, remove, change elements), while tuples cannot be changed after creation. Use cases: use tuples for fixed, unchanging data (like coordinate points (x, y)), and lists for dynamic data (like a to-do list). Tuples are also more memory-efficient due to immutability and can be used as dictionary keys."},
        {"role": "user", "content": "What about dictionaries? I see them used a lot but don't fully understand their advantages."},
        {"role": "assistant", "content": "Dictionaries are one of Python's most powerful data structures! Their advantages include: 1) Fast value lookup by key with O(1) time complexity; 2) Intuitive key-value pair representation; 3) Perfect for storing structured data, like a person's information: {'name': 'John', 'age': 25, 'city': 'NYC'}. They're especially useful when working with JSON data and configuration files."},
        {"role": "user", "content": "I want to build a website with Python. What framework should I learn?"},
        {"role": "assistant", "content": "There are three mainstream Python web frameworks: 1) Django - a full-featured 'heavyweight' framework with built-in ORM, admin backend, and user authentication; 2) Flask - a lightweight microframework that's flexible but requires assembling components yourself; 3) FastAPI - a modern framework with great performance and automatic API documentation. For beginners, I recommend starting with Flask."},
        {"role": "user", "content": "Between Flask and Django, which is better for building a blog website?"},
        {"role": "assistant", "content": "For a blog, Django is more suitable! Here's why: 1) Django has a powerful built-in ORM for easy database operations; 2) It includes an admin backend so you don't have to build a management interface; 3) The user authentication system works out of the box; 4) The template system is mature. While Flask is more flexible, for common applications like blogs, Django's 'batteries included' philosophy helps you get started faster."},
    ]

    print("\n=== Starting to add messages to context ===\n")
    for i, message in enumerate(test_conversation, 1):
        print(f"Adding message {i}/{len(test_conversation)}: [{message['role']}] {message['content'][:50]}...")
        compressor.add_message(message["content"], role=message["role"])
        
        stats = compressor.get_stats()
        print(f"Current tokens: {stats['current_tokens']}/{stats['t_max']}")
        
        if stats['compression_count'] > 0:
            print(f"Compression triggered {stats['compression_count']} times")
        print()
    
    context = compressor.get_current_context()
    
    print("\n" + "="*80)
    print(f"=== Final context ({len(context)} messages) ===")
    print("="*80 + "\n")
    
    for i, msg in enumerate(context, 1):
        role_display = msg.role.upper()
        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        
        if msg.metadata.get("type") == "summary":
            print(f"[{role_display}] (summary)")
            print(f"{content_preview}")
        else:
            print(f"{i}. [{role_display}]")
            print(f"{content_preview}")
        
        print(f"Tokens: {msg.token_count}")
        print()
    
    stats = compressor.get_stats()
    print("\n" + "="*80)
    print("=== Compression statistics ===")
    print("="*80)
    print(f"Total messages: {stats['total_messages']}")
    print(f"Current tokens: {stats['current_tokens']}")
    print(f"Compression count: {stats['compression_count']}")
    print(f"Saved tokens: {stats['total_tokens_saved']}")
    print(f"Has summary: {stats['has_summary']}")
    print(f"Anchor index: {stats['anchor_index']}")
    print(f"T_max threshold: {stats['t_max']}")
    print(f"T_retained threshold: {stats['t_retained']}")
    
    if stats['compression_count'] > 0:
        compression_ratio = (stats['total_tokens_saved'] / 
                           (stats['current_tokens'] + stats['total_tokens_saved']) * 100)
        print(f"Compression ratio: {compression_ratio:.1f}%")
    

if __name__ == "__main__":
    simple_summarizer_example_main()
    print("\n" + "="*50 + "\n")
    llm_summarizer_example()
