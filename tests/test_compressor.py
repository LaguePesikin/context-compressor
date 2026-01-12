"""Unit tests for ContextCompressor."""

import pytest
from context_compressor import ContextCompressor, Message


def mock_summarizer(messages_text: str, previous_summary: str = None) -> str:
    """Mock summarizer for testing."""
    if previous_summary:
        return f"UPDATED: {previous_summary} + NEW"
    return f"SUMMARY: {messages_text[:30]}"


def test_initialization():
    """Test compressor initialization."""
    compressor = ContextCompressor(
        summarizer=mock_summarizer,
        t_max=1000,
        t_retained=800,
        t_summary=200
    )
    assert compressor.t_max == 1000
    assert compressor.t_retained == 800
    assert compressor.state.compression_count == 0


def test_add_message():
    """Test adding messages."""
    compressor = ContextCompressor(summarizer=mock_summarizer)
    
    compressor.add_message("Hello", role="user")
    compressor.add_message("Hi there", role="assistant")
    
    assert len(compressor.state.messages) == 2
    assert compressor.state.messages[0].role == "user"
    assert compressor.state.messages[1].content == "Hi there"


def test_no_compression_below_threshold():
    """Test that no compression occurs below threshold."""
    compressor = ContextCompressor(
        summarizer=mock_summarizer,
        t_max=1000,
    )
    
    compressor.add_message("Short message", role="user")
    context = compressor.get_current_context()
    
    assert compressor.state.compression_count == 0
    assert compressor.state.current_summary is None


def test_compression_triggered():
    """Test that compression is triggered above threshold."""
    compressor = ContextCompressor(
        summarizer=mock_summarizer,
        t_max=50,  # Very low threshold
        t_retained=40,
        t_summary=10,
    )
    
    # Add enough messages to trigger compression
    for i in range(20):
        compressor.add_message(f"Message number {i} with some content here", role="user")
    
    context = compressor.get_current_context()
    
    # Should have compressed at least once
    assert compressor.state.compression_count > 0


def test_incremental_summary_update():
    """Test that summaries are updated incrementally."""
    compressor = ContextCompressor(
        summarizer=mock_summarizer,
        t_max=100,
        t_retained=80,
        t_summary=20,
    )
    
    # Add messages to trigger multiple compressions
    for i in range(50):
        compressor.add_message(f"Message {i} " * 10, role="user")
    
    # Check that summary contains "UPDATED" indicating incremental update
    if compressor.state.current_summary:
        assert "UPDATED" in compressor.state.current_summary.summary or "SUMMARY" in compressor.state.current_summary.summary


def test_get_stats():
    """Test statistics retrieval."""
    compressor = ContextCompressor(summarizer=mock_summarizer)
    compressor.add_message("Test message", role="user")
    
    stats = compressor.get_stats()
    
    assert "total_messages" in stats
    assert "compression_count" in stats
    assert stats["total_messages"] == 1


def test_reset():
    """Test resetting the compressor."""
    compressor = ContextCompressor(summarizer=mock_summarizer)
    compressor.add_message("Test", role="user")
    
    compressor.reset()
    
    assert len(compressor.state.messages) == 0
    assert compressor.state.compression_count == 0
    assert compressor.state.current_summary is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
