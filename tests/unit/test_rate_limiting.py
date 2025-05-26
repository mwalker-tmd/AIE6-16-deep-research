import pytest
import asyncio
import time
from src.open_deep_research.rate_limiting import RateLimiter, RetryPolicy, with_retry, with_rate_limit

@pytest.mark.asyncio
async def test_rate_limiter():
    """Test that rate limiter properly limits request rate."""
    limiter = RateLimiter(rate=2.0, burst=2)  # 2 requests per second, burst of 2
    
    # First two requests should go through immediately
    start = time.time()
    await limiter.acquire()
    await limiter.acquire()
    first_two_time = time.time() - start
    assert first_two_time < 0.1  # Should be very fast
    
    # Third request should be delayed
    start = time.time()
    await limiter.acquire()
    third_time = time.time() - start
    assert third_time >= 0.5  # Should be delayed by at least 0.5 seconds

@pytest.mark.asyncio
async def test_retry_policy():
    """Test that retry policy properly handles retries with exponential backoff."""
    policy = RetryPolicy(
        max_retries=2,
        initial_delay=0.1,
        max_delay=1.0,
        backoff_factor=2.0,
        jitter=False
    )
    
    # Test delay calculation
    assert policy.get_delay(0) == 0.1  # Initial delay
    assert policy.get_delay(1) == 0.2  # First retry
    assert policy.get_delay(2) == 0.4  # Second retry
    assert policy.get_delay(3) == 0.8  # Should be capped at max_delay

@pytest.mark.asyncio
async def test_with_retry_decorator():
    """Test that retry decorator properly retries failed operations."""
    attempts = 0
    
    @with_retry(RetryPolicy(max_retries=2, initial_delay=0.1))
    async def failing_function():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("Temporary failure")
        return "success"
    
    result = await failing_function()
    assert result == "success"
    assert attempts == 3

@pytest.mark.asyncio
async def test_with_rate_limit_decorator():
    """Test that rate limit decorator properly limits function calls."""
    call_times = []
    
    @with_rate_limit(rate=2.0, burst=2)
    async def test_function():
        call_times.append(time.time())
        return "success"
    
    # First two calls should go through immediately
    start = time.time()
    await test_function()
    await test_function()
    first_two_time = time.time() - start
    assert first_two_time < 0.1  # Should be very fast
    
    # Third and fourth calls should be delayed
    await test_function()
    await test_function()
    total_time = time.time() - start
    
    # Should take at least 1 second due to rate limiting
    assert total_time >= 1.0
    
    # Check that calls after burst were properly spaced
    for i in range(2, len(call_times)):  # Only check calls after burst
        time_diff = call_times[i] - call_times[i-1]
        assert time_diff >= 0.5  # Should be at least 0.5 seconds between calls

@pytest.mark.asyncio
async def test_retry_with_custom_should_retry():
    """Test that retry decorator respects custom should_retry function."""
    attempts = 0
    
    def should_retry(e: Exception) -> bool:
        return isinstance(e, ValueError)
    
    @with_retry(should_retry=should_retry)
    async def test_function():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("Should retry")
        raise TypeError("Should not retry")
    
    with pytest.raises(TypeError):
        await test_function()
    assert attempts == 3  # Should retry twice for ValueError, then fail on TypeError 