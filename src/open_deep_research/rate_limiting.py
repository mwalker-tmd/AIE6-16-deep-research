import asyncio
import time
import random
from typing import Optional, Callable, Any, TypeVar, Awaitable
from functools import wraps

T = TypeVar('T')

class RateLimiter:
    """Rate limiter that uses a token bucket algorithm."""
    
    def __init__(self, rate: float, burst: int = 1):
        """
        Initialize the rate limiter.
        
        Args:
            rate: Number of requests allowed per second
            burst: Maximum number of requests allowed in burst
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token from the bucket."""
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + time_passed * self.rate)
            self.last_update = now
            
            if self.tokens < 1:
                # Calculate wait time
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                self.last_update = time.time()  # Update last_update after sleep
            else:
                self.tokens -= 1
                # Only enforce minimum delay if we're out of burst tokens
                if self.tokens == 0:  # We just used the last token
                    min_delay = 1.0 / self.rate
                    time_since_last = time.time() - self.last_update
                    if time_since_last < min_delay:
                        await asyncio.sleep(min_delay - time_since_last)
                        self.last_update = time.time()

class RetryPolicy:
    """Retry policy with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 10.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize the retry policy.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_factor: Factor to increase delay between retries
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt."""
        delay = min(
            self.initial_delay * (self.backoff_factor ** attempt),
            self.max_delay
        )
        if self.jitter:
            delay *= (0.5 + random.random())
        return delay

def with_retry(
    retry_policy: Optional[RetryPolicy] = None,
    should_retry: Optional[Callable[[Exception], bool]] = None
):
    """
    Decorator to add retry logic to async functions.
    
    Args:
        retry_policy: Retry policy to use
        should_retry: Function to determine if an exception should trigger a retry
    """
    if retry_policy is None:
        retry_policy = RetryPolicy()
    
    default_should_retry = lambda e: isinstance(e, (asyncio.TimeoutError, ConnectionError)) or "429" in str(e)
    should_retry_fn = should_retry if should_retry is not None else default_should_retry
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(retry_policy.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retry_policy.max_retries and should_retry_fn(e):
                        delay = retry_policy.get_delay(attempt)
                        print(f"Retry {attempt + 1}/{retry_policy.max_retries} after {delay:.2f}s delay")
                        await asyncio.sleep(delay)
                    else:
                        raise
            
            raise last_exception
        
        return wrapper
    
    return decorator

def with_rate_limit(rate: float, burst: int = 1):
    """
    Decorator to add rate limiting to async functions.
    
    Args:
        rate: Number of requests allowed per second
        burst: Maximum number of requests allowed in burst
    """
    limiter = RateLimiter(rate, burst)
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            await limiter.acquire()
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator 