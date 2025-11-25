import asyncio
import logging
import traceback

LOG = logging.getLogger(__name__)


def async_retry(max_retries: int = 10, delay: float = 0.5):
    """
    Decorator for retrying async functions.

    Args:
        max_retries: how many attempts before failing.
        delay: base delay (in seconds); we use simple linear backoff.

    Behavior:
        - Logs each failure.
        - Waits delay * attempt_number between retries.
        - After max_retries, raises RuntimeError chained with the last error
          so the real cause is visible in logs.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    LOG.error("Agent attempt %s failed: %r", attempt, e)
                    LOG.debug("Traceback:\n%s", traceback.format_exc())
                    await asyncio.sleep(delay * attempt)
            # Surface underlying cause
            raise RuntimeError(
                f"Agent failed after {max_retries} attempts"
            ) from last_err
        return wrapper
    return decorator


# Backwards compatibility alias for any old imports using retry_async
def retry_async(max_retries: int = 10, delay: float = 0.5):
    return async_retry(max_retries=max_retries, delay=delay)
