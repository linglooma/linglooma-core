import time
import logging

logging.basicConfig(level=logging.INFO)


def log_execution_time(func):
    """Decorator to log execution time of a function"""

    async def wrapper(*args, **kwargs):
        start_time = time.monotonic()
        result = await func(*args, **kwargs)
        end_time = time.monotonic()
        logging.info(f"⏳ {func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper
