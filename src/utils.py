"""
EcoGrid AI - Utility Functions
Common helpers used across the application.
"""
import time
import logging
import functools
import hashlib
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(f"ecogrid.{name}")

def timer(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger = get_logger("timer")
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper

def generate_stable_id(*fields) -> str:
    """Generate a deterministic ID from fields for dedup."""
    combined = "|".join(str(f) for f in fields)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]

def safe_divide(numerator, denominator, default=0.0):
    """Safe division handling zero denominators."""
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator

def format_number(value, decimals=2):
    """Format a number for display."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:,.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:,.{decimals}f}K"
    return f"{value:,.{decimals}f}"

def parse_timestamp(ts, fmt=None):
    """Parse a timestamp string into datetime."""
    if isinstance(ts, datetime):
        return ts
    if fmt:
        return datetime.strptime(str(ts), fmt)
    # Try common formats
    for f in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(str(ts), f)
        except (ValueError, TypeError):
            continue
    return None
