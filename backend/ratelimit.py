"""Rate limiting utility for Graphsight API."""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
LLM_RATE_LIMIT = "60/minute"
