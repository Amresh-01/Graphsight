"""Middleware package initialization."""

from .routing import TenantRoutingMiddleware

__all__ = ["TenantRoutingMiddleware"]
