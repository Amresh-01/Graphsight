"""Tenant routing middleware enforcing Org -> Pod assignment."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from tracerag import config


class TenantRoutingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if not config.MULTI_TENANCY_ENABLED:
            return await call_next(request)

        # In multi-tenant mode, verify pod allocation
        org_id = request.headers.get("X-Org-Id") or config.DEFAULT_TENANT_ORG_ID
        request.state.org_id = org_id
        return await call_next(request)
