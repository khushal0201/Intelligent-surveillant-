from __future__ import annotations
import json
import logging
import sys
import time
import uuid
import contextvars
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

trace_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_id_ctx.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info).splitlines()[-1]
        for k in ("store_id", "endpoint", "latency_ms", "event_count",
                  "status_code", "method"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [h]
    root.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").handlers = [h]
    logging.getLogger("uvicorn.access").propagate = False


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tid = request.headers.get("x-trace-id") or uuid.uuid4().hex[:12]
        trace_id_ctx.set(tid)
        t0 = time.perf_counter()
        response = None
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            store_id = request.path_params.get("store_id") if hasattr(request, "path_params") else None
            logging.getLogger("api").info(
                "request",
                extra={
                    "endpoint": request.url.path,
                    "method": request.method,
                    "latency_ms": latency_ms,
                    "status_code": status,
                    "store_id": store_id,
                },
            )
            if response is not None:
                response.headers["x-trace-id"] = tid
