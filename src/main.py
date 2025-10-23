from __future__ import annotations

from fastapi import FastAPI

from src.core.logging_config import configure_logging

configure_logging()

from src.api.endpoints import router  # noqa: E402  (import after logging is configured)

app = FastAPI(title="Deep Thinking RAG API", version="1.0.0")
app.include_router(router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


__all__ = ["app"]
