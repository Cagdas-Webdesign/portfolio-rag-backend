# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage: resolve dependencies from the lockfile into a self-contained venv
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first: this layer is cached until the lockfile changes.
# --no-install-project keeps application code out of the dependency layer,
# --no-dev keeps test and lint tooling out of the runtime image entirely.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------------------
# Runtime stage: interpreter + virtualenv, no build tools, no source checkout
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# Non-root by default. The application writes nothing to disk.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORTFOLIO_RAG_ENVIRONMENT=production

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# The knowledge base is data the service serves, not build input: it is read at
# startup so a retrieved passage can be resolved back to its text. Markdown
# only — no credentials, nothing generated.
COPY --chown=app:app knowledge/ ./knowledge/

USER app

# Documentation only, and the default the CMD below falls back to. A platform
# that injects PORT (Cloud Run does, with 8080) overrides it at runtime.
EXPOSE 8000

# No secrets are baked in: every setting comes from the environment at runtime.
# Override PORTFOLIO_RAG_ALLOWED_ORIGINS with the real client origin(s).

# The port is read from the environment for the same reason as in the CMD: a
# probe against a fixed 8000 would report unhealthy whenever the platform picked
# a different one. (Cloud Run ignores this directive and runs its own probes.)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=2).status == 200 else 1)"]

# Production start command. One worker: the process is I/O bound and a process
# manager or the platform handles replication.
#
# Through a shell on purpose: the port has to be expanded at runtime, because a
# container that ignores an injected PORT never passes a platform's startup
# probe (Cloud Run injects 8080). `exec` hands PID 1 to uvicorn, so SIGTERM
# reaches it directly and a scale-to-zero shutdown stays graceful rather than
# being killed after the grace period. Written as JSON so no second shell is
# interposed. The default keeps `docker run -p 8000:8000` working unchanged.
CMD ["sh", "-c", "exec uvicorn portfolio_rag.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
