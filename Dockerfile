FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_RETRIES=10 \
    UV_HTTP_TIMEOUT=120 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 依赖层只取决于锁文件。源码变化时无需重新下载和编译全部依赖。
COPY pyproject.toml uv.lock ./

# PyStemmer publishes no Linux arm64 wheel, so Apple Silicon builds it from
# source.  Keep the compiler out of the final filesystem and persist uv's cache
# across retries so a transient registry failure does not restart every download.
RUN --mount=type=cache,target=/root/.cache/uv \
    apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && uv sync --frozen --no-dev --no-install-project \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# 项目自身是纯 Python，在依赖层之后单独安装，保持开发构建缓存稳定。
COPY README.md ./
COPY src ./src
COPY schema.sql ./schema.sql
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uvicorn", "minibrain.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
