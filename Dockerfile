########################################
# DEV STAGE
########################################
FROM python:3.12-slim AS dev

# --- Install system dependencies ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# --- Set working dir & venv ---
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_LINK_MODE=copy

# --- Copy dependency metadata ---
COPY pyproject.toml uv.lock* ./

# --- Install all dev dependencies into .venv ---
RUN uv sync --dev --no-install-project

COPY src ./src
RUN uv pip install -e .

# --- Auto-activate venv for bash ---
RUN echo 'source /app/.venv/bin/activate' >> /root/.bashrc

# --- Keep container alive for interactive shell ---
COPY docker/jupyter_lab_cfg.py /etc/jupyter/

CMD ["tail", "-f", "/dev/null"]

