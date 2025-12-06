# Dockerfile for QuantConnect Trading Bot
# Provides sandboxed execution environment for autonomous Claude Code development

FROM python:3.10-slim

# Security: Set up non-root user for sandboxed execution
RUN groupadd -r sandboxuser && \
    useradd --uid 10001 --no-log-init -r -g sandboxuser sandboxuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir lean quantconnect-stubs

# Copy project files
COPY . .

# Install project in editable mode
RUN pip install --no-cache-dir -e .

# Create directories for backups and results
RUN mkdir -p /app/.backups /app/results /tmp/backtests && \
    chown -R sandboxuser:sandboxuser /app /tmp/backtests

# Switch to non-root user
USER sandboxuser

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]
