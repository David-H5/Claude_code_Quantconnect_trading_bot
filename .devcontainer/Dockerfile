# Dockerfile for QuantConnect Trading Bot with Claude Code
# Provides sandboxed execution environment for autonomous development on Windows 10/WSL2

FROM python:3.12-slim-bookworm

# Install system dependencies including security tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    sudo \
    ripgrep \
    iptables \
    zsh \
    fzf \
    ca-certificates \
    gnupg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 (required for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Set up non-root user with sudo access
ARG USERNAME=node
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /workspace

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir lean quantconnect-stubs pre-commit

# Copy project files
COPY . .

# Install project in editable mode
RUN pip install --no-cache-dir -e . || true

# Create directories for backups, results, and command history
RUN mkdir -p /workspace/.backups /workspace/results /commandhistory /tmp/backtests \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME:$USERNAME /workspace /commandhistory /tmp/backtests

# Configure shell
ENV SHELL=/bin/zsh
ENV HISTFILE=/commandhistory/.bash_history

# Switch to non-root user
USER $USERNAME

# Default command
CMD ["zsh"]
