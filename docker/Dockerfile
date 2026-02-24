# =============================================================================
# SHALOM - Multi-stage Docker image
# =============================================================================
# Build:
#   docker build -t shalom -f docker/Dockerfile .
#
# With Quantum ESPRESSO:
#   docker build --build-arg INSTALL_QE=true -t shalom-qe -f docker/Dockerfile .
#
# Run:
#   docker run --rm shalom run Si --backend vasp        # CLI mode
#   docker run --rm -i shalom mcp-server                # MCP server mode
#   docker run --rm shalom test                         # Run tests
#   docker run --rm shalom-qe run Si -b qe --calc scf --execute -np 2
# =============================================================================

# --- Stage 1: Builder --------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY shalom/ shalom/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[all]"

# --- Stage 2: Runtime --------------------------------------------------------
FROM python:3.11-slim

LABEL org.opencontainers.image.title="SHALOM" \
      org.opencontainers.image.description="System of Hierarchical Agents for Logical Orchestration of Materials" \
      org.opencontainers.image.url="https://github.com/hipo-son/SHALOM" \
      org.opencontainers.image.source="https://github.com/hipo-son/SHALOM" \
      org.opencontainers.image.licenses="MIT"

# Optional: install Quantum ESPRESSO (pass --build-arg INSTALL_QE=true)
ARG INSTALL_QE=false
RUN if [ "$INSTALL_QE" = "true" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends quantum-espresso && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY pyproject.toml README.md ./
COPY shalom/ shalom/
COPY tests/ tests/

# Install in editable mode (uses cached site-packages from builder)
RUN pip install --no-cache-dir --no-deps -e .

# Directories for DFT output and pseudopotentials
RUN mkdir -p /data/shalom-runs /data/pseudopotentials
ENV SHALOM_WORKSPACE=/data/shalom-runs
ENV SHALOM_PSEUDO_DIR=/data/pseudopotentials

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
