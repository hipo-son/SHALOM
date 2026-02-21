# SHALOM container targeting HPC/SLURM environments
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy the modern packaging files
COPY pyproject.toml ./
COPY README.md ./
COPY shalom/ shalom/

# Upgrade pip and install package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -e ".[all]"

# Entrypoint default command
CMD ["python", "-m", "pytest"]
