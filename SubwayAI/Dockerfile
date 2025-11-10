# Dockerfile for Subway Surfers AI Agent
# Note: This is a basic setup. BlueStacks integration in Docker requires additional configuration.

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    android-tools-adb \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ADB port
EXPOSE 5555

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "agent.py", "--headless"]

# Build instructions:
# docker build -t subway-ai .
# 
# Run instructions:
# docker run -it --network host subway-ai
#
# Note: For full functionality, BlueStacks must be running on the host system
# with ADB debugging enabled and accessible on port 5555.