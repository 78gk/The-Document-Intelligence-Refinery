# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for pdfplumber (if any, usually just python is fine)
# but for some OCR features it might need more.
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements and install them
# Since we have a pyproject.toml, we can use pip install .
COPY pyproject.toml .
RUN pip install .

# Copy the rest of the application code
COPY . .

# Environment variables for OpenRouter
ENV OPENROUTER_API_KEY=""

# Command to run the application
ENTRYPOINT ["python", "main.py"]
