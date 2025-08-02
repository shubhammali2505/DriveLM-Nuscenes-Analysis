# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy the project files (based on your folder structure)
COPY data_analysis/ ./data_analysis/
COPY RAG/ ./RAG/
COPY evaluation/ ./evaluation/
COPY main.py .
COPY app.py .
COPY startup.py .

# Create necessary directories that appear in your structure
RUN mkdir -p output
RUN mkdir -p results
RUN mkdir -p visualizations
RUN mkdir -p results/evaluation

# Create analysis.log file
RUN touch analysis.log

# Create a startup script that runs startup.py
RUN echo '#!/bin/bash\necho "🚗 Starting DriveLM Analysis Suite..."\necho "📱 Access the application at http://localhost:8501"\necho "🛑 Press Ctrl+C to stop"\necho "Data expected at: /app/data/"\necho "NuScenes: /app/data/nusccens"\necho "DriveLM: /app/data/drivelm_data/train_sample.json"\necho ""\npython3 startup.py' > /app/start.sh
RUN chmod +x /app/start.sh

# Expose the Streamlit port
EXPOSE 8501

# Health check to ensure the service is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the startup script
CMD ["/bin/bash", "/app/start.sh"]