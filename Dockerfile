
FROM python:3.12

# Set working directory inside the container
WORKDIR /app

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/main.py ./main.py

# Copy Haar Cascade file
COPY app/haarcascade_frontalface_default.xml ./haarcascade_frontalface_default.xml

# Copy models
COPY app/models/ ./models/

# Expose the port
EXPOSE 8000

# Start the app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
