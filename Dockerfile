FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown for downloading large model files
RUN pip install gdown

# Create models folder and download models from Google Drive
RUN mkdir models && \
    gdown --id 1HFdqSelxrQz9Y4oJVa2_y0Ha5W-2p6Tx -O models/age_model_50epochs.h5 && \
    gdown --id 1jtk7nqH-y9e6ODg3-oGjziAhkIKMh4-A -O models/emotion_detection_model_50epochs_2.h5 && \
    gdown --id 1_3u6-tufhVkQCymUl9jDJGyOCrnTjcKb -O models/gender_model_50epochs.h5

# Copy app files
COPY app/main.py ./main.py
COPY app/haarcascade_frontalface_default.xml ./haarcascade_frontalface_default.xml

# Expose the port your FastAPI app will run on
EXPOSE 8080

# Run the app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
