FROM python:3.12

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gdown

RUN mkdir models && \
    gdown --id 1HFdqSelxrQz9Y4oJVa2_y0Ha5W-2p6Tx -O models/age_model_50epochs.h5 && \
    gdown --id 1jtk7nqH-y9e6ODg3-oGjziAhkIKMh4-A -O models/emotion_detection_model_50epochs_2.h5 && \
    gdown --id 1_3u6-tufhVkQCymUl9jDJGyOCrnTjcKb -O models/gender_model_50epochs.h5

COPY app/main.py ./main.py
COPY app/haarcascade_frontalface_default.xml ./haarcascade_frontalface_default.xml

EXPOSE 8080

ENV CUDA_VISIBLE_DEVICES=""

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
