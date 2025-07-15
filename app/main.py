from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_and_crop_face(pil_image):
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("No face detected, using original resized image")
        return pil_image.resize(IMG_SIZE)
    x, y, w, h = faces[0]
    face_img = cv_image[y:y+h, x:x+w]
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return face_pil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = (200, 200)
CLASS_LABELS = {0: "Male", 1: "Female"}
class_labels_emotion = ['angry','disgust','fear','happy','neutral','sad','surprise']

model_emotion = load_model('models/emotion_detection_model_50epochs_2.h5', compile=False)
model_gender = load_model('models/gender_model_50epochs.h5', compile=False)
model_age = load_model('models/age_model_50epochs.h5', compile=False)

@app.get("/")
async def root():
    return {"message": "FastAPI model service is running"}

@app.post("/predict_emotion")
async def predict_emotion(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    face_img = detect_and_crop_face(img)
    face_img = face_img.convert("L").resize((48, 48))
    img_array = np.array(face_img) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)
    prediction = model_emotion.predict(img_array)
    predicted_class = class_labels_emotion[np.argmax(prediction)]
    return {"predicted_emotion": predicted_class}

@app.post("/predict_gender")
async def predict_gender(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    face_img = detect_and_crop_face(img).resize(IMG_SIZE)
    img_array = np.array(face_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model_gender.predict(img_array)
    probability = float(pred[0][0])
    predicted_class = int(probability > 0.5)
    return {"gender": CLASS_LABELS.get(predicted_class, "Unknown"), "probability": probability}

@app.post("/predict_age")
async def predict_age(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    face_img = detect_and_crop_face(img).resize(IMG_SIZE)
    img_array = np.array(face_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model_age.predict(img_array)
    predicted_age = int(pred[0][0])
    return {"predicted_age": predicted_age}

@app.post("/predict_all")
async def predict_all(file: UploadFile = File(...)):
    image_bytes = await file.read()
    gender = await predict_gender(image_bytes)
    emotion = await predict_emotion(image_bytes)
    age = await predict_age(image_bytes)
    return {
        "gender": gender,
        "emotion": emotion,
        "age": age
    }
