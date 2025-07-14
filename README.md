# 👥 Face Analysis API

> A FastAPI server for face detection, cropping, and multi-task prediction of **emotion**, **age**, and **gender** using separate trained CNN models.

---

## 📑 Table of Contents

- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)


---

## 📖 About

This project is a production-ready API that:

- Detects faces in photos using OpenCV.
- Crops and preprocesses them automatically.
- Predicts **emotion**, **age**, and **gender** in one request using **three separate models**.
- Returns all predictions in a single JSON response.

It is perfect for integration into apps for user profiling, photography, surveillance, or entertainment.

---

## ✨ Features

✅ Face detection with OpenCV Haar cascades  
✅ Automatic face cropping and resizing  
✅ Three separate deep learning models:  
- Emotion model (48×48 grayscale)  
- Age model (200×200 RGB)  
- Gender model (200×200 RGB)  
✅ Single endpoint for all predictions  
✅ Easy retraining and model replacement  

---

## 🗂️ Project Structure

face_analysis_api/
│
├── app/
│ ├── main.py # FastAPI app
│ ├── utils.py # Face detection & preprocessing
│ └── models/
│ ├── emotion_model.h5
│ ├── age_model.h5
│ └── gender_model.h5
│
├── training/
│ ├── train_emotion_model.py
│ ├── train_age_model.py
│ └── train_gender_model.py
│
├── requirements.txt
└── README.md