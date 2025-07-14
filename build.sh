#!/bin/bash

echo "🚀 Installation des dépendances..."
pip install -r requirements.txt

echo "✅ Installation de gdown pour Google Drive"
pip install gdown

echo "📁 Création du dossier models..."
mkdir -p models

echo "📥 Téléchargement des modèles .h5 depuis Google Drive..."
gdown --id 1HFdqSelxrQz9Y4oJVa2_y0Ha5W-2p6Tx -O models/age_model_50epochs.h5
gdown --id 1jtk7nqH-y9e6ODg3-oGjziAhkIKMh4-A -O models/emotion_detection_model_50epochs_2.h5
gdown --id 1_3u6-tufhVkQCymUl9jDJGyOCrnTjcKb -O models/gender_model_50epochs.h5

echo "✅ Tous les fichiers .h5 ont été téléchargés dans le dossier models/"
