from fastapi import FastAPI, UploadFile, File
import shutil
import os
from typing import Dict

from face_model import get_face_emotion
from speech_model import get_speech_emotion
from fusion import fuse_emotions

app = FastAPI(title="Multimodal Emotion Detection API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Emotion Detection Backend is Active"
    }


# -----------------------------
# SAFE PREDICTION WRAPPER
# -----------------------------
def safe_predict_face(image_path: str):
    try:
        return get_face_emotion(image_path)
    except Exception:
        return "neutral", 0.5


def safe_predict_voice(audio_path: str):
    try:
        return get_speech_emotion(audio_path)
    except Exception:
        return "neutral", 0.5


# -----------------------------
# MAIN PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
) -> Dict:

    # Save image
    image_path = os.path.join(UPLOAD_DIR, "frame.jpg")
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Save audio
    audio_path = os.path.join(UPLOAD_DIR, "voice.wav")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # -----------------------------
    # MODEL PREDICTIONS
    # -----------------------------
    face_emotion, face_conf = safe_predict_face(image_path)
    voice_emotion, voice_conf = safe_predict_voice(audio_path)

    # -----------------------------
    # FUSION
    # -----------------------------
    result = fuse_emotions(
        (face_emotion, face_conf),
        (voice_emotion, voice_conf)
    )

    # -----------------------------
    # RESPONSE
    # -----------------------------
    return {
        "final_emotion": result.get("emotion", "neutral"),
        "face_emotion": result.get("face", face_emotion),
        "voice_emotion": result.get("voice", voice_emotion),
        "confidence": result.get("confidence", 0.5)
    }