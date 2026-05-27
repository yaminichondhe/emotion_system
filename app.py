from fastapi import FastAPI, UploadFile, File
import shutil
import os

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
# MAIN PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):

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
    try:
        face_emotion, face_conf = get_face_emotion(image_path)
    except Exception as e:
        face_emotion, face_conf = "neutral", 0.5

    try:
        voice_emotion, voice_conf = get_speech_emotion(audio_path)
    except Exception as e:
        voice_emotion, voice_conf = "neutral", 0.5

    # -----------------------------
    # FUSION
    # -----------------------------
    result = fuse_emotions(
        (face_emotion, face_conf),
        (voice_emotion, voice_conf)
    )

    # -----------------------------
    # RESPONSE (RASPBERRY PI + DASHBOARD)
    # -----------------------------
    return {
        "final_emotion": result["emotion"],
        "face_emotion": result["face"],
        "voice_emotion": result["voice"],
        "confidence": result["confidence"]
    }