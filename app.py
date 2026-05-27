from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import time

from face_model import get_face_emotion
from speech_model import get_speech_emotion
from fusion import fuse_emotions

app = FastAPI(title="Multimodal Emotion Detection API")

# -----------------------------
# CORS (for dashboard later)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# CONFIG
# -----------------------------
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

    start_time = time.time()

    # -----------------------------
    # SAVE IMAGE
    # -----------------------------
    image_path = os.path.join(UPLOAD_DIR, "frame.jpg")
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # -----------------------------
    # SAVE AUDIO
    # -----------------------------
    audio_path = os.path.join(UPLOAD_DIR, "voice.wav")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # -----------------------------
    # FACE EMOTION MODEL
    # -----------------------------
    try:
        face_emotion, face_conf = get_face_emotion(image_path)
    except Exception as e:
        print("Face model error:", e)
        face_emotion, face_conf = "neutral", 0.5

    # -----------------------------
    # SPEECH EMOTION MODEL
    # -----------------------------
    try:
        voice_emotion, voice_conf = get_speech_emotion(audio_path)
    except Exception as e:
        print("Speech model error:", e)
        voice_emotion, voice_conf = "neutral", 0.5

    # -----------------------------
    # FUSION ENGINE
    # -----------------------------
    try:
        result = fuse_emotions(
            (face_emotion, face_conf),
            (voice_emotion, voice_conf)
        )
    except Exception as e:
        print("Fusion error:", e)
        result = {
            "emotion": "neutral",
            "face": face_emotion,
            "voice": voice_emotion,
            "confidence": 0.5
        }

    end_time = time.time()

    # -----------------------------
    # RESPONSE (FOR RASPI + DASHBOARD)
    # -----------------------------
    return {
        "final_emotion": result["emotion"],
        "face_emotion": result["face"],
        "voice_emotion": result["voice"],
        "confidence": float(result["confidence"]),
        "processing_time_sec": round(end_time - start_time, 3)
    }