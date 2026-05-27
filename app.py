from fastapi import FastAPI, UploadFile, File
import shutil, os

from face_model import get_face_emotion
from speech_model import get_speech_emotion
from fusion import fuse_emotions

app = FastAPI(title="Multimodal Emotion Detection API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# STORE LATEST RESULT
# -------------------------
latest_result = {
    "face_emotion": "-",
    "voice_emotion": "-",
    "final_emotion": "-",
    "confidence": 0.0
}

# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/")
def home():
    return {"status": "running"}

# -------------------------
# GET LATEST FOR DASHBOARD
# -------------------------
@app.get("/latest")
def latest():
    return latest_result

# -------------------------
# MAIN PREDICT API
# -------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...),
                   audio: UploadFile = File(...)):

    image_path = os.path.join(UPLOAD_DIR, "frame.jpg")
    audio_path = os.path.join(UPLOAD_DIR, "voice.wav")

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    try:
        face_emotion, face_conf = get_face_emotion(image_path)
    except:
        face_emotion, face_conf = "neutral", 0.5

    try:
        voice_emotion, voice_conf = get_speech_emotion(audio_path)
    except:
        voice_emotion, voice_conf = "neutral", 0.5

    result = fuse_emotions(
        (face_emotion, face_conf),
        (voice_emotion, voice_conf)
    )

    global latest_result
    latest_result = {
        "face_emotion": face_emotion,
        "voice_emotion": voice_emotion,
        "final_emotion": result["emotion"],
        "confidence": result["confidence"]
    }

    return latest_result