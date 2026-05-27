from fastapi import FastAPI, UploadFile, File
import shutil
import os

app = FastAPI(title="Multimodal Emotion Detection API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# LAZY IMPORTS (VERY IMPORTANT)
# -----------------------------
def load_face_model():
    from face_model import get_face_emotion
    return get_face_emotion

def load_speech_model():
    from speech_model import get_speech_emotion
    return get_speech_emotion

from fusion import fuse_emotions


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

    image_path = os.path.join(UPLOAD_DIR, "frame.jpg")
    audio_path = os.path.join(UPLOAD_DIR, "voice.wav")

    # Save image
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Save audio
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # -----------------------------
    # LOAD MODELS ONLY WHEN NEEDED
    # -----------------------------
    face_emotion, face_conf = "neutral", 0.5
    voice_emotion, voice_conf = "neutral", 0.5

    try:
        face_model = load_face_model()
        face_emotion, face_conf = face_model(image_path)
    except Exception as e:
        print("Face model error:", e)

    try:
        speech_model = load_speech_model()
        voice_emotion, voice_conf = speech_model(audio_path)
    except Exception as e:
        print("Speech model error:", e)

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
        "final_emotion": result["emotion"],
        "face_emotion": result["face"],
        "voice_emotion": result["voice"],
        "confidence": result["confidence"]
    }