from transformers import pipeline
import speech_recognition as sr

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

def get_speech_emotion(audio_path):

    r = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
    except:
        return "neutral", 0.5

    result = emotion_classifier(text)[0]

    return result["label"], result["score"]