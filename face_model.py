import cv2

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_face_emotion(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return "neutral", 0.5

    # SIMPLE heuristic (no ML dependency)
    h, w = faces[0][2], faces[0][3]

    if h * w > 20000:
        return "surprised", 0.7
    else:
        return "neutral", 0.6