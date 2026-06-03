import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"      # match predict.py (DeepFace + Keras 3)
os.environ["DEEPFACE_HOME"] = "C:/deepface"  # store weights in an ASCII-only path

import cv2
from deepface import DeepFace

# Detector DeepFace uses. "yunet" is fast and handles angled/multiple faces well.
# Alternatives: "retinaface" (most accurate but slower), "ssd", "mtcnn", "opencv".
DETECTOR = "yunet"


def detect_faces(image):
    """Detect faces with DeepFace. Returns a list of (x, y, w, h) boxes."""
    try:
        found = DeepFace.extract_faces(
            image,
            detector_backend=DETECTOR,
            enforce_detection=False,   # don't crash when no face is found
            align=False,
        )
    except Exception:
        return []

    boxes = []
    H, W = image.shape[:2]
    for f in found:
        if f.get("confidence", 1) <= 0:     # skip the "no face found" fallback
            continue
        a = f["facial_area"]
        x, y, w, h = a["x"], a["y"], a["w"], a["h"]
        if w >= W and h >= H:               # skip whole-image fallback too
            continue
        boxes.append((x, y, w, h))
    return boxes


def draw_faces(image, faces):
    """Draw a green rectangle around each detected face."""
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


def extract_faces(image, padding=0):
    """
    Detect faces and return a list of (face_crop, box) pairs.
    - face_crop: the cut-out face image, to send to the mood model
    - box: the (x, y, w, h) location, for drawing/labeling later
    """
    results = []
    H, W = image.shape[:2]
    for (x, y, w, h) in detect_faces(image):
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, W)
        y2 = min(y + h + padding, H)
        if x2 > x1 and y2 > y1:
            results.append((image[y1:y2, x1:x2], (x, y, w, h)))
    return results


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        exit()
    print("Webcam running — press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = draw_faces(frame, detect_faces(frame))
        cv2.imshow("Face Detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()