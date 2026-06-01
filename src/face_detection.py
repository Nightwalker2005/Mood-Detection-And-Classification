import cv2

# Load OpenCV's pre-trained face detector (ships with opencv-python).
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(image):
    """Detect faces in a BGR image. Returns boxes as (x, y, w, h)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,      # was 5 — higher = fewer false detections
        minSize=(80, 80),    # was (30, 30) — ignore small background patches
    )
    return faces


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
    faces = detect_faces(image)
    results = []
    for (x, y, w, h) in faces:
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, image.shape[1])
        y2 = min(y + h + padding, image.shape[0])
        face_crop = image[y1:y2, x1:x2]
        results.append((face_crop, (x, y, w, h)))
    return results


# Quick standalone demo: run this file directly to test detection.
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
        faces = detect_faces(frame)
        frame = draw_faces(frame, faces)
        cv2.imshow("Face Detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()