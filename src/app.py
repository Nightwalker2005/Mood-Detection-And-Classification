import cv2
from face_detection import extract_faces
from predict import predict_mood


def label_face(frame, box, mood, confidence):
    """Draw the box and write the mood + confidence above it."""
    (x, y, w, h) = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = f"{mood} ({confidence * 100:.0f}%)"
    cv2.putText(
        frame, text, (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )


def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Running — press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for face_crop, box in extract_faces(frame):
            mood, confidence = predict_mood(face_crop)
            label_face(frame, box, mood, confidence)

        cv2.imshow("Mood Detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()