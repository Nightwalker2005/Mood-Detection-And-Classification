import os
import sys
import cv2
from face_detection import extract_faces
from predict import predict_mood

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def resize_to_fit(image, max_side=900):
    """Shrink an image so its longest side is at most max_side (keeps aspect ratio)."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:                       # only shrink, never enlarge
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def label_face(frame, box, mood, confidence):
    (x, y, w, h) = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = f"{mood} ({confidence * 100:.0f}%)"
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def process_frame(frame):
    for face_crop, box in extract_faces(frame):
        mood, confidence = predict_mood(face_crop)
        label_face(frame, box, mood, confidence)
    return frame


def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    print("Webcam running — press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Mood Detection (press q to quit)", process_frame(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_image(path, filename, show=True):
    image = cv2.imread(path)
    if image is None:
        print(f"Could not read image: {path}")
        return
    image = resize_to_fit(image)          # shrink huge photos so they fit the screen
    image = process_frame(image)
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, image)
    print(f"Saved labeled image to: {out_path}")
    if show:
        cv2.imshow("Mood Detection (press any key to close)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Could not open video: {path}")
        return
    print("Playing video — press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_to_fit(frame)      # same fit-to-screen for video
        cv2.imshow("Mood Detection (press q to quit)", process_frame(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_all_images():
    files = [f for f in os.listdir(INPUT_DIR)
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    if not files:
        print(f"No images found in '{INPUT_DIR}'.")
        return
    print(f"Processing {len(files)} image(s)...")
    for filename in files:
        run_image(os.path.join(INPUT_DIR, filename), filename, show=False)
    print(f"Done. Labeled images are in the '{OUTPUT_DIR}' folder.")


def pick_file():
    """Open a file-picker window and return the chosen path (or '')."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a photo or video",
        initialdir=os.path.abspath(INPUT_DIR),
        filetypes=[
            ("Images and videos",
             "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def dispatch(path, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in IMAGE_EXTS:
        run_image(path, filename)
    elif ext in VIDEO_EXTS:
        run_video(path)
    else:
        print(f"Unsupported file type: {ext}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(sys.argv) < 2:
        run_webcam()
        return

    arg = sys.argv[1]

    if arg.lower() in ("all", "--all"):
        run_all_images()
        return

    if arg.lower() in ("pick", "--pick"):
        path = pick_file()
        if not path:
            print("No file selected.")
            return
        dispatch(path, os.path.basename(path))
        return

    path = os.path.join(INPUT_DIR, arg)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        print(f"Put your file in the '{INPUT_DIR}' folder and pass just its name.")
        return
    dispatch(path, arg)


if __name__ == "__main__":
    main()