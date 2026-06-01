import os
# --- These two lines fix issues specific to your setup. ---
# They MUST come before importing deepface.
os.environ["TF_USE_LEGACY_KERAS"] = "1"        # makes DeepFace work with Keras 3
os.environ["DEEPFACE_HOME"] = "C:/deepface"    # store weights in an all-ASCII path
# -----------------------------------------------------------

from deepface import DeepFace


def predict_mood(face_crop):
    """
    Take a cropped face image and return (mood, confidence).
    - mood: "Positive", "Negative", or "Neutral"
    - confidence: a float between 0 and 1
    """
    result = DeepFace.analyze(
        face_crop,
        actions=["emotion"],
        detector_backend="skip",   # we already cropped the face ourselves
    )
    scores = result[0]["emotion"]  # 7 emotions, each a score out of 100

    # Fold DeepFace's 7 emotions into your 3 groups
    groups = {
        "Positive": scores["happy"] + scores["surprise"],
        "Negative": scores["angry"] + scores["disgust"] + scores["fear"] + scores["sad"],
        "Neutral": scores["neutral"],
    }

    mood = max(groups, key=groups.get)   # whichever group scores highest
    confidence = groups[mood] / 100.0    # turn the percentage into 0–1
    return mood, confidence