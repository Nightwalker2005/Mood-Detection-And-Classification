# Mood-Detection-Classification-Project

Classify moods into 3 groups — Positive, Negative, and Neutral. The system detects faces in an image or webcam feed and classifies each face's mood along with a confidence score.

## Overview

The pipeline runs in two stages:

1. **Face detection** — locate faces in an image or video frame.
2. **Mood classification** — pass each detected face to a trained model that predicts its mood and a confidence level.

Detected emotions are grouped into three categories:
- **Positive** — e.g. happy, surprised
- **Negative** — e.g. angry, sad, fearful, disgusted
- **Neutral** — neutral

## Project Structure

```
data/         # datasets (kept local, not committed)
models/       # trained model files (kept local)
notebooks/    # Jupyter notebooks for experiments
src/          # source code
  face_detection.py   # find faces in an image/frame
  preprocess.py       # prepare cropped faces for the model
  train.py            # train the emotion classifier
  predict.py          # run a face through the model -> mood + confidence
  app.py              # entry point tying it together
```

## Setup

1. Clone the repository:
```
   git clone https://github.com/Nightwalker2005/Mood-Detection-Classification-Project.git
   cd Mood-Detection-Classification-Project
```
2. Create and activate a virtual environment (Python 3.13):
```
   py -3.13 -m venv venv
   venv\Scripts\activate
```
3. Install dependencies:
```
   pip install -r requirements.txt
```

## Tech Stack

- Python 3.13
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib

## Team

- Nightwalker2005
- (Andy Sackey)