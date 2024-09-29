from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2

def prepare():
    print("[INFO] loading model and label binarizer...")
    global model, lb, mean
    model = load_model("Model/FrameTagging.h5")
    lb = pickle.loads(open("Model/Labels.pickle", "rb").read())
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")

def predict(videoPath: str, labels: list[str]) -> list[tuple[str, float]]:
    vs = cv2.VideoCapture(videoPath)
    (W, H) = (None, None)
    predictionStats = [(0, i) for i in range(1000)] # Счётчик предсказаний
    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        frame -= mean

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        i = np.argmax(preds)
        predictionStats[i] = (predictionStats[i][0] + 1, i)
        # Добавить куда-то в счётчик
    s = sum(x[0] for x in predictionStats)
    predictionStats.sort()
    predictionStats.reverse()
    
    results = []
    for i in range(3): 
        stats = predictionStats[i]
        results.append((labels[lb.classes_[stats[1]]], stats[0] / s))
    return results
