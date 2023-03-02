import numpy as np


def print_landmarks(landmarks: np.ndarray):
    s = []
    for i, landmark in enumerate(landmarks.squeeze()):
        s.append(f"{tuple(landmark.round().astype(int))} #{i+1}")

    s = "[" + ", ".join(s) + "]"

    return s
