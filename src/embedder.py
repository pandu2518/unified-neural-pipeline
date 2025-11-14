# src/embedder.py

import numpy as np

def simple_embed(audio, sr, win_ms=25, hop_ms=10):
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(audio) < win:
        energy = np.array([float((audio**2).mean())])
    else:
        frames = [audio[i:i+win] for i in range(0, len(audio)-win+1, hop)]
        energy = np.array([float((f**2).mean()) for f in frames])
    return np.array([float(energy.mean()), float(energy.std())], dtype='float32')

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    na = a / (np.linalg.norm(a) + 1e-9)
    nb = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(na, nb))
