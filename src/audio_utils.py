# src/audio_utils.py
import soundfile as sf
import numpy as np
from pathlib import Path

def load_mono(path, sr=16000):
    """
    Load WAV and return (audio: np.float32 mono, sample_rate)
    Uses a simple linear resample if the file SR != sr.
    """
    data, file_sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype('float32')
    if file_sr != sr:
        duration = len(data) / file_sr
        new_n = int(duration * sr)
        # linear interpolation resample (ok for demo)
        data = np.interp(np.linspace(0, len(data), new_n), np.arange(len(data)), data).astype('float32')
        file_sr = sr
    return data, file_sr

def save_wav(path, audio, sr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)
