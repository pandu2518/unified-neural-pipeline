# src/vad.py

import numpy as np

def energy_vad(audio, sr, frame_ms=30, energy_th=0.0005, min_s=0.2):
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len // 2
    energies = []
    times = []
    for i in range(0, max(1, len(audio)-frame_len+1), hop):
        f = audio[i:i+frame_len]
        energies.append(float((f**2).mean()))
        times.append(i / sr)
    mask = [e > energy_th for e in energies]
    segments = []
    i = 0
    while i < len(mask):
        if mask[i]:
            start = times[i]
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            end = times[j-1] + frame_len / sr
            if end - start >= min_s:
                segments.append({'start': round(start, 3), 'end': round(end, 3)})
            i = j
        else:
            i += 1
    return segments
