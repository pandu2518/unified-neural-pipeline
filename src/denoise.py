# src/denoise.py

import numpy as np

def spectral_gate(audio, sr, noise_frames=6):
    win = int(0.02 * sr)         # 20 ms frames
    hop = win // 2
    if len(audio) < win:
        return audio
    # frame array
    frames = [audio[i:i+win] for i in range(0, len(audio)-win+1, hop)]
    if len(frames) < noise_frames:
        return audio
    mags = [np.abs(np.fft.rfft(f)) for f in frames]
    noise_spec = np.mean(mags[:noise_frames], axis=0)
    out = np.zeros(len(audio) + win, dtype='float32')  # extra for overlap-add
    idx = 0
    for f in frames:
        spec = np.fft.rfft(f)
        mag = np.abs(spec)
        mask = mag > (noise_spec * 1.2)   # simple threshold
        spec = spec * mask
        rec = np.fft.irfft(spec)
        out[idx:idx+win] += rec[:win]
        idx += hop
    out = out[:len(audio)]
    mx = np.max(np.abs(out)) + 1e-9
    return (out / mx).astype('float32')
