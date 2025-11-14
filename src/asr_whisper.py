# src/asr_whisper.py

import whisper
import tempfile
import soundfile as sf
import torch
from pathlib import Path

class WhisperASR:
    def __init__(self, model_name='small', device=None):
        # device: None -> auto-detect CUDA if available
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(f"[ASR] Loading Whisper '{model_name}' on {self.device} (may download weights)...")
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe(self, audio, sr, **kwargs):
        """
        Transcribe numpy audio (float32) using Whisper.
        We save a temporary WAV file and let whisper.transcribe handle resampling internally.
        """
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp.name
        tmp.close()
        sf.write(tmp_path, audio, sr)
        try:
            result = self.model.transcribe(tmp_path, **kwargs)
            text = result.get('text', '').strip()
            # Whisper does not provide a direct 'confidence' per segment in this API; use compression_ratio as a proxy if present
            conf = result.get('compression_ratio', 1.0)
            return {'text': text, 'confidence': float(conf)}
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass
