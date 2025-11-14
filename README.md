📌 Unified Neural Pipeline (Simple Version)

This project implements a simple audio-based neural pipeline using PyTorch and Whisper for audio transcription and similarity matching.
It is designed to demonstrate clear reasoning, modular coding, and a complete end-to-end AI workflow without unnecessary complexity.

✅ Features

🔊 Audio loading & preprocessing (16 kHz mono)

✂️ Voice Activity Detection (basic energy-threshold method)

🔈 Denoising (simple spectral gating)

🧬 Embedding extraction using Whisper

🎯 Cosine-similarity–based audio matching

📝 JSON output containing detected segments

🎧 Export of matched audio snippets

📁 Project Structure

unified-neural-pipeline/

│

├── simple_pipeline.py        # Main end-to-end pipeline

├── requirements.txt          # Dependencies

│

├── src/

│   ├── audio_utils.py

│   ├── vad.py

│   ├── denoise.py

│   ├── embedder.py

│   ├── asr_whisper.py

│   └── punctuator.py

│

└── examples/

    ├── harvard.wav
    
    ├── jackhammer.wav
    
    └── out/                 # Output folder (generated)

🚀 How to Run
1. Create virtual environment
python -m venv .venv

2. Activate it
# PowerShell
.\.venv\Scripts\Activate.ps1

3. Install dependencies
pip install -r requirements.txt

4. Run the pipeline
python simple_pipeline.py \
    --mix examples/jackhammer.wav \
    --target examples/harvard.wav \
    --out examples/out \
    --sr 16000 \
    --th 0.6

🎧 Using Your Own Audio Files

Place your .wav files into the examples/ folder, then run:

python simple_pipeline.py --mix examples/<your-mix>.wav --target examples/<your-target>.wav --out examples/out

📝 Output

The pipeline generates:

diarization.json → timestamps, scores, transcription

target_*.wav → extracted matched segments

Console logs → VAD, similarity scores, ASR text

🛠️ Tech Stack

Python 3.8+

PyTorch

OpenAI Whisper

NumPy / SoundFile

Custom VAD, denoise, embedding modules

📄 Notes

This is intentionally simple, human-written code focusing on:

clarity over complexity

modular functions

reproducible results

easy evaluation of reasoning and pipeline design
