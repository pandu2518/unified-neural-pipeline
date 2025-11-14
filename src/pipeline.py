# src/pipeline.py
import json
import os
from .audio_utils import load_mono, save_wav
from .denoise import spectral_gate
from .vad import energy_vad
from .embedder import simple_embed, cosine_sim
from .asr_whisper import WhisperASR
from .punctuator import simple_punct

def run_pipeline(mix_path, target_path, out_dir, sr=16000, match_th=0.6):
    # 1) Load
    print("[PIPE] Loading audio...")
    mix, _ = load_mono(mix_path, sr)
    target, _ = load_mono(target_path, sr)

    # 2) Denoise (simple)
    print("[PIPE] Running denoise (spectral gate)...")
    mix_clean = spectral_gate(mix, sr)

    # 3) VAD
    print("[PIPE] Running VAD...")
    segments = energy_vad(mix_clean, sr)
    print(f"[PIPE] VAD found {len(segments)} speech segments.")

    # 4) Target embedding
    ref_emb = simple_embed(target, sr)

    # 5) Load ASR model
    asr = WhisperASR(model_name='small', device=None)

    results = []
    os.makedirs(out_dir, exist_ok=True)

    for seg in segments:
        s_idx = int(seg['start'] * sr)
        e_idx = int(seg['end'] * sr)
        seg_audio = mix_clean[s_idx:e_idx]

        # 6) Match with target
        emb = simple_embed(seg_audio, sr)
        score = cosine_sim(ref_emb, emb)
        matched = score >= match_th
        print(f"[PIPE] Segment {seg['start']:.2f}-{seg['end']:.2f} score={score:.3f} matched={matched}")

        if not matched:
            continue

        # 7) Transcribe matched segment
        trans = asr.transcribe(seg_audio, sr)
        text = simple_punct(trans.get('text', ''))

        # 8) Save chunk
        fname = f"target_{int(seg['start']*1000)}_{int(seg['end']*1000)}.wav"
        out_file = os.path.join(out_dir, fname)
        save_wav(out_file, seg_audio, sr)

        results.append({
            'speaker': 'Target',
            'start': seg['start'],
            'end': seg['end'],
            'text': text,
            'confidence': float(trans.get('confidence', score)),
            'score': round(score, 3),
            'file': out_file
        })

    # 9) Write JSON
    out_json = os.path.join(out_dir, "diarization.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[PIPE] Done. Wrote {len(results)} entries to {out_json}")
    return results
