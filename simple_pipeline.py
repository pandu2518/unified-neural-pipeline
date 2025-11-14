# simple_pipeline.py
import argparse
import pathlib
from src.pipeline import run_pipeline

def main():
    p = argparse.ArgumentParser(description="Simple Unified Neural Pipeline (offline)")
    p.add_argument('--mix', required=True, help="Path to mixture audio (wav)")
    p.add_argument('--target', required=True, help="Path to target speaker sample (wav)")
    p.add_argument('--out', required=True, help="Output folder (will be created)")
    p.add_argument('--sr', type=int, default=16000, help="Target sample rate (default 16000)")
    p.add_argument('--th', type=float, default=0.6, help="Matching threshold (0..1)")
    args = p.parse_args()

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    results = run_pipeline(args.mix, args.target, args.out, sr=args.sr, match_th=args.th)
    print(f"Done. Found {len(results)} matched segments. Results written to {args.out}")

if __name__ == "__main__":
    main()
