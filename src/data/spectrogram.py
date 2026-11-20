"""
NIPS4B WAV files -> log-mel spectrograms (the default model input).

Contract (README section 6):
    audio_to_spectrogram(path) -> float32 array (1, 128, T), values 0..1
    saved as data/processed/train/trainfile001.npy, data/processed/test/testfile0001.npy, ...

Why WAV and not the MFCC text files: MFCCs keep only 16 numbers per time step,
so a spectrogram rebuilt from them is blurry. Here librosa computes a real
log-mel spectrogram from the audio.

All settings come from configs/default.yaml (audio section):
    sample_rate, duration (pad/crop length), n_mels, n_fft, hop_length, fmin, fmax, top_db

Fixed length: every clip is padded with silence at the END (or cropped) to
exactly audio.duration seconds BEFORE the spectrogram is made, so
    T = 1 + (sample_rate * duration) // hop_length
NIPS4B clips are 1.0 to 5.004 s long. The real (unpadded) length of each clip is
written to data/processed/train_index.csv / test_index.csv.

"Light cleaning" (nothing removed for being steady, so frogs, cicadas and crickets stay in):
  - fmin cuts rumble below 100 Hz
  - dB values are floored at (loudest point - top_db)
  - scaled to 0..1: (dB + top_db) / top_db
Heavier cleaning is optional, see src/data/clean_spectrograms.py (ablation row).

Run from the project root:
    python -m src.data.spectrogram
    python -m src.data.spectrogram --override audio.duration=5.0
"""

import argparse
import re
from pathlib import Path

import librosa
import numpy as np

from src.config import ROOT, load_config

WAV_FOLDER = "NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV"


def n_frames(cfg):
    """Number of time frames T for the configured clip length."""
    a = cfg.audio
    return 1 + int(round(a.duration * a.sample_rate)) // a.hop_length


def audio_to_spectrogram(path, cfg=None):
    """WAV file -> lightly cleaned log-mel spectrogram, float32, shape (1, n_mels, T), values 0..1."""
    cfg = cfg or load_config()
    a = cfg.audio
    n_samples = int(round(a.duration * a.sample_rate))

    y, _ = librosa.load(path, sr=a.sample_rate, mono=True)
    y = y[:n_samples] if len(y) >= n_samples else np.pad(y, (0, n_samples - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y, sr=a.sample_rate, n_fft=a.n_fft, hop_length=a.hop_length,
        n_mels=a.n_mels, fmin=a.fmin, fmax=a.fmax or a.sample_rate / 2, power=2.0,
    )
    if mel.max() <= 1e-10:                      # completely silent file
        return np.zeros((1, a.n_mels, n_frames(cfg)), dtype=np.float32)

    db = librosa.power_to_db(mel, ref=np.max, top_db=a.top_db)   # [-top_db, 0]
    spec = (db + a.top_db) / a.top_db                             # [0, 1]
    return spec[np.newaxis].astype(np.float32)


def clip_id(wav_path):
    """nips4b_birds_trainfile001.wav -> trainfile001"""
    m = re.search(r"(trainfile\d+|testfile\d+)", Path(wav_path).stem, re.IGNORECASE)
    return m.group(1).lower() if m else Path(wav_path).stem


def process_split(wav_root, out_root, split, cfg):
    wavs = sorted(p for p in wav_root.rglob("*.wav") if clip_id(p).startswith(f"{split}file"))
    if not wavs:
        print(f"No {split} WAV files found under {wav_root} - check --wav-dir")
        return

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    T, sr, hop = n_frames(cfg), cfg.audio.sample_rate, cfg.audio.hop_length
    rows, cropped = ["id,duration_s,frames"], 0

    for i, wav in enumerate(wavs):
        fid = clip_id(wav)
        np.save(out_dir / f"{fid}.npy", audio_to_spectrogram(wav, cfg))
        dur = librosa.get_duration(path=wav)
        cropped += dur > cfg.audio.duration + 0.05
        rows.append(f"{fid},{dur:.4f},{min(T, 1 + int(dur * sr) // hop)}")
        if (i + 1) % 100 == 0 or i + 1 == len(wavs):
            print(f"  {split}: {i + 1}/{len(wavs)}")

    (out_root / f"{split}_index.csv").write_text("\n".join(rows) + "\n")
    print(f"{split}: {len(wavs)} files -> {out_dir}  shape (1, {cfg.audio.n_mels}, {T}) each, "
         f"{cropped} lost more than 0.05 s to cropping")


def main():
    p = argparse.ArgumentParser(description="NIPS4B WAV -> log-mel spectrograms")
    p.add_argument("--wav-dir", type=Path, default=None,
                   help=f"folder with the WAV files (default: <paths.raw_dir>/{WAV_FOLDER})")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output folder (default: paths.processed_dir)")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. audio.duration=5.0")
    args = p.parse_args()

    cfg = load_config(overrides=args.override)
    wav_root = args.wav_dir or ROOT / cfg.paths.raw_dir / WAV_FOLDER
    out_root = args.out_dir or ROOT / cfg.paths.processed_dir

    a = cfg.audio
    print(f"{a.sample_rate} Hz, pad/crop to {a.duration} s -> T = {n_frames(cfg)} frames, "
          f"{a.n_mels} mel bins")
    for split in ("train", "test"):
        process_split(wav_root, out_root, split, cfg)
    print(f"Done. Spectrograms are in {out_root}")


if __name__ == "__main__":
    main()
