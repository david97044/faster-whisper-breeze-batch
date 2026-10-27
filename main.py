#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from faster_whisper import WhisperModel
from tqdm import tqdm


# ==================================================
# Config
# ==================================================
FILE_TYPES = (".mp4", ".mkv", ".m4a")

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "Videos"
SRT_DIR = BASE_DIR / "SRT"
TEXT_DIR = BASE_DIR / "TEXTONLY"

MODEL_NAME = "SoybeanMilk/faster-whisper-Breeze-ASR-25"

DEVICE = "cuda"
COMPUTE_TYPE = "float16"

LANGUAGE = "zh"
BEAM_SIZE = 5


# ==================================================
# Utils
# ==================================================
def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def format_time(seconds):
    ms = int(round(seconds * 1000))
    h = ms // (3600 * 1000)
    ms %= (3600 * 1000)
    m = ms // (60 * 1000)
    ms %= (60 * 1000)
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ==================================================
# Main
# ==================================================
def main():
    print("=" * 100)
    print("Working dir:", BASE_DIR)
    print("=" * 100)

    ensure_dir(SRT_DIR)
    ensure_dir(TEXT_DIR)

    video_files = sorted([
        f for f in VIDEOS_DIR.iterdir()
        if f.suffix.lower() in FILE_TYPES
    ])

    if not video_files:
        print("[WARN] No videos found.")
        return

    print("Videos:", [v.name for v in video_files])
    print("=" * 100)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    print("[INFO] Loading model...")
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    print("[INFO] Model loaded:", MODEL_NAME)
    print("=" * 100)

    # --------------------------------------------------
    # Process videos
    # --------------------------------------------------
    for video_path in video_files:
        base_name = video_path.stem

        srt_path = SRT_DIR / f"{base_name}.srt"
        txt_path = TEXT_DIR / f"{base_name}.txt"

        print(f"\n=== Processing: {video_path.name} ===")

        segments, info = model.transcribe(
            str(video_path),
            beam_size=BEAM_SIZE,
            language=LANGUAGE,
            vad_filter=True,   # 🔥 建議加（更穩）
        )

        with open(srt_path, "w", encoding="utf-8") as srt_f, \
             open(txt_path, "w", encoding="utf-8") as txt_f:

            idx = 1

            # 🔥 generator → 即時輸出
            for seg in tqdm(segments, desc=video_path.name, unit="seg"):
                start = format_time(seg.start)
                end = format_time(seg.end)
                text = seg.text.strip()

                if not text:
                    continue

                srt_block = f"{idx}\n{start} --> {end}\n{text}\n\n"

                # write files
                srt_f.write(srt_block)
                txt_f.write(text + "\n")

                # 🔥 即時印出
                print(srt_block, end="")

                idx += 1

        print(f"[OK] SRT  : {srt_path}")
        print(f"[OK] TEXT : {txt_path}")

    print("\n[DONE] All videos processed.")


# ==================================================
# Entry
# ==================================================
if __name__ == "__main__":
    main()
