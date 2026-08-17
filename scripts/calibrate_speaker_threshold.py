"""Re-derive SPEAKER_MATCH_THRESHOLD from real recordings.

The shipped default (`app.config.SPEAKER_MATCH_THRESHOLD`) was measured on
LibriSpeech validation-clean: studio microphones, read prose, calm voices.
That is the easy case. Real conditions — a laptop mic across a room, a
television in the background, a voice that is tired, hoarse, crying, or
whispering — push same-speaker similarity *down*, and a threshold calibrated
on clean speech will start rejecting the very turns that matter most.

Run this before trusting verification with real users.

## Recording a corpus

One WAV per utterance, 16 kHz mono, named ``<speaker>_<n>.wav``:

    corpus/
      ada_0.wav  ada_1.wav  ada_2.wav  ada_3.wav
      bo_0.wav   bo_1.wav   bo_2.wav

At least 3 speakers with 3 utterances each, and **deliberately include the
hard cases for the enrolled speaker**: tired, upset, whispering, unwell,
across the room, with a TV on. Those are what set the false-rejection rate,
and a corpus of only calm clear speech will tell you a comfortable lie.

## Reading the output

The two errors are not symmetric, so do not simply take the EER point:

- **FRR** (false rejection) — the real user's turn scored as somebody else.
  Costs them memory formation on that turn, and does so most often when
  they are distressed. This is the error to minimise.
- **FAR** (false acceptance) — another person's speech scored as the user.
  Costs a wrong memory, which is visible and deletable in Settings.

Pick the highest threshold whose FRR is 0 across every hard case, then step
down one notch for headroom. Do not pick the EER crossover.

Usage:
    python scripts/calibrate_speaker_threshold.py corpus/ [--enroll 2]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="directory of <speaker>_<n>.wav files")
    parser.add_argument(
        "--enroll", type=int, default=2, help="utterances per speaker used for enrollment"
    )
    parser.add_argument(
        "--seconds", type=float, default=6.0, help="seconds of each utterance to use"
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("Needs numpy and soundfile: pip install soundfile", file=sys.stderr)
        return 1

    from app.voice.embedder import SpeakerEmbedder, cosine_similarity

    embedder = SpeakerEmbedder()
    if not embedder.available:
        print(
            f"Speaker model not found at {embedder.model_path}.\n"
            "Run scripts/fetch_voice_models.py first.",
            file=sys.stderr,
        )
        return 1

    files = sorted(args.corpus.glob("*.wav"))
    if not files:
        print(f"No .wav files in {args.corpus}", file=sys.stderr)
        return 1

    by_speaker: dict[str, list] = defaultdict(list)
    for path in files:
        audio, rate = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16000:
            print(f"  skipping {path.name}: {rate} Hz, need 16000", file=sys.stderr)
            continue
        embedding = embedder.embed(audio[: int(rate * args.seconds)])
        if embedding is None:
            print(f"  skipping {path.name}: too short to embed", file=sys.stderr)
            continue
        by_speaker[path.name.rsplit("_", 1)[0]].append((path.name, embedding))

    usable = {k: v for k, v in by_speaker.items() if len(v) > args.enroll}
    if len(usable) < 2:
        print(
            f"Need at least 2 speakers with more than {args.enroll} usable utterances; "
            f"found {len(usable)}.",
            file=sys.stderr,
        )
        return 1

    print(f"\n{len(usable)} speakers: " + ", ".join(f"{k}({len(v)})" for k, v in usable.items()))

    centroids = {}
    for speaker, entries in usable.items():
        stacked = np.vstack([e for _, e in entries[: args.enroll]])
        mean = stacked.mean(axis=0)
        centroids[speaker] = mean / max(float(np.linalg.norm(mean)), 1e-8)

    same, cross = [], []
    for speaker, entries in usable.items():
        for name, embedding in entries[args.enroll :]:
            same.append((speaker, name, cosine_similarity(embedding, centroids[speaker])))
            for other in centroids:
                if other != speaker:
                    cross.append((name, other, cosine_similarity(embedding, centroids[other])))

    same_scores = np.array([s for _, _, s in same])
    cross_scores = np.array([s for _, _, s in cross])

    print(f"\nsame-speaker  n={len(same_scores):3d}  min={same_scores.min():.3f}  "
          f"mean={same_scores.mean():.3f}")
    print(f"other-speaker n={len(cross_scores):3d}  max={cross_scores.max():.3f}  "
          f"mean={cross_scores.mean():.3f}")

    print("\nWorst same-speaker scores (these set your false-rejection rate):")
    for speaker, name, score in sorted(same, key=lambda r: r[2])[:5]:
        print(f"  {score:.3f}  {name} vs enrolled {speaker}")

    print("\n  threshold |  FRR (reject real user) |  FAR (accept other)")
    print("  ----------+-------------------------+--------------------")
    for threshold in [round(x, 2) for x in np.arange(0.20, 0.75, 0.05)]:
        frr = float((same_scores < threshold).mean())
        far = float((cross_scores >= threshold).mean())
        marker = "  <- current default" if abs(threshold - _current_default()) < 1e-9 else ""
        print(f"     {threshold:.2f}   |         {frr:6.3f}          |       {far:6.3f}{marker}")

    safe = [t for t in np.arange(0.20, 0.75, 0.01) if (same_scores < t).mean() == 0.0]
    if safe:
        print(f"\nHighest threshold with zero false rejections: {max(safe):.2f}")
        print(f"Suggested default (one notch down for headroom): {max(max(safe) - 0.05, 0.2):.2f}")
    else:
        print("\nNo threshold rejected zero real-user utterances — enrollment is probably poor.")
    print("\nRead the docstring before acting on this: do not pick the EER crossover.")
    return 0


def _current_default() -> float:
    from app.config import SPEAKER_MATCH_THRESHOLD

    return SPEAKER_MATCH_THRESHOLD


if __name__ == "__main__":
    raise SystemExit(main())
