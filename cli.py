"""
cli.py
-------
Command-line interface for the Empathy Engine.

Usage:
  python3 cli.py "I can't believe how amazing this is!"
  python3 cli.py --demo
  python3 cli.py --file sentences.txt
  python3 cli.py "text" --tts gtts --emotion transformer
"""

import argparse
import sys
import os


DEMO_CASES = [
    ("This is the best news I have heard all year! I am absolutely thrilled!",           "joy_high"),
    ("That is pretty good I suppose.",                                                    "joy_low"),
    ("I am so frustrated. This is the third time this has happened.",                    "frustration"),
    ("I cannot believe this happened again, I need this fixed immediately!",             "frustrated_urgent"),
    ("I feel completely empty. Nothing matters anymore.",                                 "sadness"),
    ("I am scared about what might happen to my account.",                               "fear"),
    ("Wait, are you serious?! That is completely unbelievable!",                         "surprise"),
    ("Oh wow that is amazing, I love this deal, when can we start?",                    "excited_joy"),
    ("I understand, that must have been really difficult for you.",                      "empathy"),
    ("Finally! I am so relieved this is sorted out at last.",                           "relief"),
    ("Your account balance is currently one hundred and forty-two dollars.",             "neutral"),
]


def run_cli():
    parser = argparse.ArgumentParser(
        description="The Empathy Engine — emotionally-aware TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("text",    nargs="?", help="Input text to synthesize")
    parser.add_argument("--tts",   default=None, help="TTS engine: pyttsx3 | gtts | google_cloud | elevenlabs")
    parser.add_argument("--emotion",default=None, help="Emotion model: transformer")
    parser.add_argument("--output",default="outputs", help="Output directory")
    parser.add_argument("--demo",  action="store_true", help="Run all demo cases")
    parser.add_argument("--file",  default=None, help="Path to text file (one sentence per line)")
    args = parser.parse_args()

    if not args.text and not args.demo and not args.file:
        parser.print_help()
        sys.exit(1)

    from pipeline import EmpathyPipeline

    pipeline = EmpathyPipeline(
        emotion_model=args.emotion,
        tts_engine=args.tts,
        output_dir=args.output,
    )

    if args.demo:
        print("=" * 60)
        print("  EMPATHY ENGINE — DEMO RUN")
        print("=" * 60 + "\n")
        for text, label in DEMO_CASES:
            fname  = f"demo_{label}.mp3"
            result = pipeline.run(text, filename=fname)
            _print_result(result)

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"Processing {len(lines)} lines from {args.file}...\n")
        for i, line in enumerate(lines):
            result = pipeline.run(line, filename=f"line_{i+1:03d}.mp3")
            _print_result(result)

    else:
        result = pipeline.run(args.text)
        _print_result(result)
        print(f"Audio saved to: {result.audio_path}")


def _print_result(r):
    print(f"  TEXT:    {r.text[:70]}")
    print(f"  EMOTIONS (full vector):")
    top5 = sorted(r.emotion_vector.items(), key=lambda x: x[1], reverse=True)
    for emotion, score in top5:
        if score < 0.08:
            continue
        bar = "█" * int(score * 20)
        print(f"    {emotion:15s} {score:.2f} {bar}")
    print(f"  DOMINANT:  {r.emotion}  secondary={r.secondary}  intensity={r.intensity:.2f}")
    print(f"  VOICE:     rate={r.rate_percent:.0f}%  "
          f"pitch={r.pitch_hz:.0f}Hz({r.pitch_st:+.2f}st)  "
          f"vol={r.volume_db:+.1f}dB  emphasis={r.emphasis}")
    print(f"  OUTPUT:    {r.audio_path}  ({r.latency_ms}ms)")
    print()


if __name__ == "__main__":
    run_cli()
