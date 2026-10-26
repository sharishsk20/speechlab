"""Command-line interface.

Examples:
  python cli.py transcribe clip.wav --model base --srt out.srt
  python cli.py evaluate clip.wav reference.txt --models tiny base small
  python cli.py wer "the cat sat" "the cat sit"
"""
import argparse
import json

import asr
import metrics


def cmd_transcribe(args):
    r = asr.transcribe(args.audio, size=args.model, language=args.language)
    print(r["text"])
    print(f"\n[{r['language']}] {r['duration_s']:.1f}s audio, {r['runtime_s']:.1f}s runtime, RTF {r['rtf']:.2f}")
    if args.srt:
        with open(args.srt, "w", encoding="utf-8") as f:
            f.write(asr.to_srt(r["segments"]))
        print(f"SRT written to {args.srt}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"JSON written to {args.json}")


def cmd_evaluate(args):
    with open(args.reference, encoding="utf-8") as f:
        ref = f.read()
    print(f"{'model':<8}{'WER':>8}{'CER':>8}{'S':>5}{'D':>5}{'I':>5}{'RTF':>8}")
    for size in args.models:
        r = asr.transcribe(args.audio, size=size, language=args.language)
        w = metrics.wer(ref, r["text"])
        c = metrics.cer(ref, r["text"])
        k = w["counts"]
        print(f"{size:<8}{w['wer']:>8.1%}{c['cer']:>8.1%}{k['S']:>5}{k['D']:>5}{k['I']:>5}{r['rtf']:>8.2f}")


def cmd_wer(args):
    w = metrics.wer(args.reference, args.hypothesis)
    c = metrics.cer(args.reference, args.hypothesis)
    print(f"WER: {w['wer']:.2%}   CER: {c['cer']:.2%}   {w['counts']}")


parser = argparse.ArgumentParser(description="SpeechLab CLI")
sub = parser.add_subparsers(dest="cmd", required=True)

p = sub.add_parser("transcribe")
p.add_argument("audio")
p.add_argument("--model", default="base")
p.add_argument("--language", default=None)
p.add_argument("--srt")
p.add_argument("--json")
p.set_defaults(func=cmd_transcribe)

p = sub.add_parser("evaluate")
p.add_argument("audio")
p.add_argument("reference", help="path to ground-truth text file")
p.add_argument("--models", nargs="+", default=["tiny", "base"])
p.add_argument("--language", default=None)
p.set_defaults(func=cmd_evaluate)

p = sub.add_parser("wer")
p.add_argument("reference")
p.add_argument("hypothesis")
p.set_defaults(func=cmd_wer)

args = parser.parse_args()
args.func(args)
