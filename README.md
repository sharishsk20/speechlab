# SpeechLab

An end-to-end speech recognition toolkit: transcribe audio with OpenAI Whisper (via `faster-whisper`), measure accuracy with a from-scratch WER/CER implementation, inspect the signal acoustically, and benchmark model sizes against each other.

## Features

| Module | What it does |
|---|---|
| `asr.py` | Whisper transcription with word-level timestamps and confidence, VAD filtering, SRT export |
| `metrics.py` | WER and CER via Levenshtein alignment, implemented from scratch (S/D/I counts, aligned word diff) |
| `features.py` | Waveform, spectrogram, MFCC, pYIN pitch contour, energy, pause detection |
| `app.py` | Streamlit UI: Transcribe, Evaluate (WER), Acoustic view, Compare models |
| `cli.py` | Command-line transcribe, evaluate and WER |
| `test_metrics.py` | Unit tests for the error-rate code |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python test_metrics.py           # sanity check
streamlit run app.py
```

The first run of each model size downloads its weights (tiny ≈ 75 MB, base ≈ 145 MB, small ≈ 480 MB). MP3/M4A input needs `ffmpeg` on PATH.

## Usage

**Web app**: upload or record audio, then use the tabs.

- *Transcribe*: transcript, timestamped segments, low-confidence words, TXT/SRT download.
- *Evaluate (WER)*: paste a reference transcript; get WER, CER, substitution/deletion/insertion counts and a colour-coded alignment.
- *Acoustic view*: waveform with pauses shaded, spectrogram (formants), F0 contour (prosody), MFCC heatmap.
- *Compare models*: tiny vs base vs small on the same clip, accuracy against real-time factor.

**CLI**

```bash
python cli.py transcribe clip.wav --model small --srt clip.srt
python cli.py evaluate clip.wav reference.txt --models tiny base small
python cli.py wer "the cat sat on the mat" "the cat sit on mat"
```

`python make_sample.py` writes a synthetic vowel clip (/a/ /i/ /u/) to try the Acoustic tab without a microphone.

## How the pieces work

**WER.** `WER = (S + D + I) / N`, where N is the number of reference words. Text is lower-cased and punctuation-stripped, then aligned by dynamic programming (edit distance) with a backtrace that labels every word correct, substituted, deleted or inserted. CER applies the same alignment to characters.

**ASR pipeline.** Audio → 16 kHz mono → optional Silero VAD to drop silence → Whisper encoder–decoder with beam search → segments and words with timestamps and probabilities. Real-time factor (RTF) is runtime divided by audio duration; below 1.0 is faster than real time.

**Acoustic view.** The spectrogram shows formant bands (F1/F2 identify vowels), pYIN tracks fundamental frequency for intonation, MFCCs are the classic ASR feature front end, and long silent gaps expose phrasing and pause structure.

## Experiment ideas for the report

1. WER against model size on clean vs noisy audio (add noise at 20/10/5 dB SNR).
2. Accent or language comparison with your own recordings, using `--language` forced vs auto-detect.
3. Do low-confidence words predict the actual errors? Compare flagged words against the alignment's S/D/I positions.
4. VAD on vs off: effect on hallucinated text during silence.
5. Speaking rate against WER, using the pause statistics from `features.summarize`.

## Project layout

```
speechlab/
├── app.py
├── asr.py
├── cli.py
├── features.py
├── metrics.py
├── make_sample.py
├── test_metrics.py
├── requirements.txt
└── README.md
```
