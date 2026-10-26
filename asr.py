"""ASR engine wrapper around faster-whisper with timing and caching."""
import time

_MODELS = {}


def get_model(size="base", device="cpu", compute_type="int8"):
    from faster_whisper import WhisperModel

    key = (size, device, compute_type)
    if key not in _MODELS:
        _MODELS[key] = WhisperModel(size, device=device, compute_type=compute_type)
    return _MODELS[key]


def transcribe(path, size="base", language=None, vad=True, beam_size=5):
    """Transcribe an audio file.

    Returns dict: text, language, segments[{start,end,text,words[...]}], runtime_s, rtf.
    """
    model = get_model(size)
    t0 = time.time()
    seg_iter, info = model.transcribe(
        path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad,
        word_timestamps=True,
    )
    segments = []
    for s in seg_iter:
        segments.append({
            "start": s.start,
            "end": s.end,
            "text": s.text.strip(),
            "words": [
                {"word": w.word.strip(), "start": w.start, "end": w.end, "prob": w.probability}
                for w in (s.words or [])
            ],
        })
    runtime = time.time() - t0
    text = " ".join(s["text"] for s in segments).strip()
    return {
        "text": text,
        "language": info.language,
        "language_prob": info.language_probability,
        "duration_s": info.duration,
        "segments": segments,
        "runtime_s": runtime,
        "rtf": runtime / info.duration if info.duration else 0.0,
    }


def to_srt(segments):
    def ts(t):
        h, rem = divmod(int(t * 1000), 3600000)
        m, rem = divmod(rem, 60000)
        s, ms = divmod(rem, 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{ts(s['start'])} --> {ts(s['end'])}", s["text"], ""]
    return "\n".join(lines)


def low_confidence_words(segments, threshold=0.6):
    return [w for s in segments for w in s["words"] if w["prob"] < threshold]
