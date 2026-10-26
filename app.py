"""SpeechLab: Streamlit app. Run with:  streamlit run app.py"""
import os
import tempfile

import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import streamlit as st

import asr
import features
import metrics

st.set_page_config(page_title="SpeechLab", page_icon="🎙️", layout="wide")
st.title("🎙️ SpeechLab")
st.caption("Automatic speech recognition, error analysis and acoustic inspection in one place.")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=1)
    language = st.text_input("Language code (blank = auto-detect)", value="")
    use_vad = st.checkbox("Voice activity detection filter", value=True)
    conf_thresh = st.slider("Low-confidence word threshold", 0.1, 0.95, 0.6, 0.05)
    st.markdown("---")
    st.markdown("**Input**")
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "flac", "ogg"])
    recorded = st.audio_input("…or record from microphone")

audio_file = uploaded or recorded
audio_path = None
if audio_file is not None:
    suffix = os.path.splitext(getattr(audio_file, "name", "rec.wav"))[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(audio_file.getvalue())
    tmp.close()
    audio_path = tmp.name
    st.audio(audio_file)

if audio_path is None:
    st.info("Upload or record audio in the sidebar to begin.")
    st.stop()

tab_tr, tab_eval, tab_ac, tab_cmp = st.tabs(
    ["📝 Transcribe", "📊 Evaluate (WER)", "🔊 Acoustic view", "⚖️ Compare models"]
)

# ---------------- transcribe ----------------
with tab_tr:
    if st.button("Transcribe", type="primary"):
        with st.spinner(f"Running Whisper-{model_size}…"):
            st.session_state["result"] = asr.transcribe(
                audio_path, size=model_size, language=language or None, vad=use_vad
            )
    result = st.session_state.get("result")
    if result:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Language", f"{result['language']} ({result['language_prob']:.0%})")
        c2.metric("Audio length", f"{result['duration_s']:.1f}s")
        c3.metric("Runtime", f"{result['runtime_s']:.1f}s")
        c4.metric("Real-time factor", f"{result['rtf']:.2f}")
        st.subheader("Transcript")
        st.write(result["text"])
        st.subheader("Segments")
        st.dataframe(
            pd.DataFrame(
                [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]}
                 for s in result["segments"]]
            ),
            use_container_width=True,
        )
        low = asr.low_confidence_words(result["segments"], conf_thresh)
        st.subheader(f"Low-confidence words ({len(low)})")
        if low:
            st.dataframe(
                pd.DataFrame([{"word": w["word"], "start": round(w["start"], 2),
                               "confidence": round(w["prob"], 2)} for w in low]),
                use_container_width=True,
            )
        else:
            st.success("No words below the threshold.")
        d1, d2 = st.columns(2)
        d1.download_button("Download .txt", result["text"], "transcript.txt")
        d2.download_button("Download .srt", asr.to_srt(result["segments"]), "transcript.srt")

# ---------------- evaluate ----------------
with tab_eval:
    st.write("Paste the ground-truth transcript to compute WER and CER against the ASR output.")
    reference = st.text_area("Reference transcript", height=140)
    result = st.session_state.get("result")
    if reference and result:
        w = metrics.wer(reference, result["text"])
        c = metrics.cer(reference, result["text"])
        k = w["counts"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("WER", f"{w['wer']:.1%}")
        m2.metric("CER", f"{c['cer']:.1%}")
        m3.metric("Substitutions", k["S"])
        m4.metric("Deletions", k["D"])
        m5.metric("Insertions", k["I"])
        st.subheader("Word alignment")
        colors = {"C": "#1b7f3b", "S": "#b45309", "D": "#b91c1c", "I": "#1d4ed8"}
        html = " ".join(
            f"<span style='color:{colors[t]};{'text-decoration:line-through;' if t == 'D' else ''}"
            f"{'font-weight:600;' if t != 'C' else ''}'>{word}</span>"
            for word, t in metrics.render_alignment(w["ops"])
        )
        st.markdown(html, unsafe_allow_html=True)
        st.caption("green = correct · orange = substitution (ref→hyp) · red strike = deletion · blue = insertion")
    elif reference:
        st.warning("Run a transcription first.")

# ---------------- acoustic ----------------
with tab_ac:
    y, sr = features.load_audio(audio_path)
    S_db, hop = features.spectrogram_db(y, sr)
    times, f0 = features.pitch_track(y, sr, hop)
    stats = features.summarize(y, sr)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean F0", f"{stats['mean_f0_hz']:.0f} Hz")
    c2.metric("Voiced frames", f"{stats['voiced_ratio']:.0%}")
    c3.metric("Pauses ≥0.3s", stats["n_pauses"])
    c4.metric("Pause time", f"{stats['pause_time_s']:.1f}s")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#4f46e5")
    axes[0].set_title("Waveform (oscillogram)")
    librosa.display.specshow(S_db, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=axes[1], cmap="magma")
    axes[1].set_title("Spectrogram (formants appear as dark horizontal bands)")
    axes[1].set_ylim(0, 5000)
    axes[2].plot(times, f0, color="#059669", lw=1.6)
    axes[2].set_title("Pitch contour (F0, prosody)")
    axes[2].set_ylabel("Hz")
    axes[2].set_xlabel("Time (s)")
    for s, e in features.find_pauses(y, sr):
        axes[0].axvspan(s, e, color="#f59e0b", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("MFCC")
    M = features.mfcc(y, sr, hop=hop)
    fig2, ax2 = plt.subplots(figsize=(11, 3))
    librosa.display.specshow(M, sr=sr, hop_length=hop, x_axis="time", ax=ax2, cmap="coolwarm")
    ax2.set_ylabel("coefficient")
    fig2.tight_layout()
    st.pyplot(fig2)

# ---------------- compare ----------------
with tab_cmp:
    st.write("Run several Whisper sizes on this clip and compare accuracy against speed.")
    ref2 = st.text_area("Reference transcript (needed for WER)", key="ref2", height=100)
    sizes = st.multiselect("Models", ["tiny", "base", "small", "medium"], default=["tiny", "base"])
    if st.button("Run comparison") and sizes:
        rows = []
        bar = st.progress(0.0)
        for i, sz in enumerate(sizes):
            r = asr.transcribe(audio_path, size=sz, language=language or None, vad=use_vad)
            row = {"model": sz, "runtime_s": round(r["runtime_s"], 2), "rtf": round(r["rtf"], 3)}
            if ref2:
                row["WER"] = round(metrics.wer(ref2, r["text"])["wer"], 4)
                row["CER"] = round(metrics.cer(ref2, r["text"])["cer"], 4)
            row["text"] = r["text"]
            rows.append(row)
            bar.progress((i + 1) / len(sizes))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        if ref2:
            st.bar_chart(df.set_index("model")[["WER", "CER"]])
        st.bar_chart(df.set_index("model")[["runtime_s"]])
