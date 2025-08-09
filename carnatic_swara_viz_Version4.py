"""
Carnatic Swara Visualizer + Mini-Games (Single File)
----------------------------------------------------

What this does:
- Load audio from YouTube (yt-dlp + ffmpeg) or local upload
- Detect pitch (pYIN / YIN fallback), auto-estimate Sa (tonic) or manual Sa
- Map notes to swaras (12-TET approximation) or a custom scale (e.g., a Pann approximation)
- Visualize swaras over time (2D lanes) and explore a 3D scatter
- Rough phrase detection to segment the performance
- Play mini-games to learn/validate swaras and transitions

Setup:
1) Create venv:
   python -m venv .venv && source .venv/bin/activate
2) Install deps:
   pip install -r requirements.txt
3) Ensure ffmpeg is available (installed via packages.txt on Streamlit Cloud)
4) Run:
   streamlit run carnatic_swara_viz.py

Notes:
- Pitch tracking is best on monophonic lead melodies.
- Swara mapping uses 12-TET as a practical approximation.
- Custom scale mode lets you enter labels and semitone offsets for Pann-like explorations.
"""

import os
import io
import uuid
import math
import random
import shutil
import tempfile
import subprocess
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import plotly.express as px
import scipy.signal as sps
import soundfile as sf
import librosa
import librosa.effects
import streamlit as st

# --------------------------
# Defaults and Constants
# --------------------------
DEFAULT_SWARA_ORDER = ["S","R1","R2","G2","G3","M1","M2","P","D1","D2","N2","N3","S'"]
DEFAULT_ET_SEMITONES = {
    "S": 0, "R1": 1, "R2": 2, "G2": 3, "G3": 4, "M1": 5, "M2": 6,
    "P": 7, "D1": 8, "D2": 9, "N2": 10, "N3": 11, "S'": 12
}

# --------------------------
# Helper: Download YouTube Audio
# --------------------------
def download_audio_from_youtube(url: str, out_dir: str) -> str:
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp not found in PATH. Install it (pip or pipx).")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH.")

    base = os.path.join(out_dir, str(uuid.uuid4()))
    cmd1 = ["yt-dlp", "-x", "--audio-format", "m4a", "-o", f"{base}.%(ext)s", url]
    subprocess.run(cmd1, check=True)

    media = None
    for ext in (".m4a", ".mp3", ".webm", ".opus"):
        cand = f"{base}{ext}"
        if os.path.exists(cand):
            media = cand
            break
    if media is None:
        for f in os.listdir(out_dir):
            if f.startswith(os.path.basename(base)) and f.endswith((".m4a",".mp3",".webm",".opus")):
                media = os.path.join(out_dir, f)
                break
    if media is None:
        raise RuntimeError("Could not locate downloaded audio file.")

    wav_path = f"{base}.wav"
    cmd2 = ["ffmpeg", "-y", "-i", media, "-ac", "1", "-ar", "22050", wav_path]
    subprocess.run(cmd2, check=True)
    return wav_path

# --------------------------
# Pitch Extraction
# --------------------------
def load_audio_mono(path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = librosa.effects.preemphasis(y)
    y = librosa.util.normalize(y)
    return y, sr

def _next_pow2(n: int, min_val: int = 256, max_val: int = 16384) -> int:
    n = max(min_val, int(n))
    v = 1 << (n - 1).bit_length()
    return min(v, max_val)

def estimate_pitch_track(
    y: np.ndarray,
    sr: int,
    fmin: float = 75.0,
    fmax: float = 800.0,
    frame_length_ms: float = 20.0,
    hop_length_ms: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_length = int(sr * frame_length_ms / 1000.0)
    hop_length = max(1, int(sr * hop_length_ms / 1000.0))
    frame_length = _next_pow2(frame_length)
    try:
        f0, _, voiced_prob = librosa.pyin(
            y, fmin=fmin, fmax=fmax,
            frame_length=frame_length,
            hop_length=hop_length
        )
        conf = np.nan_to_num(voiced_prob, nan=0.0)
    except Exception:
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, frame_length=frame_length, hop_length=hop_length)
        conf = np.ones_like(f0, dtype=float) * 0.5

    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    return f0, conf, times

def detect_phrases(
    y: np.ndarray,
    sr: int,
    hop_length_ms: float = 10.0,
    smooth_ms: float = 120.0,
    thresh: float = 0.4
) -> List[Tuple[float, float]]:
    hop = max(1, int(sr * hop_length_ms / 1000.0))
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    env = S.mean(axis=0)
    win = int((smooth_ms/1000.0) * (sr/hop))
    if win > 3:
        env = sps.medfilt(env, kernel_size=win if win % 2 == 1 else win + 1)
    env = (env - env.min()) / (env.ptp() + 1e-9)
    active = env > thresh
    segments = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            t0 = i * hop / sr
            t1 = j * hop / sr
            if t1 - t0 > 0.25:
                segments.append((round(t0, 3), round(t1, 3)))
            i = j
        else:
            i += 1
    return segments

# --------------------------
# Scale / Swara Tables
# --------------------------
def build_scale_table(sa_hz: float, labels: List[str], semitone_map: Dict[str, int]) -> List[Dict[str, Any]]:
    rows = []
    for i, lbl in enumerate(labels):
        semi = int(semitone_map[lbl])
        hz = sa_hz * (2 ** (semi / 12.0))
        rows.append({"label": lbl, "semitones": semi, "ref_hz": hz, "order": i})
    return rows

def estimate_tonic_sa(f0_hz: np.ndarray) -> float:
    hz = np.array(f0_hz)
    hz = hz[np.isfinite(hz) & (hz > 0)]
    if len(hz) < 50:
        return 132.0
    cents = (1200.0 * np.log2(hz / hz.mean())) % 1200.0
    hist, _ = np.histogram(cents, bins=12, range=(0,1200))
    peak_bin = np.argmax(hist)
    mean_hz = np.median(hz)
    sa = mean_hz / (2 ** (peak_bin / 12.0))
    while sa < 100: sa *= 2
    while sa > 300: sa /= 2
    return float(sa)

def map_freq_to_scale_rows(
    f0_hz: np.ndarray,
    conf: np.ndarray,
    times: np.ndarray,
    scale_table: List[Dict[str, Any]],
    snap_thresh_cents: float = 70.0
) -> List[Dict[str, Any]]:
    rows = []
    for f, c, t in zip(f0_hz, conf, times):
        if not np.isfinite(f) or f <= 0:
            rows.append({"time_s": float(t), "freq_hz": np.nan, "conf": float(c),
                         "label": None, "order": None, "octave": None})
            continue
        best = None
        for octv in range(-2, 5):
            for sw in scale_table:
                ref = sw["ref_hz"] * (2 ** octv)
                cents = abs(1200.0 * np.log2(f / ref))
                if best is None or cents < best[0]:
                    best = (cents, sw["label"], sw["order"], octv)
        if best and best[0] <= snap_thresh_cents:
            rows.append({"time_s": float(t), "freq_hz": float(f), "conf": float(c),
                         "label": best[1], "order": int(best[2]), "octave": int(best[3])})
        else:
            rows.append({"time_s": float(t), "freq_hz": float(f), "conf": float(c),
                         "label": None, "order": None, "octave": None})
    return rows

# --------------------------
# Visualization
# --------------------------
def draw_2d_lane_plot(df: pd.DataFrame, lane_labels: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, label in enumerate(lane_labels):
        ax.axhline(i, linewidth=0.6, alpha=0.3)
        ax.text(-0.5, i, label, va='center', ha='right', fontsize=9)
    mask = df["order"].notna()
    if mask.any():
        t = df.loc[mask, "time_s"].values
        lane = df.loc[mask, "order"].values.astype(int)
        conf = df.loc[mask, "conf"].values
        segments = []
        x_last, y_last = None, None
        for tt, ll in zip(t, lane):
            if x_last is not None and ll == y_last:
                segments.append([(x_last, y_last), (tt, ll)])
            x_last, y_last = tt, ll
        if segments:
            lc = LineCollection(segments, linewidths=2, alpha=0.6)
            ax.add_collection(lc)
        ax.scatter(t, lane, s=10 + 60*conf, alpha=0.6)
        ax.set_xlim(left=0, right=max(t)+0.1)
    else:
        ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(lane_labels)-0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Swara lane")
    ax.set_title("Swara Lanes (relative to Sa)")
    fig.tight_layout()
    return fig

# --------------------------
# Audio Slicing for Game
# --------------------------
def slice_audio_wav_bytes(y: np.ndarray, sr: int, t0: float, t1: float) -> bytes:
    t0 = max(0.0, float(t0))
    t1 = max(t0 + 0.05, float(t1))
    i0 = int(t0 * sr)
    i1 = int(t1 * sr)
    i1 = min(len(y), max(i1, i0 + 1))
    seg = y[i0:i1].astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="Carnatic Swara Visualizer + Games", layout="wide")
st.title("Carnatic Swara Visualizer + Mini-Games")
st.caption("Paste a YouTube link or upload audio → pitch → Sa → swaras/octaves → time-synced visualization → play games")

# Session state containers
if "analysis" not in st.session_state:
    st.session_state.analysis = {}
if "game" not in st.session_state:
    st.session_state.game = {"score": 0, "total": 0, "last_result": None, "current_q": None}

with st.sidebar:
    st.header("Input")
    yt_url = st.text_input("YouTube URL (optional)")
    uploaded = st.file_uploader("Or upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a","flac","ogg"])

    st.header("Analysis")
    fmin = st.number_input("Min F0 (Hz)", 60.0, 500.0, 75.0, 1.0)
    fmax = st.number_input("Max F0 (Hz)", 200.0, 2000.0, 800.0, 10.0)
    frame_len_ms = st.number_input("Frame length (ms)", 5.0, 100.0, 20.0, 1.0)
    hop_len_ms = st.number_input("Hop length (ms)", 1.0, 50.0, 10.0, 1.0)

    st.header("Tonic (Sa)")
    sa_mode = st.selectbox("How to set Sa?", ["Auto-estimate", "Manual (Hz)"])
    sa_manual = st.number_input("Manual Sa (Hz)", 50.0, 400.0, 132.0, 1.0)

    st.header("Scale Mode")
    scale_mode = st.selectbox("Choose scale definition", ["Carnatic (12-TET swaras)", "Custom scale (labels + semitone offsets)"])
    custom_labels = "S,R2,G3,P,D2,S'"
    custom_semitones = "0,2,4,7,9,12"
    if scale_mode == "Custom scale (labels + semitone offsets)":
        custom_labels = st.text_input("Custom labels (comma-separated)", value=custom_labels)
        custom_semitones = st.text_input("Semitone offsets relative to Sa (comma-separated)", value=custom_semitones)

    st.header("Run")
    snap_thresh = st.slider("Snap threshold (cents)", 10, 120, 70, 5)
    accept_btn = st.button("Run Analysis", use_container_width=True)

# Temp workdir for yt-dlp
_tmp_dir = tempfile.mkdtemp()
audio_path = None

# Run analysis
if accept_btn:
    try:
        if yt_url:
            with st.status("Downloading audio from YouTube…", expanded=False):
                audio_path = download_audio_from_youtube(yt_url, _tmp_dir)
                st.success(f"Downloaded and converted to WAV.")
        if uploaded is not None and audio_path is None:
            suffix = os.path.splitext(uploaded.name)[1] or ".wav"
            audio_path = os.path.join(_tmp_dir, f"upload{suffix}")
            with open(audio_path, "wb") as f:
                f.write(uploaded.read())
            st.success(f"Using uploaded file: {uploaded.name}")

        if audio_path is None:
            st.warning("Please provide a YouTube URL or upload an audio file.")
        else:
            y, sr = load_audio_mono(audio_path)
            st.write(f"Loaded audio: {len(y)/sr:.1f}s at {sr} Hz")

            f0_hz, f0_conf, times = estimate_pitch_track(
                y, sr,
                fmin=float(fmin), fmax=float(fmax),
                frame_length_ms=float(frame_len_ms),
                hop_length_ms=float(hop_len_ms)
            )

            if sa_mode == "Auto-estimate":
                sa_hz = estimate_tonic_sa(f0_hz)
                st.success(f"Estimated Sa: ~{sa_hz:.2f} Hz")
            else:
                sa_hz = float(sa_manual)
                st.info(f"Manual Sa: {sa_hz:.2f} Hz")

            # Scale configuration
            if scale_mode == "Carnatic (12-TET swaras)":
                lane_labels = DEFAULT_SWARA_ORDER
                semi_map = DEFAULT_ET_SEMITONES
            else:
                lbls = [s.strip() for s in custom_labels.split(",") if s.strip()]
                semis_list = [s.strip() for s in custom_semitones.split(",") if s.strip()]
                if len(lbls) != len(semis_list):
                    st.error("Custom labels and semitone offsets must have the same count.")
                    st.stop()
                try:
                    semis = [int(x) for x in semis_list]
                except ValueError:
                    st.error("Semitone offsets must be integers.")
                    st.stop()
                lane_labels = lbls
                semi_map = {lbl: semi for lbl, semi in zip(lbls, semis)}

            scale_table = build_scale_table(sa_hz, lane_labels, semi_map)
            df_rows = map_freq_to_scale_rows(f0_hz, f0_conf, times, scale_table, snap_thresh_cents=float(snap_thresh))
            df = pd.DataFrame(df_rows)
            phrases = detect_phrases(y, sr, hop_length_ms=float(hop_len_ms))

            st.session_state.analysis = {
                "y": y, "sr": sr, "df": df, "phrases": phrases,
                "lane_labels": lane_labels, "sa_hz": sa_hz,
                "scale_table": scale_table
            }
            st.session_state.game = {"score": 0, "total": 0, "last_result": None, "current_q": None}

    except Exception as e:
        st.error(f"Analysis failed: {e}")

# Tabs: Analysis + Games
tab1, tab2 = st.tabs(["Analysis", "Games"])

with tab1:
    if not st.session_state.analysis:
        st.info("Run analysis to see visualizations and data.")
    else:
        y = st.session_state.analysis["y"]
        sr = st.session_state.analysis["sr"]
        df = st.session_state.analysis["df"]
        phrases = st.session_state.analysis["phrases"]
        lane_labels = st.session_state.analysis["lane_labels"]

        st.subheader("Mapped Timeline (preview)")
        st.dataframe(df.head(200))
        csv_buf = io.StringIO()
        out_df = df.rename(columns={"label": "swara_label", "order": "swara_order"})
        out_df.to_csv(csv_buf, index=False)
        st.download_button("Download CSV", data=csv_buf.getvalue(), file_name="swara_timeline.csv")

        st.subheader("2D Swara Lane Visualization")
        fig = draw_2d_lane_plot(out_df, lane_labels)
        st.pyplot(fig, clear_figure=True)

        st.subheader("3D Scatter (experimental)")
        df3d = out_df.dropna().copy()
        if not df3d.empty:
            df3d["swara_idx"] = df3d["swara_order"]
            fig3 = px.scatter_3d(
                df3d.sample(min(len(df3d), 5000), random_state=42) if len(df3d) > 5000 else df3d,
                x="time_s", y="octave", z="swara_idx",
                size="conf", hover_data=["swara_label","freq_hz"]
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough confident pitch frames to show 3D scatter.")

        st.subheader("Detected Phrase Segments (rough)")
        st.write(phrases[:20])

with tab2:
    st.header("Music Games")
    if not st.session_state.analysis:
        st.info("Run analysis first to unlock the games.")
    else:
        y = st.session_state.analysis["y"]
        sr = st.session_state.analysis["sr"]
        df = st.session_state.analysis["df"]
        lane_labels = st.session_state.analysis["lane_labels"]

        conf_thresh = st.slider("Confidence threshold for questions", 0.0, 1.0, 0.6, 0.05)
        voiced_idx = df.index[df["label"].notna() & (df["conf"] >= conf_thresh)].tolist()

        game_type = st.selectbox("Choose a mini-game", [
            "Identify the Swara",
            "Transition Direction (Up / Down / Same)"
        ])

        st.markdown(f"Score: {st.session_state.game['score']} / {st.session_state.game['total']}")
        if st.session_state.game["last_result"] is not None:
            if st.session_state.game["last_result"]:
                st.success("Correct!")
            else:
                st.error("Not quite—try the next one.")

        def slice_audio_wav_bytes(y: np.ndarray, sr: int, t0: float, t1: float) -> bytes:
            t0 = max(0.0, float(t0))
            t1 = max(t0 + 0.05, float(t1))
            i0 = int(t0 * sr)
            i1 = int(t1 * sr)
            i1 = min(len(y), max(i1, i0 + 1))
            seg = y[i0:i1].astype(np.float32)
            buf = io.BytesIO()
            sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
            buf.seek(0)
            return buf.read()

        def new_question_identify_swara():
            if not voiced_idx:
                return None
            idx = random.choice(voiced_idx)
            row = df.iloc[idx]
            t = float(row["time_s"])
            dur = 0.6
            t0 = max(0.0, t - dur/2)
            t1 = min(len(y)/sr, t + dur/2)
            audio_bytes = slice_audio_wav_bytes(y, sr, t0, t1)
            correct = str(row["label"])
            choices_pool = list({str(x) for x in df["label"].dropna().unique().tolist()})
            if correct not in choices_pool:
                choices_pool.append(correct)
            random.shuffle(choices_pool)
            choices = [correct]
            for ch in choices_pool:
                if ch != correct and len(choices) < 4:
                    choices.append(ch)
            random.shuffle(choices)
            return {
                "type": "identify_swara",
                "audio": audio_bytes,
                "t0": t0, "t1": t1,
                "correct": correct,
                "choices": choices,
                "meta": {"time_s": t}
            }

        def new_question_transition_direction():
            valid_pairs = []
            times = df["time_s"].values
            orders = df["order"].values
            confs = df["conf"].values
            for i in range(1, len(df)):
                if (df["label"].notna().iloc[i] and df["label"].notna().iloc[i-1]
                    and confs[i] >= conf_thresh and confs[i-1] >= conf_thresh):
                    dt = times[i] - times[i-1]
                    if 0.02 <= dt <= 0.25:
                        valid_pairs.append(i)
            if not valid_pairs:
                return None
            i = random.choice(valid_pairs)
            prev_i = i - 1
            o_prev = int(orders[prev_i])
            o_curr = int(orders[i])
            delta = o_curr - o_prev
            if delta > 0:
                correct = "Up"
            elif delta < 0:
                correct = "Down"
            else:
                correct = "Same"
            t0 = max(0.0, times[prev_i] - 0.1)
            t1 = min(len(y)/sr, times[i] + 0.2)
            audio_bytes = slice_audio_wav_bytes(y, sr, t0, t1)
            return {
                "type": "transition_dir",
                "audio": audio_bytes,
                "t0": t0, "t1": t1,
                "correct": correct,
                "choices": ["Up", "Down", "Same"],
                "meta": {"from": str(df["label"].iloc[prev_i]), "to": str(df["label"].iloc[i])}
            }

        colA, colB = st.columns([1,1])
        with colA:
            if st.button("New Question", use_container_width=True):
                if game_type == "Identify the Swara":
                    q = new_question_identify_swara()
                else:
                    q = new_question_transition_direction()
                if q is None:
                    st.warning("Not enough confident frames to generate a question. Try lowering the threshold or re-running analysis.")
                st.session_state.game["current_q"] = q
                st.session_state.game["last_result"] = None

        q = st.session_state.game.get("current_q")
        if q is not None:
            st.subheader("Listen and answer")
            st.audio(q["audio"], format="audio/wav")
            user_choice = st.radio("Your answer:", q["choices"], index=None, horizontal=True)
            with colB:
                if st.button("Submit", use_container_width=True, disabled=(user_choice is None)):
                    st.session_state.game["total"] += 1
                    if user_choice == q["correct"]:
                        st.session_state.game["score"] += 1
                        st.session_state.game["last_result"] = True
                        st.balloons()
                    else:
                        st.session_state.game["last_result"] = False
                        st.info(f"Correct answer: {q['correct']}")
        else:
            st.info("Click 'New Question' to begin.")

        st.divider()
        with st.expander("Tips and Notes"):
            st.markdown(
                "- 'Identify the Swara' plays a short snippet; pick the swara you hear at that moment.\n"
                "- 'Transition Direction' plays two close frames; decide if the melody moved Up, Down, or stayed the Same.\n"
                "- You can tighten or loosen the confidence threshold to adjust difficulty.\n"
                "- Use the Custom scale mode to enter labels and semitone offsets for Pann-like explorations."
            )
