"""
Carnatic Swara Explorer — Fast, Stepwise, and Talkative

Goals:
- Start instantly with a lightweight UI (no heavy imports at startup).
- Run heavy work only AFTER Step 1 (when the user loads audio).
- Keep users informed with clear, live status messages during each step.
- Offer simple, interactive visualizations with fast defaults and optional deeper analysis.

How to run:
  streamlit run ragames/carnatic_swara_app.py

Notes:
- YouTube audio requires yt-dlp and ffmpeg installed on the server/environment.
- Pitch (pYIN) is slower; YIN is faster. Choose in Step 2.
- You can limit analysis duration for responsiveness.
"""

import os
import io
import uuid
import time
import shutil
import tempfile
import subprocess
from typing import Optional, Tuple, Dict, Any

import streamlit as st

# -------------------------------
# Lightweight startup: no heavy imports here.
# Heavy deps (librosa, numpy, pandas) are lazy-imported only when needed.
# -------------------------------

APP_STATE_KEYS = [
    "audio_path",      # str | None
    "audio_info",      # dict | None: {sr:int, duration:float}
    "y", "sr",         # numpy array and int (set only when analysis requested)
    "waveform_df",     # pandas DataFrame | None
    "pitch_df",        # pandas DataFrame | None
    "swara_df",        # pandas DataFrame | None
    "last_error"       # str | None
]

for k in APP_STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

# -------------------------------
# Utilities
# -------------------------------
def have_tool(exe: str) -> bool:
    return shutil.which(exe) is not None

def lazy_libs() -> Dict[str, Any]:
    """
    Import heavy libraries lazily and return them in a dict.
    """
    if "_libs" in st.session_state and st.session_state["_libs"] is not None:
        return st.session_state["_libs"]

    with st.status("Loading analysis libraries…", expanded=True) as status:
        st.write("Importing numpy, pandas, and librosa…")
        import numpy as np
        import pandas as pd
        try:
            import librosa
            import librosa.display  # optional
        except Exception as e:
            st.session_state["last_error"] = f"Failed to import librosa: {e}"
            raise

        st.write("Libraries ready.")
        status.update(label="Libraries loaded", state="complete")
        libs = {"np": np, "pd": pd, "librosa": librosa}
        st.session_state["_libs"] = libs
        return libs

def download_youtube_audio(url: str, outdir: str, target_sr: int = 22050) -> Tuple[Optional[str], Optional[str]]:
    """
    Download YouTube audio using yt-dlp and convert to mono WAV with ffmpeg.
    Returns (wav_path, error).
    """
    if not have_tool("yt-dlp"):
        return None, "yt-dlp is not installed on this environment."
    if not have_tool("ffmpeg"):
        return None, "ffmpeg is not installed on this environment."

    tmp_id = str(uuid.uuid4())
    out_audio = os.path.join(outdir, f"{tmp_id}.m4a")
    out_wav = os.path.join(outdir, f"{tmp_id}.wav")

    try:
        subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "m4a", "-o", out_audio, url],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        return None, f"yt-dlp failed: {e.stderr.decode(errors='ignore')[:500]}"

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_audio, "-ac", "1", "-ar", str(target_sr), out_wav],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        return None, f"ffmpeg failed: {e.stderr.decode(errors='ignore')[:500]}"

    return out_wav, None

def quick_audio_info(path: str) -> Dict[str, Any]:
    """
    Quickly probe audio info using ffprobe if available; otherwise, defer to librosa when loading.
    """
    info = {"sr": None, "duration": None}
    if have_tool("ffprobe"):
        try:
            p = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=sample_rate,duration",
                 "-of", "default=nw=1:nk=1", path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            lines = p.stdout.decode().strip().splitlines()
            if len(lines) >= 2:
                # Order can vary; attempt to parse both as floats/ints
                a, b = lines[0], lines[1]
                for x in (a, b):
                    if x.isdigit():
                        info["sr"] = int(x)
                    else:
                        try:
                            info["duration"] = float(x)
                        except Exception:
                            pass
        except Exception:
            pass
    return info

# -------------------------------
# Analysis helpers (lazy import)
# -------------------------------
def load_audio(path: str, target_sr: int = 22050) -> Tuple[Any, int, float]:
    libs = lazy_libs()
    np, pd, librosa = libs["np"], libs["pd"], libs["librosa"]

    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = librosa.util.normalize(y)
    duration = len(y) / sr
    return y, sr, duration

def compute_waveform_preview(path: str, target_sr: int, max_duration_s: float = 30.0):
    libs = lazy_libs()
    np, pd, librosa = libs["np"], libs["pd"], libs["librosa"]

    # Load at lower sr for speed if target_sr is high
    sr_preview = min(16000, target_sr)
    y, sr = librosa.load(path, sr=sr_preview, mono=True, duration=max_duration_s)
    y = librosa.util.normalize(y)

    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    df = pd.DataFrame({"time_s": times, "rms": rms})
    return df, sr

def compute_pitch(y, sr, method: str = "YIN", fmin: float = 75.0, fmax: float = 800.0,
                  max_duration_s: Optional[float] = 60.0):
    libs = lazy_libs()
    np, pd, librosa = libs["np"], libs["pd"], libs["librosa"]

    if max_duration_s is not None:
        y = y[: int(sr * max_duration_s)]

    if method.upper() == "PYIN":
        # pYIN: slower but yields per-frame confidence
        f0, _, voiced_prob = librosa.pyin(
            y, fmin=fmin, fmax=fmax, frame_length=2048, hop_length=256
        )
        conf = np.nan_to_num(voiced_prob, nan=0.0)
    else:
        # YIN: fast; no confidence -> set to 0.5
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, frame_length=2048, hop_length=256)
        conf = np.ones_like(f0) * 0.5

    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=256)
    df = pd.DataFrame({"time_s": times, "f0_hz": f0, "conf": conf})
    return df

DEFAULT_SWARA_LABELS = ["S","R1","R2","G2","G3","M1","M2","P","D1","D2","N2","N3","S'"]
DEFAULT_SWARA_SEMITONES = [0,1,2,3,4,5,6,7,8,9,10,11,12]

def estimate_sa(f0_series) -> float:
    libs = lazy_libs()
    np, pd, librosa = libs["np"], libs["pd"], libs["librosa"]

    hz = f0_series.to_numpy()
    hz = hz[np.isfinite(hz) & (hz > 0)]
    if len(hz) < 50:
        return 132.0
    cents = (1200.0 * np.log2(hz / np.mean(hz))) % 1200.0
    hist, _ = np.histogram(cents, bins=12, range=(0, 1200))
    peak_bin = int(np.argmax(hist))
    mean_hz = np.median(hz)
    sa = mean_hz / (2 ** (peak_bin / 12.0))
    while sa < 100: sa *= 2
    while sa > 300: sa /= 2
    return float(sa)

def map_to_swara(pitch_df, sa_hz: float, snap_cents: float = 70.0):
    libs = lazy_libs()
    np, pd, librosa = libs["np"], libs["pd"], libs["librosa"]

    rows = []
    for t, f, c in pitch_df[["time_s","f0_hz","conf"]].itertuples(index=False):
        if not (isinstance(f, (int, float)) and f > 0 and np.isfinite(f)):
            rows.append({"time_s": t, "f0_hz": None, "conf": float(c),
                         "swara": None, "lane": None})
            continue
        best = (1e9, None, None)  # cents, swara, lane
        for lane, semi in enumerate(DEFAULT_SWARA_SEMITONES):
            ref = sa_hz * (2 ** (semi / 12))
            cents = abs(1200.0 * np.log2(f / ref))
            if cents < best[0]:
                best = (cents, DEFAULT_SWARA_LABELS[lane], lane)
        if best[0] <= snap_cents:
            rows.append({"time_s": t, "f0_hz": float(f), "conf": float(c),
                         "swara": best[1], "lane": int(best[2])})
        else:
            rows.append({"time_s": t, "f0_hz": float(f), "conf": float(c),
                         "swara": None, "lane": None})
    return pd.DataFrame(rows)

# -------------------------------
# UI — Fast Starting Screen
# -------------------------------
st.set_page_config(page_title="Carnatic Swara Explorer", layout="wide")
st.title("Carnatic Swara Explorer")
st.caption("Fast, stepwise, and informative. Load audio first, then choose what to compute — no long waits up front.")

with st.sidebar:
    st.header("Step 1: Load Audio")
    src = st.radio("Choose source", ["YouTube URL", "Upload file"], index=0, horizontal=False)
    yt_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…") if src == "YouTube URL" else None
    uploaded = st.file_uploader("Upload audio", type=["wav","mp3","m4a","flac","ogg"]) if src == "Upload file" else None

    st.subheader("Load Options")
    target_sr = st.selectbox("Target sample rate (Hz)", [16000, 22050, 32000, 44100], index=1)
    if "load_limit" not in st.session_state:
        st.session_state["load_limit"] = 0
    load_btn = st.button("Load Audio", type="primary", use_container_width=True)

    st.markdown("---")
    st.header("Step 2: Visualize / Analyze")
    viz_choice = st.selectbox("What do you want to compute?",
                              ["None",
                               "Waveform preview (fast)",
                               "Pitch curve (fast YIN)",
                               "Pitch curve (accurate pYIN)",
                               "Swara lane mapping"],
                              index=0)
    max_dur = st.slider("Limit analysis to first N seconds", 10, 180, 60, 10)
    fmin = st.number_input("Min pitch (Hz)", 50.0, 300.0, 75.0, 1.0)
    fmax = st.number_input("Max pitch (Hz)", 200.0, 1500.0, 800.0, 10.0)
    sa_mode = st.radio("Sa (tonic)", ["Auto", "Manual"], index=0, horizontal=True)
    sa_manual = st.number_input("Manual Sa (Hz)", 80.0, 300.0, 132.0, 1.0)
    go_btn = st.button("Run Selected Visualization", use_container_width=True)

# -------------------------------
# Starting screen info (always instant)
# -------------------------------
colA, colB = st.columns([2,1])
with colA:
    st.subheader("What you can do")
    st.markdown(
        "- Step 1: Load audio from YouTube or upload a file. We show status as it happens.\n"
        "- Step 2: Choose a visualization. We only compute what you ask for, with live progress.\n"
        "- Start with the fast waveform preview. Then try fast YIN. pYIN is slower but more accurate.\n"
        "- For swaras, use Auto Sa first; switch to Manual if desired."
    )
with colB:
    st.info("Tip: If the app seemed slow before, it's now stepwise. Nothing heavy runs until you click Load or Run.")

# -------------------------------
# Step 1: Load audio (on-demand)
# -------------------------------
if load_btn:
    st.session_state["last_error"] = None
    st.session_state["waveform_df"] = None
    st.session_state["pitch_df"] = None
    st.session_state["swara_df"] = None
    st.session_state["y"] = None
    st.session_state["sr"] = None
    st.session_state["audio_info"] = None

    with st.status("Preparing to load audio…", expanded=True) as status:
        tmpdir = tempfile.mkdtemp()
        if src == "YouTube URL":
            if not yt_url:
                st.error("Please paste a YouTube URL.")
                status.update(state="error")
            else:
                st.write("Checking tools: yt-dlp and ffmpeg…")
                if not have_tool("yt-dlp") or not have_tool("ffmpeg"):
                    st.error("Missing yt-dlp or ffmpeg. Install them in your environment.")
                    status.update(state="error")
                else:
                    st.write("Downloading audio with yt-dlp…")
                    wav_path, err = download_youtube_audio(yt_url, tmpdir, target_sr=target_sr)
                    if err:
                        st.error(err)
                        status.update(label="Failed to load audio", state="error")
                    else:
                        st.write("Download complete. Probing audio info…")
                        info = quick_audio_info(wav_path)
                        st.session_state["audio_path"] = wav_path
                        st.session_state["audio_info"] = info
                        st.success("Audio ready.")
                        status.update(label="Audio loaded", state="complete")
        else:
            if uploaded is None:
                st.error("Please upload an audio file.")
                status.update(state="error")
            else:
                st.write("Saving uploaded file…")
                upath = os.path.join(tmpdir, uploaded.name)
                with open(upath, "wb") as f:
                    f.write(uploaded.read())
                st.write("Converting (if needed) will happen during analysis; for now, using the uploaded file directly.")
                st.session_state["audio_path"] = upath
                st.session_state["audio_info"] = quick_audio_info(upath)
                st.success("Audio ready.")
                status.update(label="Audio loaded", state="complete")

# Show audio player and basic metadata if ready
if st.session_state["audio_path"]:
    st.success("Audio is loaded.")
    st.audio(st.session_state["audio_path"])
    info = st.session_state["audio_info"] or {}
    sr_info = info.get("sr") or "unknown"
    dur_info = info.get("duration")
    st.caption(f"Sample rate: {sr_info} Hz | Duration: {f'{dur_info:.1f} s' if dur_info else 'unknown'}")

# -------------------------------
# Step 2: Run selected visualization (on-demand)
# -------------------------------
def ensure_audio_loaded_to_memory():
    """
    Load audio samples into memory only when needed.
    """
    if st.session_state["y"] is not None and st.session_state["sr"] is not None:
        return True
    if not st.session_state["audio_path"]:
        st.warning("Load audio in Step 1 first.")
        return False

    with st.status("Loading audio into memory…", expanded=True) as status:
        try:
            y, sr, duration = load_audio(st.session_state["audio_path"], target_sr=target_sr)
            st.session_state["y"] = y
            st.session_state["sr"] = sr
            if st.session_state["audio_info"] is None:
                st.session_state["audio_info"] = {"sr": sr, "duration": duration}
            st.write(f"Loaded mono audio: {duration:.1f} seconds at {sr} Hz")
            status.update(label="Audio loaded in memory", state="complete")
            return True
        except Exception as e:
            st.error(f"Failed to load audio: {e}")
            status.update(state="error")
            return False

def run_waveform_preview():
    with st.status("Computing waveform preview…", expanded=True) as status:
        try:
            df, sr_preview = compute_waveform_preview(
                st.session_state["audio_path"], target_sr=target_sr, max_duration_s=30.0
            )
            st.session_state["waveform_df"] = df
            st.write(f"Preview computed at {sr_preview} Hz, showing first ~30s for speed.")
            status.update(label="Waveform ready", state="complete")
            st.line_chart(df.set_index("time_s"))
        except Exception as e:
            st.error(f"Waveform preview failed: {e}")
            status.update(state="error")

def run_pitch(method: str):
    if not ensure_audio_loaded_to_memory():
        return
    with st.status(f"Estimating pitch with {method}…", expanded=True) as status:
        try:
            df = compute_pitch(
                st.session_state["y"],
                st.session_state["sr"],
                method=method,
                fmin=float(fmin),
                fmax=float(fmax),
                max_duration_s=float(max_dur)
            )
            st.session_state["pitch_df"] = df
            st.write(f"Pitch frames: {len(df)}; duration displayed: up to {max_dur}s.")
            status.update(label="Pitch curve ready", state="complete")
            # Plot using Streamlit native line chart; NaNs are okay.
            st.line_chart(df.set_index("time_s")["f0_hz"])
        except Exception as e:
            st.error(f"Pitch estimation failed: {e}")
            status.update(state="error")

def run_swara_mapping():
    if st.session_state["pitch_df"] is None:
        st.warning("Run a pitch curve first (YIN or pYIN).")
        return
    with st.status("Mapping to swaras…", expanded=True) as status:
        try:
            if sa_mode == "Auto":
                sa_hz = estimate_sa(st.session_state["pitch_df"]["f0_hz"])
                st.write(f"Auto-estimated Sa ≈ {sa_hz:.2f} Hz")
            else:
                sa_hz = float(sa_manual)
                st.write(f"Manual Sa set to {sa_hz:.2f} Hz")

            swara_df = map_to_swara(st.session_state["pitch_df"], sa_hz=sa_hz, snap_cents=70.0)
            st.session_state["swara_df"] = swara_df
            status.update(label="Swara mapping ready", state="complete")

            # Build a simple scatter chart (time vs lane). We'll also show a legend for lanes.
            plot_df = swara_df.dropna(subset=["lane"]).copy()
            if plot_df.empty:
                st.info("Not enough confident frames were mapped to swaras at the current settings.")
                return
            st.write("Swara lanes legend: 0:S, 1:R1, 2:R2, 3:G2, 4:G3, 5:M1, 6:M2, 7:P, 8:D1, 9:D2, 10:N2, 11:N3, 12:S'")
            st.scatter_chart(plot_df.rename(columns={"time_s": "x", "lane": "y"})[["x","y"]], height=250)
            st.caption("Each point shows a mapped frame. Use pitch settings or manual Sa to refine.")

        except Exception as e:
            st.error(f"Swara mapping failed: {e}")
            status.update(state="error")

if go_btn:
    choice = viz_choice
    if not st.session_state["audio_path"]:
        st.warning("Please load audio first in Step 1.")
    else:
        if choice == "Waveform preview (fast)":
            run_waveform_preview()
        elif choice == "Pitch curve (fast YIN)":
            run_pitch("YIN")
        elif choice == "Pitch curve (accurate pYIN)":
            run_pitch("PYIN")
        elif choice == "Swara lane mapping":
            run_swara_mapping()
        else:
            st.info("Choose a visualization from the sidebar, then click Run.")

# -------------------------------
# Helpful footer
# -------------------------------
st.markdown("""
---
Troubleshooting tips:
- If YouTube load fails, ensure yt-dlp and ffmpeg are installed and accessible in PATH.
- For faster results, start with "Waveform preview (fast)" and "Pitch curve (fast YIN)".
- Limit analysis duration in the sidebar to speed things up.
- If swara mapping looks sparse, try Manual Sa or widen the pitch range (fmin/fmax).
""")
