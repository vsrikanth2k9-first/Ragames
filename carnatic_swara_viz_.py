import os
import io
import uuid
import random
import shutil
import tempfile
import subprocess

# Headless-safe Matplotlib config
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
import streamlit as st

# -------------- Helper Functions -------------- #
def download_and_convert_youtube(url, out_dir, target_sr=22050):
    if not shutil.which("yt-dlp") or not shutil.which("ffmpeg"):
        return None, "yt-dlp/ffmpeg not installed on server!"
    tmpid = str(uuid.uuid4())
    out_m4a = os.path.join(out_dir, f"{tmpid}.m4a")
    out_wav = os.path.join(out_dir, f"{tmpid}.wav")
    cmd1 = ["yt-dlp", "-x", "--audio-format", "m4a", "-o", out_m4a, url]
    try:
        subprocess.run(cmd1, check=True)
        cmd2 = ["ffmpeg", "-y", "-i", out_m4a, "-ac", "1", "-ar", str(target_sr), out_wav]
        subprocess.run(cmd2, check=True)
        return out_wav, None
    except Exception as e:
        return None, f"Error downloading audio: {e}"

def load_audio(path, target_sr=22050):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = librosa.util.normalize(y)
    return y, sr

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10,2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    return fig

def plot_pitch(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=800)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
    fig, ax = plt.subplots(figsize=(10,2))
    ax.plot(times, f0, '.', markersize=2, alpha=0.7)
    ax.set_title("Pitch Curve (f0 estimate)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return fig

def plot_swara_lanes(f0, times, sa_hz=132.0, ragam=None):
    swaras = ["S", "R2", "G3", "M1", "P", "D2", "N3", "S'"]
    semitones = [0,2,4,5,7,9,11,12]
    lanes = []
    for hz in f0:
        if hz is None or np.isnan(hz) or hz <= 0: lanes.append(None)
        else:
            cents = 1200*np.log2(hz/sa_hz)
            lane = None
            for s, st in enumerate(semitones):
                if abs(cents - st*100) < 70: lane = s
            lanes.append(lane)
    fig, ax = plt.subplots(figsize=(10,2))
    ax.plot(times, lanes, '.', alpha=0.6)
    ax.set_yticks(range(len(swaras)))
    ax.set_yticklabels(swaras)
    ax.set_title("Swara Lane (Sa-relative)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Swara")
    if ragam: ax.text(0.98, 0.98, f"Ragam: {ragam}", transform=ax.transAxes, ha='right', va='top', fontsize=12, alpha=0.6)
    return fig

# -------------- Streamlit App -------------- #
st.set_page_config(page_title="Carnatic Swara Interactive", layout="wide")
st.title("Carnatic Swara Explorer")

with st.sidebar:
    st.header("Step 1: Load Audio")
    yt_url = st.text_input("Paste YouTube URL (audio only)")
    uploaded = st.file_uploader("Or upload audio file", type=["wav","mp3","m4a","flac","ogg"])
    run_download = st.button("Fetch Audio")

    st.header("Step 2: Listen")
    audio_ready = st.session_state.get("audio_path") is not None
    menu_visual = st.radio("Visualization", ["Waveform", "Pitch Curve", "Swara Lane"], index=0)
    menu_ragam = st.selectbox("Ragam/Pann overlay", ["None", "Kalyani", "Todi", "Bhairavi"])

if run_download:
    tmpdir = tempfile.mkdtemp()
    if yt_url:
        wav_path, err = download_and_convert_youtube(yt_url, tmpdir)
        if wav_path: st.session_state["audio_path"] = wav_path
        else: st.error(err)
    elif uploaded:
        upath = os.path.join(tmpdir, uploaded.name)
        with open(upath, "wb") as f: f.write(uploaded.read())
        st.session_state["audio_path"] = upath
    else:
        st.error("Provide a YouTube link or upload audio.")

if st.session_state.get("audio_path"):
    st.success("Audio loaded!")
    st.audio(st.session_state["audio_path"])

    y, sr = load_audio(st.session_state["audio_path"])
    st.write(f"Audio length: {len(y)/sr:.1f} sec | Sample rate: {sr}")

    # Step-by-step visualization
    if menu_visual == "Waveform":
        st.pyplot(plot_waveform(y, sr))
    elif menu_visual == "Pitch Curve":
        st.pyplot(plot_pitch(y, sr))
    elif menu_visual == "Swara Lane":
        try:
            f0, _, _ = librosa.pyin(y, fmin=75, fmax=800)
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
            ragam = menu_ragam if menu_ragam != "None" else None
            st.pyplot(plot_swara_lanes(f0, times, sa_hz=132.0, ragam=ragam))
        except Exception as e:
            st.warning(f"Swara lane plot failed: {e}")

    st.markdown("**Try changing visualization type and ragam overlay in the sidebar.**")

    # Optional: Step 3 - Add interactive games, phrase detection, etc.

else:
    st.info("Step 1: Paste YouTube link or upload audio, then click Fetch Audio.")

st.markdown("""
---
**How this works:**  
1. Paste a YouTube link or upload an audio file.  
2. Listen and preview audio inside the app.  
3. Choose visualization: waveform (raw audio), pitch curve (melody), or swara lane (Carnatic notes).  
4. Optionally overlay ragam/pann.  
5. Further steps: interactive games, phrase analysis, CSV download, etc.
""")
