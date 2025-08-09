import os
import io
import uuid
import tempfile
import shutil
import subprocess

# Headless-safe matplotlib config
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

# ---- Helper Functions ---- #
def download_youtube_audio(url, outdir, sr=22050):
    if not shutil.which("yt-dlp") or not shutil.which("ffmpeg"):
        return None, "yt-dlp/ffmpeg not installed on server."
    tmpid = str(uuid.uuid4())
    out_m4a = os.path.join(outdir, f"{tmpid}.m4a")
    out_wav = os.path.join(outdir, f"{tmpid}.wav")
    try:
        subprocess.run(["yt-dlp", "-x", "--audio-format", "m4a", "-o", out_m4a, url], check=True)
        subprocess.run(["ffmpeg", "-y", "-i", out_m4a, "-ac", "1", "-ar", str(sr), out_wav], check=True)
        return out_wav, None
    except Exception as e:
        return None, f"Error downloading audio: {e}"

def load_audio(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
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

def plot_swara_lane(f0, times, sa_hz=132.0):
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
    return fig

# ---- Streamlit App ---- #
st.set_page_config(page_title="Carnatic Swara Stepwise", layout="wide")
st.title("Carnatic Swara Explorer (Stepwise & Interactive)")

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "audio_ready" not in st.session_state:
    st.session_state.audio_ready = False
if "y" not in st.session_state:
    st.session_state.y = None
    st.session_state.sr = None

# ---- STEP 1: LOAD AUDIO ---- #
with st.sidebar:
    st.header("Step 1: Load Audio")
    yt_url = st.text_input("Paste YouTube URL (audio only)")
    uploaded = st.file_uploader("Or upload audio file", type=["wav","mp3","m4a","flac","ogg"])
    if st.button("Load Audio"):
        tmpdir = tempfile.mkdtemp()
        if yt_url:
            wav_path, err = download_youtube_audio(yt_url, tmpdir)
            if wav_path:
                st.session_state.audio_path = wav_path
                st.session_state.audio_ready = True
                st.session_state.y = None
                st.session_state.sr = None
            else:
                st.session_state.audio_path = None
                st.session_state.audio_ready = False
                st.error(err)
        elif uploaded:
            upath = os.path.join(tmpdir, uploaded.name)
            with open(upath, "wb") as f: f.write(uploaded.read())
            st.session_state.audio_path = upath
            st.session_state.audio_ready = True
            st.session_state.y = None
            st.session_state.sr = None
        else:
            st.session_state.audio_path = None
            st.session_state.audio_ready = False
            st.error("Provide a YouTube link or upload audio.")

# ---- Only proceed to next step if audio is ready ---- #
if st.session_state.audio_ready and st.session_state.audio_path:
    st.success("Audio loaded!")
    st.audio(st.session_state.audio_path)
    # Step 2: Load audio data
    if st.session_state.y is None or st.session_state.sr is None:
        y, sr = load_audio(st.session_state.audio_path)
        st.session_state.y = y
        st.session_state.sr = sr

    st.write(f"Audio length: {len(st.session_state.y)/st.session_state.sr:.1f} sec | Sample rate: {st.session_state.sr}")

    # ---- STEP 2: VISUALIZATION MENU ---- #
    st.header("Step 2: Visualization")
    menu_visual = st.radio("Choose visualization:",
        ["Waveform", "Pitch Curve", "Swara Lane"], index=0
    )

    # ---- STEP 3: PLOT (only run after menu selection) ---- #
    if menu_visual == "Waveform":
        st.pyplot(plot_waveform(st.session_state.y, st.session_state.sr))
    elif menu_visual == "Pitch Curve":
        st.pyplot(plot_pitch(st.session_state.y, st.session_state.sr))
    elif menu_visual == "Swara Lane":
        try:
            f0, _, _ = librosa.pyin(st.session_state.y, fmin=75, fmax=800)
            times = librosa.frames_to_time(np.arange(len(f0)), sr=st.session_state.sr)
            st.pyplot(plot_swara_lane(f0, times, sa_hz=132.0))
        except Exception as e:
            st.warning(f"Swara lane plot failed: {e}")

    # ---- STEP 4: (Future) Add analysis, games etc. ---- #

else:
    st.info("Step 1: Paste YouTube link or upload audio, then click Load Audio.")

st.markdown("""
---
**Workflow:**  
1. Step 1: Paste YouTube link or upload audio, click Load Audio.  
2. Step 2: Once loaded, choose visualization (waveform, pitch curve, swara lane).  
3. Step 3: Further steps (phrase analysis, games, downloads) can be added later!
""")
