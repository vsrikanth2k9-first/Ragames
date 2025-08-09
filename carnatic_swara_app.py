import streamlit as st

# --- Always fast: greeting + step 1 UI ---
st.set_page_config(page_title="Carnatic Swara Explorer", layout="wide")
st.title("Carnatic Swara Explorer")
st.write("🎶 Welcome! This is a step-by-step, resource-efficient app for Carnatic audio exploration.")
st.caption("The app guides you through loading audio, basic analysis, and visualization. Every step gives feedback!")

# --- Step 1: Audio input ---
st.header("Step 1: Load Audio")
src = st.radio("Source", ["YouTube URL", "Upload File"], index=0)
yt_url = st.text_input("YouTube URL (audio only):") if src == "YouTube URL" else None
uploaded = st.file_uploader("Or upload audio file", type=["wav", "mp3", "m4a", "flac", "ogg"]) if src == "Upload File" else None
target_sr = st.selectbox("Sample rate (Hz)", [16000, 22050, 32000, 44100], index=1)
load_btn = st.button("Load Audio", type="primary")

if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None
if "audio_ready" not in st.session_state:
    st.session_state["audio_ready"] = False
if "audio_error" not in st.session_state:
    st.session_state["audio_error"] = None

if load_btn:
    st.session_state["audio_ready"] = False
    st.session_state["audio_error"] = None
    import tempfile, shutil, uuid, os
    tmpdir = tempfile.mkdtemp()
    error, audio_path = None, None
    if src == "YouTube URL":
        st.write("Checking for yt-dlp and ffmpeg...")
        if not yt_url or not yt_url.startswith("http"):
            error = "Please paste a valid YouTube URL."
        elif not shutil.which("yt-dlp"):
            error = "yt-dlp not installed."
        elif not shutil.which("ffmpeg"):
            error = "ffmpeg not installed."
        else:
            try:
                out_m4a = os.path.join(tmpdir, f"{uuid.uuid4()}.m4a")
                out_wav = os.path.join(tmpdir, f"{uuid.uuid4()}.wav")
                import subprocess
                st.write("Downloading audio (yt-dlp)...")
                subprocess.run(["yt-dlp", "-x", "--audio-format", "m4a", "-o", out_m4a, yt_url], check=True)
                st.write("Converting to mono WAV (ffmpeg)...")
                subprocess.run(["ffmpeg", "-y", "-i", out_m4a, "-ac", "1", "-ar", str(target_sr), out_wav], check=True)
                audio_path = out_wav
            except Exception as e:
                error = f"Audio download failed: {e}"
    else:
        if uploaded is None:
            error = "Please upload an audio file."
        else:
            upath = os.path.join(tmpdir, uploaded.name)
            with open(upath, "wb") as f: f.write(uploaded.read())
            audio_path = upath
    if error:
        st.session_state["audio_error"] = error
        st.session_state["audio_path"] = None
    elif audio_path:
        st.session_state["audio_path"] = audio_path
        st.session_state["audio_ready"] = True
        st.session_state["audio_error"] = None
    else:
        st.session_state["audio_error"] = "Audio could not be loaded."
        st.session_state["audio_path"] = None

# --- Show audio status ---
if st.session_state["audio_error"]:
    st.error(f"Step 1 error: {st.session_state['audio_error']}")
elif st.session_state["audio_ready"]:
    st.success("Audio loaded! Ready for visualization.")
    st.audio(st.session_state["audio_path"])
else:
    st.info("Awaiting audio input...")

# --- Step 2: Visualization menu (shows only if audio loaded) ---
if st.session_state["audio_ready"]:
    st.header("Step 2: Visualize Audio")
    vis_type = st.selectbox("Visualization", ["Waveform", "Volume (RMS)", "Spectrogram", "Pitch Curve", "Swara Lanes"], index=0)
    max_sec = st.slider("Max audio length (for analysis, seconds)", 10, 120, 60, 10)
    plot_btn = st.button("Plot Visualization", type="secondary")
else:
    st.info("Load audio in Step 1 to enable visualization.")

# --- Step 2: Visualization logic ---
if st.session_state["audio_ready"] and plot_btn:
    st.write(f"Preparing '{vis_type}' plot... (shows progress and errors below)")
    try:
        import numpy as np, librosa, matplotlib.pyplot as plt
        y, sr = librosa.load(st.session_state["audio_path"], sr=int(target_sr), mono=True, duration=max_sec)
        duration = len(y)/sr
        st.write(f"Audio loaded for visualization: {duration:.1f} sec at {sr} Hz")
        fig, ax = plt.subplots(figsize=(8, 2))
        if vis_type == "Waveform":
            ax.plot(np.linspace(0, duration, num=len(y)), y, color='dodgerblue', linewidth=0.5)
            ax.set_title("Waveform")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            st.success("Waveform plotted!")
        elif vis_type == "Volume (RMS)":
            hop = 512
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
            ax.plot(times, rms, color='seagreen')
            ax.set_title("Volume (RMS)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("RMS")
            st.pyplot(fig)
            st.success("Volume plotted!")
        elif vis_type == "Spectrogram":
            S = np.abs(librosa.stft(y, n_fft=512))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            img = ax.imshow(S_db, aspect='auto', origin='lower',
                            extent=[0, duration, 0, sr/2], cmap='magma')
            ax.set_title("Spectrogram")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(fig)
            st.success("Spectrogram plotted!")
        elif vis_type == "Pitch Curve":
            st.info("Calculating pitch (YIN, fast)...")
            f0 = librosa.yin(y, fmin=75, fmax=800)
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
            ax.plot(times, f0, '.', ms=2, color='purple')
            ax.set_title("Pitch Curve (YIN)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            st.pyplot(fig)
            st.success("Pitch curve plotted!")
        elif vis_type == "Swara Lanes":
            st.info("Calculating pitch and mapping to Carnatic swaras...")
            f0 = librosa.yin(y, fmin=75, fmax=800)
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
            sa_hz = 132.0  # For demo; in future, auto/manual
            swaras = ["S","R2","G3","M1","P","D2","N3","S'"]
            semitones = [0,2,4,5,7,9,11,12]
            lanes = []
            for hz in f0:
                if hz is None or np.isnan(hz) or hz <= 0: lanes.append(None)
                else:
                    cents = 1200*np.log2(hz/sa_hz)
                    lane = None
                    for s, stn in enumerate(semitones):
                        if abs(cents - stn*100) < 70: lane = s
                    lanes.append(lane)
            ax.plot(times, lanes, '.', alpha=0.5, color='orange')
            ax.set_yticks(range(len(swaras)))
            ax.set_yticklabels(swaras)
            ax.set_title("Swara Lane (Sa=132Hz)")
            ax.set_xlabel("Time (s)")
            st.pyplot(fig)
            st.success("Swara lanes plotted!")
        else:
            st.warning("Unknown visualization selected.")
    except Exception as e:
        st.error(f"Visualization failed: {e}")

# --- Step 3: User help and troubleshooting ---
st.markdown("---")
st.write("""
#### Tips & Troubleshooting
- **Always see feedback:** If nothing happens, check for errors above!
- **Start simple:** Use waveform, volume, or spectrogram for fastest results.
- **Pitch and swara mapping:** These use efficient algorithms (YIN), fast for short audio. For longer files, use the slider to limit length.
- **YouTube download:** Needs `yt-dlp` and `ffmpeg` installed (system or pip).
- **Upload:** Any standard audio file (wav, mp3, m4a, flac, ogg).
- **Debug:** Every step says what's going on. If you get stuck, reload the app.
""")
