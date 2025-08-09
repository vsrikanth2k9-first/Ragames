import streamlit as st

st.set_page_config(page_title="Carnatic Swara Explorer", layout="wide")
st.title("Carnatic Swara Explorer")
st.write("🎶 Explore Carnatic music audio! Step by step, fast, and beginner-friendly.")

# ---- Step 1: Load Audio ----
st.header("Step 1: Load Audio (YouTube or File)")
src = st.radio("Audio source", ["YouTube URL", "Upload File"], index=0)
yt_url = st.text_input("Paste YouTube URL") if src == "YouTube URL" else None
uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "flac", "ogg"]) if src == "Upload File" else None
load_btn = st.button("Load Audio")

if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None
if "audio_ready" not in st.session_state:
    st.session_state["audio_ready"] = False

if load_btn:
    import tempfile, shutil, uuid, os
    st.session_state["audio_ready"] = False
    tmpdir = tempfile.mkdtemp()
    error, audio_path = None, None
    if src == "YouTube URL":
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
                subprocess.run(["yt-dlp", "-x", "--audio-format", "m4a", "-o", out_m4a, yt_url], check=True)
                subprocess.run(["ffmpeg", "-y", "-i", out_m4a, "-ac", "1", "-ar", "22050", out_wav], check=True)
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
        st.error(error)
    elif audio_path:
        st.success("Audio loaded!")
        st.session_state["audio_path"] = audio_path
        st.session_state["audio_ready"] = True
        st.audio(audio_path)
    else:
        st.warning("Audio could not be loaded.")

# ---- Step 2: Visualize ----
if st.session_state.get("audio_ready"):
    st.header("Step 2: Simple Visualization")
    st.write("Choose a simple visualization to get started:")
    vis_type = st.radio("Visualization type", ["Waveform", "Volume (RMS)", "Spectrogram"], index=0)
    plot_btn = st.button("Plot Graph")

    if plot_btn:
        st.write("Processing audio...")
        try:
            import numpy as np, librosa, matplotlib.pyplot as plt
            y, sr = librosa.load(st.session_state["audio_path"], sr=22050, mono=True)
            fig, ax = plt.subplots(figsize=(8, 2))
            if vis_type == "Waveform":
                ax.plot(np.linspace(0, len(y)/sr, num=len(y)), y, color='dodgerblue', linewidth=0.5)
                ax.set_title("Waveform")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
            elif vis_type == "Volume (RMS)":
                hop = 512
                rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
                times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
                ax.plot(times, rms, color='seagreen')
                ax.set_title("Volume (RMS)")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("RMS")
            elif vis_type == "Spectrogram":
                S = np.abs(librosa.stft(y, n_fft=512))
                S_db = librosa.amplitude_to_db(S, ref=np.max)
                img = ax.imshow(S_db, aspect='auto', origin='lower',
                                extent=[0, len(y)/sr, 0, sr/2], cmap='magma')
                ax.set_title("Spectrogram")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(fig)
            st.success("Graph plotted!")
        except Exception as e:
            st.error(f"Plotting failed: {e}")

# ---- Footer ----
st.markdown("---")
st.info(
    "Tips:\n"
    "- Start with simple plots (waveform, volume, spectrogram) for instant feedback.\n"
    "- The app only loads and processes audio after you click 'Load Audio'.\n"
    "- If something fails, you'll see an error message here.\n"
    "- For advanced features (pitch, swara), let us know!"
)
