# Carnatic Swara Explorer

A fast, interactive Streamlit app for exploring Carnatic music audio:  
- **Stepwise workflow** (no long startup delays)
- **Load audio** from YouTube (with yt-dlp/ffmpeg) or upload a file
- **Choose visualization:** waveform, pitch curve, or Carnatic swara lanes
- **Live status updates:** always tells you what's happening, so you're never left waiting without feedback
- **Analysis runs only when needed** (after you load audio and select an action)
- **Configurable options:** pitch range, Sa (tonic), duration limit, etc.

## Quickstart

1. **Install dependencies**  
   Recommended: use a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **(Optional) Install system tools for YouTube download**
   - `yt-dlp` (Python package or [release](https://github.com/yt-dlp/yt-dlp#installation))
   - `ffmpeg` (system package, e.g. `brew install ffmpeg` on macOS or `sudo apt-get install ffmpeg` on Linux)

3. **Run the app**
   ```bash
   streamlit run ragames/carnatic_swara_app.py
   ```

## Features

- **Step 1:** Load audio  
  - Paste a YouTube link (audio only), or upload your own audio file
  - See status and progress instantly

- **Step 2:** Visualize or analyze  
  - Choose from:  
    - Waveform preview (fast)
    - Pitch curve (fast YIN or accurate pYIN)
    - Swara lane mapping (Carnatic notes, auto/manual Sa)
  - Limit analysis duration for speed
  - Configure pitch range and tonic

- **Live feedback:**  
  - The app always shows what it's doing (downloads, analysis, errors, etc.)
  - No waiting in the dark—every step gives user feedback

- **Future extensions:**  
  - Phrase detection, mini-games, CSV download, custom ragam overlays

## Troubleshooting

- **yt-dlp or ffmpeg not found?**  
  - Install with `pip install yt-dlp` and system package manager for ffmpeg
- **App stuck loading?**  
  - This version only runs heavy code after you click "Load Audio" or "Run Visualization"
  - Check sidebar status messages for errors

## System requirements

- Python 3.9+
- Streamlit >= 1.31
- numpy, pandas, librosa, matplotlib, soundfile, plotly
- yt-dlp, ffmpeg for YouTube audio

## License

MIT

---
*Made with ❤️ for Carnatic music exploration*
