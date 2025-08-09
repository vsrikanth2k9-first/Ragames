# Ragame — Carnatic Swara Visualizer + Mini-Games

- Paste a YouTube link or upload audio
- Pitch → Tonic (Sa) → Swara mapping → 2D/3D visualizations
- Mini-games: identify swaras and transition direction
- Uses 12-TET mapping (custom scale mode supports Pann approximations)

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run carnatic_swara_viz.py
```

Requires `ffmpeg` on PATH:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install -y ffmpeg`
- Windows: https://ffmpeg.org/download.html

## Streamlit Cloud

This repo includes:
- `requirements.txt` for Python deps
- `packages.txt` to install `ffmpeg` apt package on Streamlit Cloud

Deploy steps:
1. Push to GitHub (branch: `main` or `streamlit-app`)
2. On Streamlit Community Cloud, create a new app:
   - Repo: `vsrikanth2k9-first/Ragame`
   - Branch: `streamlit-app` (or `main` after merge)
   - Main file: `carnatic_swara_app.py`
3. Deploy. First build can take a few minutes.

## Notes

- pYIN can be compute-intensive on large files; start with shorter clips.
- Snap threshold (cents) controls how strictly frequencies map to discrete swaras.
- YouTube downloading depends on remote network egress; if blocked, upload audio.
