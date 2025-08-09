import sys
import os
import io
import json
import math
import uuid
import shutil
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- Python version guard (keeps earlier behavior) ----------------
PY_MAJOR = sys.version_info.major
PY_MINOR = sys.version_info.minor
PY_VERSION = f"{PY_MAJOR}.{PY_MINOR}"
PY_OK = (PY_MAJOR == 3 and PY_MINOR <= 11)  # Pinned to <=3.11 for librosa/numba/scipy combo

st.set_page_config(page_title="Carnatic Swara Explorer", layout="wide")
st.title("Carnatic Swara Explorer")
st.caption("Audio exploration + Carnatic analyses (ragam identification, swara, sancharam, intonation, manodharma)")

if not PY_OK:
    st.error(
        f"Your Python version is {PY_VERSION}. Packages here are known-good on Python <= 3.11. "
        "Please run with Python 3.11.x for stability."
    )
    st.stop()

# ---------------- Session state ----------------
ss = st.session_state
ss.setdefault("audio_path", None)
ss.setdefault("audio_ready", False)
ss.setdefault("audio_error", None)
ss.setdefault("yt_logs", "")
ss.setdefault("sa_hz", None)  # Tonic cache

# ---------------- Utilities ----------------
def hz_to_cents_ratio(freq, ref):
    # 1200 * log2(freq/ref)
    return 1200.0 * np.log2(np.maximum(freq, 1e-12) / np.maximum(ref, 1e-12))

def wrap_to_octave(cents):
    # Map to [0,1200)
    x = np.mod(cents, 1200.0)
    return x

def nearest_100c_bin(cents):
    # Return integer bin 0..11 and cents deviation (-50, 50]
    bins = np.round(cents / 100.0).astype(int)
    dev = cents - (bins * 100.0)
    # Normalize deviation to (-50, 50]
    dev = ((dev + 50) % 100) - 50
    bins = np.mod(bins, 12)
    return bins, dev

def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, r.stdout, r.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

# ---------------- Step 1: Load Audio ----------------
st.header("Step 1: Load Audio")
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    src = st.radio("Source", ["YouTube URL", "Upload File"], index=1)
with c2:
    target_sr = st.selectbox("Sample rate (Hz)", [16000, 22050, 32000, 44100], index=1)
with c3:
    max_sec = st.slider("Max analysis length (s)", 10, 240, 90, 10)

yt_url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...", disabled=(src!="YouTube URL"))
uploaded = st.file_uploader("Or upload audio file", type=["wav","mp3","m4a","flac","ogg"], disabled=(src!="Upload File"))

cc1, cc2, cc3 = st.columns([1,1,1])
load_btn = cc1.button("Load Audio", type="primary")
update_ytdlp_btn = cc2.button("Update yt-dlp (pip install -U)")
clear_btn = cc3.button("Clear")

if update_ytdlp_btn:
    st.info("Updating yt-dlp via pip...")
    ok, out, err = run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
    st.code((out or "") + "\n" + (err or ""), language="bash")
    if ok:
        st.success("yt-dlp updated. Try Load Audio again.")
    else:
        st.error("Failed to update yt-dlp. See logs above.")

if clear_btn:
    ss["audio_path"] = None
    ss["audio_ready"] = False
    ss["audio_error"] = None
    ss["yt_logs"] = ""
    ss["sa_hz"] = None
    st.experimental_rerun()

if load_btn:
    ss["audio_ready"] = False
    ss["audio_error"] = None
    ss["yt_logs"] = ""
    tmpdir = tempfile.mkdtemp()
    try:
        if src == "YouTube URL":
            if not yt_url or not yt_url.startswith("http"):
                ss["audio_error"] = "Please paste a valid YouTube URL."
            elif not shutil.which("ffmpeg"):
                ss["audio_error"] = "ffmpeg not installed (add 'ffmpeg' to packages.txt)."
            else:
                out_m4a = os.path.join(tmpdir, f"{uuid.uuid4()}.m4a")
                out_wav = os.path.join(tmpdir, f"{uuid.uuid4()}.wav")
                # Try yt-dlp, capture logs
                cmd = ["yt-dlp", "-x", "--audio-format", "m4a", "-o", out_m4a, yt_url]
                ok, out, err = run_cmd(cmd)
                ss["yt_logs"] = f"$ {' '.join(cmd)}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                if not ok:
                    ss["audio_error"] = "YouTube download failed. See yt-dlp logs below. Try a simpler public URL."
                else:
                    # Convert to mono wav at target_sr
                    cmd2 = ["ffmpeg", "-y", "-i", out_m4a, "-ac", "1", "-ar", str(target_sr), out_wav]
                    ok2, out2, err2 = run_cmd(cmd2)
                    ss["yt_logs"] += f"\n$ {' '.join(cmd2)}\nSTDOUT:\n{out2}\nSTDERR:\n{err2}"
                    if not ok2:
                        ss["audio_error"] = "ffmpeg conversion failed. See logs below."
                    else:
                        ss["audio_path"] = out_wav
                        ss["audio_ready"] = True
        else:
            if uploaded is None:
                ss["audio_error"] = "Please upload an audio file."
            else:
                upath = os.path.join(tmpdir, uploaded.name)
                with open(upath, "wb") as f:
                    f.write(uploaded.read())
                # Optional resample to target_sr via librosa upon load step 2 to avoid ffmpeg here
                ss["audio_path"] = upath
                ss["audio_ready"] = True
    except Exception as e:
        ss["audio_error"] = f"Audio load failed: {e}"

# Status
if ss["audio_error"]:
    st.error(f"Step 1 error: {ss['audio_error']}")
    if ss["yt_logs"]:
        with st.expander("yt-dlp / ffmpeg logs"):
            st.code(ss["yt_logs"], language="bash")
elif ss["audio_ready"]:
    st.success("Audio loaded! Ready for visualization and analysis.")
    st.audio(ss["audio_path"])
else:
    st.info("Awaiting audio input...")

# ---------------- Step 2: Visualizations (unchanged features preserved) ----------------
if ss["audio_ready"]:
    st.header("Step 2: Visualize Audio")
    vis_type = st.selectbox("Visualization", ["Waveform", "Volume (RMS)", "Spectrogram", "Pitch Curve", "Swara Lanes"], index=0)
    plot_btn = st.button("Plot Visualization", type="secondary")

    if plot_btn:
        try:
            import librosa
            y, sr = librosa.load(ss["audio_path"], sr=int(target_sr), mono=True, duration=int(max_sec))
            duration = len(y) / sr
            fig, ax = plt.subplots(figsize=(9, 2.6))
            if vis_type == "Waveform":
                t = np.linspace(0, duration, num=len(y))
                ax.plot(t, y, color='dodgerblue', linewidth=0.5)
                ax.set_title("Waveform"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
                st.pyplot(fig); st.success("Waveform plotted.")
            elif vis_type == "Volume (RMS)":
                hop = 512
                import librosa.feature
                rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
                times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
                ax.plot(times, rms, color='seagreen')
                ax.set_title("Volume (RMS)"); ax.set_xlabel("Time (s)"); ax.set_ylabel("RMS")
                st.pyplot(fig); st.success("RMS plotted.")
            elif vis_type == "Spectrogram":
                S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
                S_db = librosa.amplitude_to_db(S, ref=np.max)
                img = ax.imshow(S_db, aspect='auto', origin='lower',
                                extent=[0, duration, 0, sr/2], cmap='magma')
                ax.set_title("Spectrogram"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                st.pyplot(fig); st.success("Spectrogram plotted.")
            elif vis_type == "Pitch Curve":
                fmin, fmax = 70, 1000
                try:
                    f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256)
                except Exception:
                    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256)
                times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=256)
                ax.plot(times, f0, '.', ms=2, color='purple')
                ax.set_title("Pitch Curve"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
                st.pyplot(fig); st.success("Pitch curve plotted.")
            elif vis_type == "Swara Lanes":
                fmin, fmax = 70, 1000
                try:
                    f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256)
                except Exception:
                    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256)
                times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=256)
                # Auto or cached Sa
                voiced = np.where(np.isfinite(f0) & (f0 > 0))[0]
                sa_hz = ss.get("sa_hz")
                if sa_hz is None:
                    if len(voiced) > 0:
                        med = np.median(f0[voiced])
                        # normalize to comfortable register [110, 300] Hz
                        while med > 300: med /= 2
                        while med < 110: med *= 2
                        sa_hz = med
                    else:
                        sa_hz = 132.0
                    ss["sa_hz"] = float(sa_hz)
                cents = hz_to_cents_ratio(f0, sa_hz)
                cents_wrapped = wrap_to_octave(cents)
                bins, dev = nearest_100c_bin(cents_wrapped)
                ax.plot(times, bins, '.', alpha=0.5, color='orange')
                ax.set_yticks(range(12)); ax.set_yticklabels(["S","R1/R2","R3","G2/G3","M1","M2","P","D1/D2","D3","N2/N3","S' (11)","(12=0)"])
                ax.set_title(f"Swara Lanes (Sa≈{sa_hz:.1f} Hz)")
                ax.set_xlabel("Time (s)")
                st.pyplot(fig); st.success("Swara lanes plotted.")
        except Exception as e:
            st.error(f"Visualization failed: {e}")

# ---------------- Step 3: Analyses ----------------
if ss["audio_ready"]:
    st.header("Step 3: Carnatic Analyses")

    with st.expander("Tonic (Sa) settings", expanded=True):
        colA, colB, colC = st.columns([1.3,1,1])
        auto_sa = colA.checkbox("Auto-detect Sa", value=True)
        default_sa = 132.0
        sa_manual = colB.number_input("Sa (Hz)", min_value=60.0, max_value=400.0, value=float(ss.get("sa_hz") or default_sa), step=1.0, format="%.1f")
        if colC.button("Reset Sa to auto"):
            ss["sa_hz"] = None
            st.experimental_rerun()

    run_analyses = st.button("Run Analyses", type="primary")

    if run_analyses:
        import librosa
        # Load audio
        y, sr = librosa.load(ss["audio_path"], sr=int(target_sr), mono=True, duration=int(max_sec))
        fmin, fmax = 70, 1000
        try:
            f0, vflag, vprob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256)
        except Exception:
            f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256)
            vflag = np.isfinite(f0)
            vprob = None
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=256)
        voiced_mask = (np.isfinite(f0) & (f0 > 0))
        f0_voiced = f0[voiced_mask]

        # Tonic detection or manual
        if auto_sa:
            if len(f0_voiced) > 10:
                med = np.median(f0_voiced)
                while med > 300: med /= 2
                while med < 110: med *= 2
                sa_hz = float(med)
            else:
                sa_hz = float(ss.get("sa_hz") or default_sa)
        else:
            sa_hz = float(sa_manual)
        ss["sa_hz"] = sa_hz

        # Compute cents tracks
        cents = hz_to_cents_ratio(np.where(voiced_mask, f0, np.nan), sa_hz)
        cents_wrapped = np.where(voiced_mask, wrap_to_octave(cents), np.nan)
        bins12, dev_cents = nearest_100c_bin(cents_wrapped)

        # Note: in bins12, unvoiced frames have arbitrary values; mask them
        bins12_masked = np.where(voiced_mask, bins12, -1)

        # Swara histogram (12-TET bins as proxy for swaras)
        valid_bins = bins12[bins12_masked >= 0]
        hist12 = np.bincount(valid_bins, minlength=12).astype(float)
        hist12_pct = 100.0 * hist12 / np.maximum(hist12.sum(), 1)

        # Intonation accuracy (mean absolute deviation to nearest 100c lane)
        valid_dev = dev_cents[voiced_mask]
        intonation_mae = float(np.nanmean(np.abs(valid_dev))) if valid_dev.size else float("nan")

        # Stability vs ornamentation
        # Compute local derivative and local std on cents (voiced)
        cents_cont = cents.copy()
        # Interpolate small gaps to compute derivatives more robustly
        if np.any(voiced_mask):
            # Simple forward fill/back fill
            c = cents_cont.copy()
            idx = np.where(voiced_mask)[0]
            if len(idx) > 0:
                first, last = idx[0], idx[-1]
                c[:first] = c[first]
                c[last+1:] = c[last]
                # linear interpolate nans inside
                nans = np.isnan(c)
                if np.any(nans):
                    xp = np.where(~nans)[0]
                    fp = c[~nans]
                    x = np.arange(len(c))
                    c[nans] = np.interp(x[nans], xp, fp)
                cents_cont = c
        # Derivative
        dc = np.abs(np.diff(cents_cont, prepend=cents_cont[0]))
        # Local std in a 7-frame window (~80 ms @ 22050/256)
        from collections import deque
        win = 7
        local_std = np.full_like(cents_cont, np.nan, dtype=float)
        q = deque(maxlen=win)
        for i, val in enumerate(cents_cont):
            q.append(val)
            if len(q) == win:
                local_std[i] = np.std(q)
        # Metrics
        stable_ratio = float(np.nanmean((dc < 5.0)[voiced_mask])) if np.any(voiced_mask) else float("nan")
        ornament_ratio = float(np.nanmean((local_std > 20.0)[voiced_mask])) if np.any(voiced_mask) else float("nan")

        # Range usage in cents (unwrapped)
        cents_unwrapped = cents.copy()
        # Unwrap by adding/subtracting 1200 when large jumps occur
        for i in range(1, len(cents_unwrapped)):
            if not np.isfinite(cents_unwrapped[i-1]) or not np.isfinite(cents_unwrapped[i]): 
                continue
            d = cents_unwrapped[i] - cents_unwrapped[i-1]
            if d > 600: cents_unwrapped[i:] -= 1200
            elif d < -600: cents_unwrapped[i:] += 1200
        used_range_cents = float(np.nanmax(cents_unwrapped) - np.nanmin(cents_unwrapped)) if np.any(voiced_mask) else float("nan")

        # Transition matrix (12x12) on run-length compressed swara bins to reduce self-hops
        seq = bins12_masked[bins12_masked >= 0].tolist()
        # Run-length compression
        rle_seq = []
        for b in seq:
            if len(rle_seq) == 0 or rle_seq[-1] != b:
                rle_seq.append(b)
        trans = np.zeros((12,12), dtype=float)
        for i in range(len(rle_seq)-1):
            a, b = rle_seq[i], rle_seq[i+1]
            trans[a, b] += 1.0
        trans_pct = 100.0 * trans / np.maximum(trans.sum(), 1)

        # N-gram phrases (sancharam proxies)
        def top_ngrams(s, n=3, topk=10):
            counts = {}
            for i in range(len(s)-n+1):
                tup = tuple(s[i:i+n])
                counts[tup] = counts.get(tup, 0) + 1
            items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            return items[:topk]
        top_bi = top_ngrams(rle_seq, n=2, topk=12)
        top_tri = top_ngrams(rle_seq, n=3, topk=12)

        # Raga identification by scale-template matching (few common melakartas)
        # Templates are 12-TET approximations relative to Sa
        RAGA_TEMPLATES = {
            "Shankarabharanam (M1)": {0,2,4,5,7,9,11},
            "Kalyani (M2)": {0,2,4,6,7,9,11},
            "Kharaharapriya (M1)": {0,2,3,5,7,9,10},
            "Harikambhoji (M1)": {0,2,4,5,7,9,10},
            "Natabhairavi (M1)": {0,2,3,5,7,8,10},
            "Charukesi (M1)": {0,2,4,5,7,8,11},
            "Mayamalavagowla (M1)": {0,1,4,5,7,8,11},
            "Todi/Hanumatodi (M1)": {0,1,3,5,7,8,10},
        }
        def raga_score(allowed: set, bins: np.ndarray, dev: np.ndarray, mask: np.ndarray):
            b = bins[mask]
            d = dev[mask]
            if b.size == 0: 
                return 0.0, 0.0, 0.0
            in_scale = np.isin(b, list(allowed))
            scale_adherence = np.mean(in_scale)
            # Intonation on only in-scale frames
            in_dev = np.abs(d[in_scale]) if np.any(in_scale) else np.array([])
            mae = float(np.mean(in_dev)) if in_dev.size else 50.0
            # Smoothness proxy: fraction of small steps between allowed bins in run-length sequence
            seq_allowed = [x for x in rle_seq if x in allowed]
            if len(seq_allowed) > 1:
                steps = [abs(seq_allowed[i+1]-seq_allowed[i])%12 for i in range(len(seq_allowed)-1)]
                smooth = np.mean([1.0 if (s in (0,1,2)) else 0.0 for s in steps])
            else:
                smooth = 0.0
            # Combine into conformity score (0..100)
            # Heuristic weights: adherence 0.6, intonation 0.2, smoothness 0.2
            score = 100.0*(0.6*scale_adherence + 0.2*(1.0 - min(mae,50.0)/50.0) + 0.2*smooth)
            return float(score), float(scale_adherence), float(mae)
        raga_results = []
        mask_valid = voiced_mask & np.isfinite(cents_wrapped)
        for name, allowed in RAGA_TEMPLATES.items():
            s, adh, mae = raga_score(allowed, bins12, dev_cents, mask_valid)
            raga_results.append((name, s, adh, mae))
        raga_results.sort(key=lambda x: x[1], reverse=True)
        top3 = raga_results[:3]

        # Manodharma extent proxy
        # Diversity in notes and transitions + range + ornamentation ratio
        def shannon_entropy(p):
            p = p[p>0]
            return -float(np.sum(p*np.log2(p))) if p.size else 0.0
        p_notes = hist12 / np.maximum(hist12.sum(), 1)
        H_notes = shannon_entropy(p_notes) / math.log2(12)  # 0..1
        p_trans = trans.flatten() / np.maximum(trans.sum(), 1)
        H_trans = shannon_entropy(p_trans) / math.log2(12*12)
        range_norm = min(max(used_range_cents/2400.0, 0.0), 1.0)  # clamp, 2 octaves -> 1.0
        ornament_norm = min(max(ornament_ratio, 0.0), 1.0) if not math.isnan(ornament_ratio) else 0.0
        manodharma_index = 100.0 * (0.35*H_notes + 0.35*H_trans + 0.2*range_norm + 0.1*ornament_norm)

        # Report dictionary
        report = {
            "sa_hz": sa_hz,
            "intonation_mae_cents": intonation_mae,
            "stable_ratio": stable_ratio,
            "ornament_ratio": ornament_ratio,
            "used_range_cents": used_range_cents,
            "hist12_pct": hist12_pct.tolist(),
            "top_ragas": [{"name": n, "conformity": s, "scale_adherence": adh, "intonation_mae_cents": mae} for (n,s,adh,mae) in top3],
            "manodharma_index": manodharma_index,
            "top_bigrams": [{"pattern": list(map(int,t)), "count": int(c)} for (t,c) in top_bi],
            "top_trigrams": [{"pattern": list(map(int,t)), "count": int(c)} for (t,c) in top_tri],
        }

        # ----- UI outputs -----
        cA, cB = st.columns([1,1])
        with cA:
            st.subheader("Ragam identification (top 3)")
            for name, score, adh, mae in top3:
                st.write(f"- {name}: {score:.1f} / 100 (adherence {adh*100:.1f}%, intonation MAE {mae:.1f} cents)")
            st.write(f"Estimated Sa: {sa_hz:.1f} Hz")
            st.write(f"Overall intonation MAE: {intonation_mae:.1f} cents")

        with cB:
            st.subheader("Manodharma extent (index)")
            st.write(f"- Manodharma Index: {manodharma_index:.1f} / 100")
            st.write(f"- Stability ratio (held notes): {stable_ratio*100:.1f}%")
            st.write(f"- Ornamentation ratio (gamaka activity): {ornament_ratio*100:.1f}%")
            st.write(f"- Pitch range used: {used_range_cents:.0f} cents")

        # Swara histogram plot
        fig1, ax1 = plt.subplots(figsize=(7,2.6))
        labels12 = ["S","r/R","R3","g/G","M1","M2","P","d/D","D3","n/N","S'11","(12)"]
        ax1.bar(np.arange(12), hist12_pct, color="teal")
        ax1.set_xticks(np.arange(12)); ax1.set_xticklabels(labels12, rotation=0)
        ax1.set_ylabel("% time"); ax1.set_title("Swara distribution (12-TET lanes)")
        st.pyplot(fig1)

        # Transition heatmap
        fig2, ax2 = plt.subplots(figsize=(5.6,5.0))
        im = ax2.imshow(trans_pct, origin="lower", cmap="viridis")
        ax2.set_xticks(np.arange(12)); ax2.set_xticklabels(labels12, rotation=90)
        ax2.set_yticks(np.arange(12)); ax2.set_yticklabels(labels12)
        ax2.set_title("Swara transition heatmap (% of transitions)")
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2)

        # Phrases (sancharam proxies)
        st.subheader("Swara sancharam analysis (frequent phrases)")
        def bins_to_swaras(bins_tuple):
            mapping = ["S","R1/2","R3","G2/3","M1","M2","P","D1/2","D3","N2/3","S'11","(12)"]
            return "-".join(mapping[b] for b in bins_tuple)
        st.write("- Top bigrams:")
        for (pat, cnt) in top_bi:
            st.write(f"  • {bins_to_swaras(pat)}  (x{cnt})")
        st.write("- Top trigrams:")
        for (pat, cnt) in top_tri:
            st.write(f"  • {bins_to_swaras(pat)}  (x{cnt})")

        # Raga Conformity / Perfection score: take best-matching raga score as "perfection for that raga"
        best_name, best_score, best_adh, best_mae = top3[0] if len(top3) else ("N/A",0.0,0.0,50.0)
        st.subheader("Raga conformity / Perfection")
        st.write(f"- Best match: {best_name} — Conformity score: {best_score:.1f} / 100")
        st.write(f"- Scale adherence: {best_adh*100:.1f}% | Intonation MAE: {best_mae:.1f} cents")

        # Download JSON report
        report_bytes = json.dumps(report, indent=2).encode("utf-8")
        st.download_button("Download analysis report (JSON)", data=report_bytes, file_name="carnatic_analysis_report.json", mime="application/json")

# ---------------- Help / Tips ----------------
st.markdown("---")
st.write("Tips & Troubleshooting:")
st.write(f"- Python: {PY_VERSION}. Use Python 3.11.x if you run into build issues with numpy/scipy/numba/librosa.")
st.write("- Streamlit Cloud: add 'ffmpeg' to packages.txt. Install 'yt-dlp' via requirements.txt (not packages.txt).")
st.write("- If YouTube keeps failing, use a simple public URL or upload a file. Use 'Update yt-dlp' to get latest extractors.")
