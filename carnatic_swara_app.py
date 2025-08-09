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

st.set_page_config(page_title="Carnatic Swara Explorer", layout="wide")
st.title("Carnatic Swara Explorer")
st.caption("Fully automatic analysis: raga (scale-only), swara lanes, transitions, intonation, manodharma")

# ---------------- Session state ----------------
ss = st.session_state
ss.setdefault("audio_path", None)
ss.setdefault("audio_ready", False)
ss.setdefault("audio_error", None)
ss.setdefault("yt_logs", "")

# ---------------- Utilities ----------------
def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, r.stdout, r.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def hz_to_cents_ratio(freq, ref):
    return 1200.0 * np.log2(np.maximum(freq, 1e-12) / np.maximum(ref, 1e-12))

def wrap_to_octave(cents):
    return np.mod(cents, 1200.0)

def smooth_interp_nan(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return x
    valid = np.isfinite(x)
    if not np.any(valid):
        return x
    # forward/back fill
    first, last = np.argmax(valid), n - 1 - np.argmax(valid[::-1])
    x[:first] = x[first]
    x[last+1:] = x[last]
    # linear interp gaps
    nans = ~np.isfinite(x)
    if np.any(nans):
        xp = np.where(~nans)[0]
        fp = x[~nans]
        xi = np.where(nans)[0]
        x[nans] = np.interp(xi, xp, fp)
    return x

def f0_quality_no_sa(f0: np.ndarray):
    # Quality without needing tonic: voiced coverage and pitch smoothness in relative cents
    f = f0[np.isfinite(f0) & (f0 > 0)]
    voiced_ratio = float(len(f) / max(1, len(f0)))
    if len(f) < 5:
        return voiced_ratio * 0.2, {"voiced_ratio": voiced_ratio, "jitter_cents": float("inf")}
    # Relative cents differences (independent of tonic)
    d = np.diff(np.log2(f)) * 1200.0
    d = d[np.isfinite(d)]
    if d.size == 0:
        jitter = 999.0
    else:
        jitter = float(np.median(np.abs(d)))  # median absolute change per frame (cents)
    # Map jitter to [0,1]: 0c -> 1.0, 25c+ -> ~0
    jscore = max(0.0, 1.0 - min(jitter / 25.0, 1.0))
    q = voiced_ratio * (0.5 + 0.5 * jscore)
    return q, {"voiced_ratio": voiced_ratio, "jitter_cents": jitter}

def choose_best_f0(y, sr):
    import librosa
    fmin, fmax = 70, 1000
    # Candidate 1: pYIN
    try:
        f0a, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256)
    except Exception:
        f0a = None
    # Candidate 2: YIN
    try:
        f0b = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256)
    except Exception:
        f0b = None

    candidates = []
    if f0a is not None:
        qa, ma = f0_quality_no_sa(f0a), "pyin"
        candidates.append((qa[0], f0a, ma, qa[1]))
    if f0b is not None:
        qb, mb = f0_quality_no_sa(f0b), "yin"
        candidates.append((qb[0], f0b, mb, qb[1]))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, f0, method, metrics = candidates[0]

    # Smooth and interpolate NaNs lightly (for stability of analyses)
    f0s = smooth_interp_nan(f0)
    return f0s, method, {"quality": score, **metrics}

def auto_sa_from_hist(f0_hz, prefer_range=(110, 300)):
    f = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    if f.size == 0:
        return 132.0, 0.0
    # Median seed
    sa = float(np.median(f))
    while sa > prefer_range[1]:
        sa /= 2.0
    while sa < prefer_range[0]:
        sa *= 2.0

    # Refine using pitch-class histogram peak
    cents = 1200.0 * np.log2(f / sa)
    pc = np.mod(cents, 1200.0)
    if pc.size == 0:
        return sa, 0.0
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    hist, edges = np.histogram(pc, bins=240, range=(0, 1200))
    smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)
    peak_bin = int(np.argmax(smooth))
    # shift so peak is at 0 cents
    peak_center = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0
    # Adjust tonic so that this peak maps to 0 cents
    sa_refined = sa * (2 ** (peak_center / 1200.0))
    # Confidence from peak prominence
    prom = float(np.max(smooth) / max(1.0, np.mean(smooth) + 1e-6))
    conf = max(0.0, min((prom - 1.2) / 3.0, 1.0))  # rough 0..1
    return sa_refined, conf

def learn_lanes_from_audio(f0_hz, sa_hz):
    cents = hz_to_cents_ratio(f0_hz, sa_hz)
    pc = np.mod(cents[np.isfinite(cents)], 1200.0)
    if pc.size == 0:
        return np.array([0.0])
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    hist, edges = np.histogram(pc, bins=240, range=(0, 1200))  # 5-cent bins
    smooth = gaussian_filter1d(hist.astype(float), sigma=2.2)
    # Peaks separated by >= 25 cents, with height threshold
    peaks, props = find_peaks(smooth, height=np.max(smooth) * 0.15, distance=5)
    if peaks.size == 0:
        return np.array([0.0])
    centers = (edges[peaks] + edges[peaks + 1]) / 2.0

    # Recenter so strongest peak is 0 (Sa)
    sa_idx = int(np.argmax(props["peak_heights"]))
    sa_pc = centers[sa_idx]
    centers = (centers - sa_pc) % 1200

    # Keep top-K most prominent (to avoid spurious peaks)
    order = np.argsort(-props["peak_heights"])
    centers = np.sort(centers[order][:8])  # up to 8 lanes
    return centers

def assign_frames_to_lanes(cents_wrapped, lane_centers):
    if lane_centers.size == 0:
        return np.full(len(cents_wrapped), -1), np.full(len(cents_wrapped), np.nan)
    diffs = np.abs((cents_wrapped[:, None] - lane_centers[None, :] + 600) % 1200 - 600)
    lane_idx = np.nanargmin(diffs, axis=1)
    dev_cents = np.take_along_axis(diffs, lane_idx[:, None], axis=1)[:, 0]
    return lane_idx, dev_cents

def window_scores(f0):
    # Score overlapping windows for cleanliness; 45s windows, 50% overlap
    n = len(f0)
    if n < 10:
        return [(0, n, 0.0)]
    win = 256 * 45  # approx frames for ~45s with hop 256 at 22050Hz (roughly)
    win = max(win, 600)  # ensure minimum
    hop = win // 2
    scores = []
    for start in range(0, n - 1, hop):
        end = min(n, start + win)
        seg = f0[start:end]
        valid = np.isfinite(seg) & (seg > 0)
        vr = float(np.mean(valid)) if end > start else 0.0
        if np.any(valid):
            d = np.diff(np.log2(seg[valid])) * 1200.0
            d = d[np.isfinite(d)]
            jitter = float(np.median(np.abs(d))) if d.size else 999.0
        else:
            jitter = 999.0
        jscore = max(0.0, 1.0 - min(jitter / 25.0, 1.0))
        s = vr * (0.5 + 0.5 * jscore)
        scores.append((start, end, s))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores

def select_frames_mask(n_frames, top_windows):
    mask = np.zeros(n_frames, dtype=bool)
    for (s, e, _) in top_windows:
        mask[s:e] = True
    return mask

def quantize_centers_to_steps(centers):
    steps = np.round(centers / 100.0).astype(int) % 12
    if 0 not in steps:
        steps = np.append(steps, 0)
    return sorted(set(int(x) for x in steps))

# Neutral, scale-only raga sets (12-TET steps relative to Sa)
RAGA_SCALES = {
    "Shankarabharanam":  {0,2,4,5,7,9,11},
    "Kalyani":           {0,2,4,6,7,9,11},
    "Kharaharapriya":    {0,2,3,5,7,9,10},
    "Harikambhoji":      {0,2,4,5,7,9,10},
    "Natabhairavi":      {0,2,3,5,7,8,10},
    "Charukesi":         {0,2,4,5,7,8,11},
    "Mayamalavagowla":   {0,1,4,5,7,8,11},
    "Todi (Hanumatodi)": {0,1,3,5,7,8,10},
    "Bhairavi (scale)":  {0,1,3,5,7,8,10},  # scale-wise overlaps Todi/hidden differences
    "Sankarabharanam":   {0,2,4,5,7,9,11},  # alias
}

def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))

def raga_guess_scale_only(lane_centers):
    obs_steps = set(quantize_centers_to_steps(lane_centers))
    scored = [(name, jaccard(obs_steps, steps)) for name, steps in RAGA_SCALES.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    top3 = scored[:3]
    confidence = top3[0][1] if top3 else 0.0
    return top3, confidence, sorted(obs_steps)

# ---------------- Step 1: Load Audio ----------------
st.header("Step 1: Load Audio")
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    src = st.radio("Source", ["YouTube URL", "Upload File"], index=1)
with c2:
    target_sr = st.selectbox("Sample rate (Hz)", [16000, 22050, 32000, 44100], index=1)
with c3:
    max_sec = st.slider("Max analysis length (s)", 10, 240, 90, 10)

yt_url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...", disabled=(src != "YouTube URL"))
uploaded = st.file_uploader("Or upload audio file", type=["wav", "mp3", "m4a", "flac", "ogg"], disabled=(src != "Upload File"))

cc1, cc2, cc3 = st.columns([1, 1, 1])
load_btn = cc1.button("Load Audio", type="primary")
update_ytdlp_btn = cc2.button("Update yt-dlp (pip install -U)")
clear_btn = cc3.button("Clear")

if update_ytdlp_btn:
    st.info("Updating yt-dlp via pip...")
    ok, out, err = run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
    st.code((out or "") + "\n" + (err or ""), language="bash")
    st.success("yt-dlp updated. Try Load Audio again." if ok else "Failed to update yt-dlp. See logs above.")

if clear_btn:
    ss["audio_path"] = None
    ss["audio_ready"] = False
    ss["audio_error"] = None
    ss["yt_logs"] = ""
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
                cmd = ["yt-dlp", "-x", "--audio-format", "m4a", "-o", out_m4a, yt_url]
                ok, out, err = run_cmd(cmd)
                ss["yt_logs"] = f"$ {' '.join(cmd)}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                if not ok:
                    ss["audio_error"] = "YouTube download failed. See yt-dlp logs below. Try a public URL or upload a file."
                else:
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
                ss["audio_path"] = upath
                ss["audio_ready"] = True
    except Exception as e:
        ss["audio_error"] = f"Audio load failed: {e}"

if ss["audio_error"]:
    st.error(f"Step 1 error: {ss['audio_error']}")
    if ss["yt_logs"]:
        with st.expander("yt-dlp / ffmpeg logs"):
            st.code(ss["yt_logs"], language="bash")
elif ss["audio_ready"]:
    st.success("Audio loaded! Ready for analysis.")
    st.audio(ss["audio_path"])
else:
    st.info("Awaiting audio input...")

# ---------------- Step 2: Visualize (optional quick plots) ----------------
if ss["audio_ready"]:
    st.header("Step 2: Quick Visualizations")
    import librosa
    y, sr = librosa.load(ss["audio_path"], sr=int(target_sr), mono=True, duration=int(max_sec))
    vis = st.selectbox("Visualization", ["Waveform", "Spectrogram"], index=0)
    if st.button("Plot"):
        fig, ax = plt.subplots(figsize=(9, 3))
        if vis == "Waveform":
            t = np.linspace(0, len(y)/sr, num=len(y))
            ax.plot(t, y, lw=0.5)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.set_title("Waveform")
        else:
            S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            ax.imshow(S_db, aspect='auto', origin='lower',
                      extent=[0, len(y)/sr, 0, sr/2], cmap='magma')
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)"); ax.set_title("Spectrogram (dB)")
        st.pyplot(fig)

# ---------------- Step 3: Fully Automatic Analyses ----------------
if ss["audio_ready"]:
    st.header("Step 3: Analyses (Automatic)")
    if st.button("Analyze (Auto)", type="primary"):
        import librosa
        y, sr = librosa.load(ss["audio_path"], sr=int(target_sr), mono=True, duration=int(max_sec))

        # Pitch-tracker ensemble
        f0, method, qmetrics = choose_best_f0(y, sr)
        if f0 is None:
            st.error("Pitch tracking failed. Try a clearer portion or another file.")
            st.stop()

        # Auto window selection (top 3 clean windows)
        scores = window_scores(f0)
        top_windows = scores[:3] if scores else []
        mask = select_frames_mask(len(f0), top_windows) if top_windows else np.ones(len(f0), dtype=bool)

        # Auto tonic and swara lanes (data-driven, no tailoring)
        sa_hz, sa_conf = auto_sa_from_hist(f0[mask])
        cents = hz_to_cents_ratio(f0, sa_hz)
        cents_wrapped = np.where(np.isfinite(cents), wrap_to_octave(cents), np.nan)
        lane_centers = learn_lanes_from_audio(f0[mask], sa_hz)
        lane_idx, dev_cents = assign_frames_to_lanes(cents_wrapped, lane_centers)

        # Metrics
        voiced_mask = np.isfinite(cents_wrapped)
        use_mask = voiced_mask & mask
        intonation_mae = float(np.nanmean(np.abs(dev_cents[use_mask]))) if np.any(use_mask) else float("nan")

        # Swara distribution (by learned lanes)
        if np.any(use_mask):
            histL = np.bincount(lane_idx[use_mask], minlength=len(lane_centers)).astype(float)
            histL_pct = 100.0 * histL / max(1.0, histL.sum())
        else:
            histL_pct = np.zeros(len(lane_centers))

        # Transition matrix on run-length compressed lane sequence
        seq = lane_idx[use_mask]
        seq = seq[np.isfinite(seq)]
        seq = seq.astype(int)
        # Run-length compress
        rle = []
        for b in seq:
            if len(rle) == 0 or rle[-1] != b:
                rle.append(b)
        trans = np.zeros((len(lane_centers), len(lane_centers)), dtype=float)
        for i in range(len(rle) - 1):
            a, b = rle[i], rle[i + 1]
            trans[a, b] += 1.0
        trans_pct = 100.0 * trans / max(1.0, trans.sum())

        # Simple manodharma proxy
        def shannon_entropy(p):
            p = p[p > 0]
            return -float(np.sum(p * np.log2(p))) if p.size else 0.0
        p_notes = (histL / max(1.0, histL.sum())) if np.any(use_mask) else np.array([1.0])
        H_notes = shannon_entropy(p_notes) / math.log2(max(2, len(lane_centers)))
        flat_trans = trans.flatten() / max(1.0, trans.sum())
        H_trans = shannon_entropy(flat_trans) / math.log2(max(4, len(lane_centers) ** 2))
        # Pitch range used (on selected frames, unwrapped)
        cents_unwrapped = cents.copy()
        for i in range(1, len(cents_unwrapped)):
            if not np.isfinite(cents_unwrapped[i-1]) or not np.isfinite(cents_unwrapped[i]):
                continue
            d = cents_unwrapped[i] - cents_unwrapped[i-1]
            if d > 600: cents_unwrapped[i:] -= 1200
            elif d < -600: cents_unwrapped[i:] += 1200
        used_range_cents = float(np.nanmax(cents_unwrapped[use_mask]) - np.nanmin(cents_unwrapped[use_mask])) if np.any(use_mask) else float("nan")
        range_norm = min(max(used_range_cents / 2400.0, 0.0), 1.0)
        # Ornamentation proxy via local variance
        c = smooth_interp_nan(cents)
        dc = np.abs(np.diff(c, prepend=c[0]))
        from collections import deque
        win = 7
        local_std = np.full_like(c, np.nan, dtype=float)
        q = deque(maxlen=win)
        for i, val in enumerate(c):
            q.append(val)
            if len(q) == win:
                local_std[i] = np.std(q)
        ornament_ratio = float(np.nanmean((local_std > 20.0)[use_mask])) if np.any(use_mask) else float("nan")
        stable_ratio = float(np.nanmean((dc < 5.0)[use_mask])) if np.any(use_mask) else float("nan")
        manodharma_index = 100.0 * (0.4 * H_notes + 0.4 * H_trans + 0.15 * range_norm + 0.05 * (ornament_ratio if np.isfinite(ornament_ratio) else 0.0))

        # Scale-only raga guess (no tailoring)
        top3, rg_conf, obs_steps = raga_guess_scale_only(lane_centers)

        # -------- Outputs --------
        cA, cB = st.columns([1, 1])
        with cA:
            st.subheader("Raga (scale-only, automatic)")
            if top3:
                name, conf = top3[0]
                label = f"{name} (confidence {conf:.2f})" if conf >= 0.5 else f"Uncertain — closest: {name} (confidence {conf:.2f})"
                st.write(f"- {label}")
                if len(top3) > 1:
                    st.write(f"- Next: {top3[1][0]} ({top3[1][1]:.2f}), {top3[2][0]} ({top3[2][1]:.2f})")
            else:
                st.write("Unable to estimate a scale reliably.")
            st.write(f"Tonic (Sa): {sa_hz:.1f} Hz (confidence {sa_conf:.2f})")
            st.write(f"Pitch tracker: {method}, quality {qmetrics.get('quality', 0):.2f}, voiced {qmetrics.get('voiced_ratio', 0)*100:.1f}%, jitter {qmetrics.get('jitter_cents', float('nan')):.1f} c")

        with cB:
            st.subheader("Intonation and expression")
            st.write(f"- Intonation MAE: {intonation_mae:.1f} cents")
            st.write(f"- Stability ratio: {stable_ratio*100:.1f}%")
            st.write(f"- Ornamentation ratio: {ornament_ratio*100:.1f}%")
            st.write(f"- Pitch range used: {used_range_cents:.0f} cents")

        # Swara lanes plot (learned lanes)
        fig1, ax1 = plt.subplots(figsize=(8, 2.6))
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=256)
        ax1.scatter(times[use_mask], lane_idx[use_mask], s=6, c="orange", alpha=0.6)
        ax1.set_title("Swara lanes (learned from performance)")
        ax1.set_xlabel("Time (s)")
        ax1.set_yticks(np.arange(len(lane_centers)))
        ax1.set_yticklabels([f"Lane {i+1}" for i in range(len(lane_centers))])
        st.pyplot(fig1)

        # Lane centers (cents) and distribution
        fig2, ax2 = plt.subplots(figsize=(7, 2.4))
        ax2.bar(np.arange(len(lane_centers)), histL_pct, color="teal")
        ax2.set_xticks(np.arange(len(lane_centers)))
        ax2.set_xticklabels([f"{c:.0f}c" for c in lane_centers], rotation=0)
        ax2.set_ylabel("% time"); ax2.set_title("Swara distribution by learned lanes")
        st.pyplot(fig2)

        # Transition heatmap
        fig3, ax3 = plt.subplots(figsize=(5.5, 5.0))
        im = ax3.imshow(trans_pct, origin="lower", cmap="viridis")
        ax3.set_xticks(np.arange(len(lane_centers))); ax3.set_xticklabels([f"L{i+1}" for i in range(len(lane_centers))], rotation=90)
        ax3.set_yticks(np.arange(len(lane_centers))); ax3.set_yticklabels([f"L{i+1}" for i in range(len(lane_centers))])
        ax3.set_title("Swara transitions (% of transitions)")
        fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        st.pyplot(fig3)

        # Frequent phrases (by lanes; data-only)
        def top_ngrams(seq, n=3, topk=10):
            counts = {}
            for i in range(len(seq) - n + 1):
                tup = tuple(seq[i:i+n])
                counts[tup] = counts.get(tup, 0) + 1
            items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            return items[:topk]

        top_bi = top_ngrams(rle, n=2, topk=10)
        top_tri = top_ngrams(rle, n=3, topk=10)

        st.subheader("Swara sancharam (frequent lane patterns)")
        if top_bi:
            st.write("- Top bigrams:")
            for (pat, cnt) in top_bi:
                st.write(f"  • {'-'.join('L'+str(i+1) for i in pat)}  (x{cnt})")
        if top_tri:
            st.write("- Top trigrams:")
            for (pat, cnt) in top_tri:
                st.write(f"  • {'-'.join('L'+str(i+1) for i in pat)}  (x{cnt})")

        # Download JSON report
        report = {
            "sa_hz": sa_hz,
            "sa_confidence": sa_conf,
            "pitch_tracker": method,
            "pitch_tracker_quality": qmetrics,
            "lane_centers_cents": [float(x) for x in lane_centers],
            "lane_hist_pct": [float(x) for x in (histL_pct.tolist() if isinstance(histL_pct, np.ndarray) else histL_pct)],
            "intonation_mae_cents": intonation_mae,
            "stable_ratio": stable_ratio,
            "ornament_ratio": ornament_ratio,
            "used_range_cents": used_range_cents,
            "raga_scale_only_top3": [{"name": n, "confidence": c} for (n, c) in top3],
            "observed_steps_12tet": obs_steps,
        }
        st.download_button("Download analysis report (JSON)", data=json.dumps(report, indent=2).encode("utf-8"),
                           file_name="carnatic_analysis_report.json", mime="application/json")

# ---------------- Tips ----------------
st.markdown("---")
st.write("- Everything runs automatically: tonic detection, lane learning, and raga (scale-only) guess.")
st.write("- If YouTube fails, upload a file instead. Install ffmpeg via packages.txt and yt-dlp via requirements.txt.")
