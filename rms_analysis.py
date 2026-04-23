"""
RMS (Root Mean Square) Analysis for Beatbox Audio Clips
--------------------------------------------------------
Computes RMS energy for one sample clip in detail, then compares
RMS across all clips in the dataset.

What RMS measures:
    RMS = sqrt( (1/N) * sum(x[n]^2) )

    It is the square root of the average squared amplitude.
    For a pure sine wave, RMS = peak_amplitude / sqrt(2).
    For speech/audio, RMS is the standard proxy for perceived loudness
    at a consistent time scale.

Can RMS measure voice loudness?
    YES, with caveats — see the printed explanation at the end.
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
AUDIO_ROOT   = Path("audio_data")
SAMPLE_RATE  = 22050
FRAME_SIZE   = 2048    # ~93 ms per frame
HOP_SIZE     = 512     # ~23 ms between frames
SOUND_LABELS = {"b": "Bass (Kick)", "k": "K-Snare", "psh": "Push Hi-hat", "nu": "Neutral Hum"}
COLORS       = {"b": "#e63946", "k": "#457b9d", "psh": "#2a9d8f", "nu": "#e9c46a"}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def compute_rms(y: np.ndarray, frame_length: int = FRAME_SIZE, hop_length: int = HOP_SIZE) -> np.ndarray:
    """
    Frame-level RMS using librosa.
    Returns an array of shape (n_frames,) in linear amplitude units.
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def rms_to_db(rms: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """
    Convert linear RMS to dBFS (decibels relative to full scale).
    dBFS = 20 * log10(rms / ref)
    0 dBFS = maximum possible amplitude; all real signals are negative dBFS.
    """
    return 20.0 * np.log10(np.clip(rms, 1e-10, None) / ref)


def collect_clips():
    """
    Walk the Phase 1 folders, collect all clips that match a known sound label.
    Returns a list of dicts: {file, participant, label, y, sr}
    """
    clips = []
    for p_dir in sorted(AUDIO_ROOT.iterdir()):
        if not p_dir.is_dir():
            continue
        phase1 = p_dir / "Phase 1"
        if not phase1.exists():
            continue
        participant = p_dir.name
        for wav in sorted(phase1.glob("*.wav")):
            parts = wav.stem.split("-")
            if len(parts) < 2:
                continue
            label = parts[1]
            if label not in SOUND_LABELS:
                continue
            y, sr = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
            clips.append({
                "file":        wav.name,
                "participant": participant,
                "label":       label,
                "y":           y,
                "sr":          sr,
            })
    return clips


# ─── Part 1: Deep dive on one sample clip ────────────────────────────────────

def analyse_single_clip(clip: dict):
    """
    Print and plot a detailed RMS analysis for one clip.
    """
    y, sr, label, fname = clip["y"], clip["sr"], clip["label"], clip["file"]
    duration_s = len(y) / sr
    t_samples  = np.linspace(0, duration_s, len(y))

    # Frame-level RMS
    rms_frames   = compute_rms(y)
    rms_db_frames = rms_to_db(rms_frames)
    t_frames     = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=HOP_SIZE)

    # Single-number summaries
    rms_global   = float(np.sqrt(np.mean(y ** 2)))
    rms_db_global = 20 * np.log10(max(rms_global, 1e-10))
    peak_amp     = float(np.max(np.abs(y)))
    peak_db      = 20 * np.log10(max(peak_amp, 1e-10))
    crest_factor = peak_amp / (rms_global + 1e-10)   # ratio of peak to RMS
    crest_db     = 20 * np.log10(max(crest_factor, 1e-10))
    rms_max_frame = float(np.max(rms_frames))
    rms_max_db   = rms_to_db(np.array([rms_max_frame]))[0]

    print("=" * 64)
    print(f"SINGLE CLIP ANALYSIS: {fname}  [{SOUND_LABELS[label]}]")
    print("=" * 64)
    print(f"  Duration          : {duration_s*1000:.1f} ms  ({len(y)} samples @ {sr} Hz)")
    print(f"  Peak amplitude    : {peak_amp:.4f}  ({peak_db:.1f} dBFS)")
    print(f"  Global RMS        : {rms_global:.4f}  ({rms_db_global:.1f} dBFS)")
    print(f"  Peak RMS (frame)  : {rms_max_frame:.4f}  ({rms_max_db:.1f} dBFS)")
    print(f"  Crest factor      : {crest_factor:.2f}x  ({crest_db:.1f} dB)")
    print(f"  RMS std over time : {float(np.std(rms_frames)):.4f}  (temporal stability)")
    print()
    print("  WHAT EACH NUMBER MEANS:")
    print(f"  • Peak amplitude  = the single loudest sample in the clip.")
    print(f"    {peak_db:.1f} dBFS means the peak reached {100*peak_amp:.1f}% of maximum possible amplitude.")
    print(f"  • Global RMS      = sqrt(mean(x²)) across the entire clip.")
    print(f"    This is the 'average power'. {rms_db_global:.1f} dBFS is the effective loudness.")
    print(f"  • Crest factor    = Peak / RMS = {crest_factor:.2f}x ({crest_db:.1f} dB).")
    print(f"    A high crest factor (>10 dB) means the sound is impulsive (spike then silence).")
    print(f"    A low crest factor means the energy is spread evenly (e.g. sustained hum).")
    print()

    # ── Plot ──
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(f"RMS Deep Dive — {fname}  [{SOUND_LABELS[label]}]", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)

    # Waveform
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_samples * 1000, y, color="#333", lw=0.6, alpha=0.8)
    ax1.axhline(rms_global,  color="#e63946", lw=1.4, linestyle="--", label=f"Global RMS = {rms_global:.3f}")
    ax1.axhline(-rms_global, color="#e63946", lw=1.4, linestyle="--")
    ax1.axhline(peak_amp,  color="#457b9d", lw=1.0, linestyle=":", label=f"Peak = {peak_amp:.3f}")
    ax1.axhline(-peak_amp, color="#457b9d", lw=1.0, linestyle=":")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Waveform with RMS envelope")
    ax1.legend(fontsize=8, loc="upper right")

    # Frame-level RMS (linear)
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(t_frames * 1000, rms_frames, alpha=0.4, color=COLORS[label])
    ax2.plot(t_frames * 1000, rms_frames, color=COLORS[label], lw=1.4)
    ax2.axhline(rms_global, color="#e63946", lw=1.2, linestyle="--", label=f"Global mean RMS")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("RMS (linear)")
    ax2.set_title(f"Frame-level RMS  (frame={FRAME_SIZE} samples, hop={HOP_SIZE} samples)")
    ax2.legend(fontsize=8)

    # Frame-level RMS (dBFS)
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(t_frames * 1000, rms_db_frames, -80, alpha=0.35, color=COLORS[label])
    ax3.plot(t_frames * 1000, rms_db_frames, color=COLORS[label], lw=1.4)
    ax3.axhline(rms_db_global, color="#e63946", lw=1.2, linestyle="--", label=f"Global RMS = {rms_db_global:.1f} dBFS")
    ax3.set_ylim(-80, 5)
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("RMS (dBFS)")
    ax3.set_title("Frame-level RMS in dBFS  (human-perceptual scale)")
    ax3.legend(fontsize=8)

    plt.savefig("rms_single_clip.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved: rms_single_clip.png")
    print()


# ─── Part 2: Compare RMS across all clips ────────────────────────────────────

def compare_all_clips(clips: list):
    """
    Compute global RMS (dBFS) for every clip, group by sound type,
    print a summary table, and plot three comparison figures.
    """
    records = []
    for c in clips:
        rms_global = float(np.sqrt(np.mean(c["y"] ** 2)))
        rms_db     = 20 * np.log10(max(rms_global, 1e-10))
        rms_frames = compute_rms(c["y"])
        records.append({
            "file":        c["file"],
            "participant": c["participant"],
            "label":       c["label"],
            "rms_linear":  rms_global,
            "rms_db":      rms_db,
            "rms_max_db":  float(rms_to_db(np.array([np.max(rms_frames)]))[0]),
            "rms_std":     float(np.std(rms_frames)),
            "duration_ms": len(c["y"]) / c["sr"] * 1000,
        })

    # ── Summary table ──
    print("=" * 64)
    print("CROSS-CLIP RMS COMPARISON")
    print("=" * 64)
    print(f"{'Sound':<8} {'N':>4} {'Mean dBFS':>10} {'Std dBFS':>9} {'Min':>7} {'Max':>7}")
    print("-" * 64)
    for label, name in SOUND_LABELS.items():
        subset = [r for r in records if r["label"] == label]
        if not subset:
            continue
        vals = [r["rms_db"] for r in subset]
        print(f"{name:<30} {len(vals):>4} {np.mean(vals):>9.1f}  {np.std(vals):>8.1f}"
              f"  {np.min(vals):>6.1f}  {np.max(vals):>6.1f}")
    print()

    # Full per-file table
    print(f"{'File':<30} {'Label':<6} {'Global RMS (dBFS)':>18} {'Peak RMS (dBFS)':>16} {'Duration (ms)':>14}")
    print("-" * 90)
    for r in records:
        print(f"{r['file']:<30} {r['label']:<6} {r['rms_db']:>17.1f}  {r['rms_max_db']:>15.1f}  {r['duration_ms']:>13.1f}")
    print()

    # ── Figure 1: Strip plot — RMS per clip, coloured by sound type ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RMS Energy Comparison Across All Clips", fontsize=13, fontweight="bold")

    ax = axes[0]
    for i, (label, name) in enumerate(SOUND_LABELS.items()):
        subset = [r for r in records if r["label"] == label]
        if not subset:
            continue
        vals = [r["rms_db"] for r in subset]
        x    = np.random.default_rng(42).uniform(i - 0.2, i + 0.2, len(vals))
        ax.scatter(x, vals, color=COLORS[label], alpha=0.75, s=55, zorder=3)
        ax.plot([i - 0.35, i + 0.35], [np.mean(vals), np.mean(vals)],
                color=COLORS[label], lw=2.5, zorder=4)
    ax.set_xticks(range(len(SOUND_LABELS)))
    ax.set_xticklabels([SOUND_LABELS[l] for l in SOUND_LABELS], fontsize=9)
    ax.set_ylabel("Global RMS (dBFS)")
    ax.set_title("Global RMS per clip\n(horizontal bar = mean)")
    ax.grid(axis="y", alpha=0.3)

    # ── Figure 2: Bar chart — mean ± std per sound type ──
    ax2 = axes[1]
    labels_ordered = list(SOUND_LABELS.keys())
    means = [np.mean([r["rms_db"] for r in records if r["label"] == l]) for l in labels_ordered]
    stds  = [np.std([r["rms_db"]  for r in records if r["label"] == l]) for l in labels_ordered]
    bars  = ax2.bar(range(len(labels_ordered)),
                    means,
                    yerr=stds,
                    color=[COLORS[l] for l in labels_ordered],
                    alpha=0.8,
                    capsize=5,
                    error_kw={"elinewidth": 1.5})
    ax2.set_xticks(range(len(labels_ordered)))
    ax2.set_xticklabels([SOUND_LABELS[l] for l in labels_ordered], fontsize=9)
    ax2.set_ylabel("Mean Global RMS (dBFS)")
    ax2.set_title("Mean ± Std RMS per sound type\n(higher dBFS = louder recording)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("rms_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved: rms_comparison.png")

    # ── Figure 3: Per-participant RMS heatmap ──
    participants = sorted(set(r["participant"] for r in records))
    labels_list  = list(SOUND_LABELS.keys())
    matrix = np.full((len(participants), len(labels_list)), np.nan)
    for r in records:
        i = participants.index(r["participant"])
        j = labels_list.index(r["label"])
        # If multiple clips per (participant, label), average them
        if np.isnan(matrix[i, j]):
            matrix[i, j] = r["rms_db"]
        else:
            matrix[i, j] = (matrix[i, j] + r["rms_db"]) / 2

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(labels_list)))
    ax.set_xticklabels([SOUND_LABELS[l] for l in labels_list], fontsize=9)
    ax.set_yticks(range(len(participants)))
    ax.set_yticklabels([f"P{p}" for p in participants])
    ax.set_title("Mean Global RMS (dBFS) per Participant × Sound\n(green = louder, red = quieter)", fontsize=11)
    plt.colorbar(im, ax=ax, label="dBFS")
    for i in range(len(participants)):
        for j in range(len(labels_list)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig("rms_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Saved: rms_heatmap.png")
    print()


# ─── Part 3: Can RMS measure voice loudness? (printed explanation) ────────────

def print_loudness_explanation():
    print("=" * 64)
    print("CAN RMS MEASURE THE LOUDNESS OF A VOICE?")
    print("=" * 64)
    print("""
SHORT ANSWER: Yes, with important caveats.

RMS is the most common computational proxy for perceived loudness,
and it works well *within a controlled recording setup*. Here is
what it can and cannot tell you:

WHAT RMS CAPTURES WELL
──────────────────────
• Physical signal power — the average energy of the sound wave.
• Relative loudness between clips recorded under the same
  conditions (same microphone, distance, gain, room).
• The shape of the amplitude envelope over time (attack, sustain,
  decay) which correlates with perceived loudness dynamics.
• Crest factor (Peak/RMS) — distinguishes impulsive sounds (high
  crest, e.g. a kick drum snap) from sustained sounds (low crest,
  e.g. a hum). This is directly relevant to beatbox classification.

WHAT RMS DOES NOT CAPTURE
──────────────────────────
1. Frequency weighting.
   Human hearing is not equally sensitive at all frequencies.
   We are most sensitive around 1–4 kHz and less sensitive at
   low and very high frequencies (ISO 226 equal-loudness contours).
   Two clips with identical RMS can sound very different in loudness
   if one is dominated by 200 Hz bass and the other by 2 kHz.

   Fix: Use A-weighted RMS (dBA) — the industry standard for
   measuring perceived loudness in acoustic measurements.

2. Temporal integration.
   Human perception integrates loudness over ~200 ms windows.
   A very brief loud spike sounds quieter than the same energy
   spread over 200 ms. RMS computed over short frames does not
   account for this.

   Fix: Use LUFS (Loudness Units relative to Full Scale), defined
   in ITU-R BS.1770. This standard applies both frequency weighting
   and 400 ms temporal integration windows. It is used in all
   broadcast and streaming standards.

3. Cross-recording normalisation.
   Comparing RMS across participants only works if all recordings
   were made at the same microphone gain, distance, and room.
   If Participant 3 sat further from the microphone, their RMS
   will appear lower even if they sang identically loudly.

   In this dataset, the heatmap (rms_heatmap.png) will reveal
   if any participant's recordings are systematically quieter —
   which would be a recording condition confound, not a genuine
   loudness difference.

4. Normalised recordings.
   If audio was normalised (gain-adjusted to peak at a fixed level)
   before saving, all peak amplitudes are equal and RMS only
   measures the ratio of sustained energy to peaks, not absolute
   effort.

BOTTOM LINE FOR THIS PROJECT
─────────────────────────────
Within this dataset:
• RMS IS useful for comparing relative loudness within the same
  participant across sound types (do they hit the bass louder
  than the hi-hat?).
• RMS IS useful as a classification feature — the amplitude
  envelope shape (attack time, RMS std, decay rate) differs
  meaningfully between b, k, psh, and nu.
• RMS IS NOT reliable for comparing absolute loudness across
  participants without verifying that all recordings were made
  under identical gain conditions.
• For perceptual loudness matching to a real instrument, use
  LUFS rather than raw RMS.
""")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nLoading all clips...")
    clips = collect_clips()
    print(f"Found {len(clips)} clips across {len(set(c['participant'] for c in clips))} participants.\n")

    # Part 1: single clip deep dive — use the first bass clip found
    sample = next(c for c in clips if c["label"] == "b")
    analyse_single_clip(sample)

    # Part 2: cross-clip comparison
    compare_all_clips(clips)

    # Part 3: explanation
    print_loudness_explanation()
