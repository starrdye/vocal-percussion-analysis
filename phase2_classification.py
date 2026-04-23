"""
Phase 2: Supervised Beatbox Classification
===========================================
Three models evaluated with Leave-One-Participant-Out (LOPO) cross-validation:
  1. SVM (RBF kernel)         — 38-feature vector
  2. Random Forest            — 38-feature vector + feature importance
  3. CNN (Mel-Spectrogram)    — PyTorch, log-mel input with augmentation

Sound classes: b (bass kick), k (snare/click), nu (neutral hum), psh (hi-hat push)
"""

import os
import warnings
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
AUDIO_DIR        = "audio_data"
VALID_PARTS      = [1, 2, 3, 4, 5, 7, 8, 10, 11]
COMMON_SOUNDS    = {'b', 'k', 'nu', 'psh'}
SOUND_LABELS     = ['b', 'k', 'nu', 'psh']   # fixed label order throughout

SR               = 22050
N_FFT            = 2048
HOP_LENGTH       = 512
N_MFCC           = 13
N_MELS           = 64
MEL_FRAMES       = 128   # fixed time axis for CNN (≈3 s at hop=512, sr=22050)

CNN_EPOCHS       = 80
CNN_LR           = 1e-3
CNN_BATCH        = 16

# Phase 1 baseline numbers (from peak_alignment_clustering results)
PHASE1 = {'overall': 76.1, 'b': 38.1, 'k': 86.4, 'nu': 90.0, 'psh': 87.5}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(path):
    """Return a 38-dim feature vector, or None on failure.

    Features (all summarised as mean + std unless noted):
      MFCCs 1-13 mean + std          → 26
      Spectral Centroid mean + std   →  2
      Spectral Rolloff mean + std    →  2
      Spectral Flux mean + std       →  2
      Zero-Crossing Rate mean + std  →  2
      RMS Energy mean + std          →  2
      Attack Time (scalar)           →  1
      Decay Rate (scalar)            →  1
                                      ─────
                              Total    38
    """
    try:
        y, sr = librosa.load(path, sr=SR)
    except Exception as e:
        print(f"    [WARN] load error {path}: {e}")
        return None
    if len(y) < 256:
        return None

    # Adapt frame size to signal length (must be power-of-2, at least 256)
    frame = min(N_FFT, len(y))
    frame = max(256, 2 ** int(np.log2(frame)))
    hop   = min(HOP_LENGTH, frame // 4)

    # ── MFCCs ──────────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=frame, hop_length=hop)
    mfcc_mean = np.mean(mfcc, axis=1)   # (13,)
    mfcc_std  = np.std(mfcc,  axis=1)   # (13,)

    # ── Spectral Centroid ──────────────────────────────────────────────────
    cent = librosa.feature.spectral_centroid(y=y, sr=sr,
                                              n_fft=frame, hop_length=hop)[0]

    # ── Spectral Rolloff ───────────────────────────────────────────────────
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                             n_fft=frame, hop_length=hop)[0]

    # ── Spectral Flux ──────────────────────────────────────────────────────
    stft = np.abs(librosa.stft(y, n_fft=frame, hop_length=hop))
    diff = np.diff(stft, axis=1)
    flux = np.sqrt(np.sum(diff ** 2, axis=0)) if diff.size > 0 else np.array([0.])

    # ── Zero-Crossing Rate ─────────────────────────────────────────────────
    zcr  = librosa.feature.zero_crossing_rate(y, frame_length=frame,
                                               hop_length=hop)[0]

    # ── RMS Energy ─────────────────────────────────────────────────────────
    rms  = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]

    # ── Attack Time: onset → energy peak (seconds) ────────────────────────
    peak_idx    = int(np.argmax(rms))
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop)
    if len(onset_frames) > 0:
        onset       = int(onset_frames[0])
        attack_time = max(0.0, (peak_idx - onset) * hop / sr)
    else:
        attack_time = peak_idx * hop / sr

    # ── Decay Rate: linear slope of RMS after peak ────────────────────────
    post = rms[peak_idx:]
    if len(post) > 1:
        decay_rate = float(np.polyfit(np.arange(len(post)), post, 1)[0])
    else:
        decay_rate = 0.0

    return np.concatenate([
        mfcc_mean, mfcc_std,                          # 26
        [np.mean(cent), np.std(cent)],                #  2
        [np.mean(roll), np.std(roll)],                #  2
        [np.mean(flux), np.std(flux)],                #  2
        [np.mean(zcr),  np.std(zcr)],                 #  2
        [np.mean(rms),  np.std(rms)],                 #  2
        [attack_time],                                 #  1
        [decay_rate],                                  #  1
    ])  # total 38


def extract_mel(path):
    """Return (N_MELS, MEL_FRAMES) log-mel spectrogram, or None on failure."""
    try:
        y, sr = librosa.load(path, sr=SR)
    except Exception as e:
        print(f"    [WARN] load error {path}: {e}")
        return None
    if len(y) < N_FFT:
        return None

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)   # (N_MELS, T)

    # Pad or truncate time axis to MEL_FRAMES
    T = mel_db.shape[1]
    if T < MEL_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, MEL_FRAMES - T)), mode='constant',
                        constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :MEL_FRAMES]

    return mel_db  # (N_MELS, MEL_FRAMES)


def get_feature_names():
    names = [f'MFCC_{i+1}_mean' for i in range(N_MFCC)]
    names += [f'MFCC_{i+1}_std'  for i in range(N_MFCC)]
    names += ['Centroid_mean', 'Centroid_std',
              'Rolloff_mean',  'Rolloff_std',
              'Flux_mean',     'Flux_std',
              'ZCR_mean',      'ZCR_std',
              'RMS_mean',      'RMS_std',
              'Attack_time',   'Decay_rate']
    return names


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset():
    print("Loading dataset …")
    records = []
    for pid in VALID_PARTS:
        phase1 = os.path.join(AUDIO_DIR, str(pid), "Phase 1")
        if not os.path.exists(phase1):
            print(f"  [WARN] {phase1} not found — skipping")
            continue
        for fname in sorted(os.listdir(phase1)):
            if not fname.endswith(".wav"):
                continue
            parts = fname.split("-")
            if len(parts) < 2 or parts[1] not in COMMON_SOUNDS:
                continue
            sound = parts[1]
            fpath = os.path.join(phase1, fname)

            feats = extract_features(fpath)
            mel   = extract_mel(fpath)
            if feats is None or mel is None:
                print(f"  [WARN] skipping {fname}")
                continue

            records.append({
                'file':        fname,
                'participant': pid,
                'sound':       sound,
                'features':    feats,
                'mel':         mel,
            })

    n_per_sound = {s: sum(1 for r in records if r['sound'] == s)
                   for s in SOUND_LABELS}
    print(f"  Loaded {len(records)} samples — {n_per_sound}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LEAVE-ONE-PARTICIPANT-OUT CV
# ══════════════════════════════════════════════════════════════════════════════

def lopo_cv(records, model_fn, use_mel=False, label=""):
    """
    For each participant p:
      train on all other participants, test on p.
    Returns (y_true, y_pred, files, {participant: fold_accuracy}).
    """
    participants = sorted({r['participant'] for r in records})
    y_true_all, y_pred_all, files_all = [], [], []
    fold_accs = {}

    for fold_i, held_out in enumerate(participants, 1):
        train = [r for r in records if r['participant'] != held_out]
        test  = [r for r in records if r['participant'] == held_out]

        if use_mel:
            X_tr = np.array([r['mel']      for r in train])
            X_te = np.array([r['mel']      for r in test])
        else:
            X_tr = np.array([r['features'] for r in train])
            X_te = np.array([r['features'] for r in test])

        y_tr = [r['sound'] for r in train]
        y_te = [r['sound'] for r in test]

        print(f"  [{label}] Fold {fold_i}/{len(participants)}"
              f" — held-out P{held_out}"
              f" ({len(train)} train / {len(test)} test)")

        clf   = model_fn(X_tr, y_tr, fold=fold_i)
        preds = clf.predict(X_te)

        fold_accs[held_out] = accuracy_score(y_te, preds)
        y_true_all.extend(y_te)
        y_pred_all.extend(preds)
        files_all.extend(r['file'] for r in test)

    return y_true_all, y_pred_all, files_all, fold_accs


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a. SVM ───────────────────────────────────────────────────────────────────

def build_svm(X_train, y_train, fold=None):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    param_grid = {
        'C':     [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
    }
    n_cv = min(3, min(sum(1 for y in y_train if y == s)
                      for s in set(y_train)))
    n_cv = max(2, n_cv)
    grid = GridSearchCV(SVC(kernel='rbf', random_state=42),
                        param_grid, cv=n_cv, n_jobs=-1)
    grid.fit(X_scaled, y_train)

    class _SVM:
        def predict(self, X):
            return grid.best_estimator_.predict(scaler.transform(X))

    return _SVM()


# ── 4b. Random Forest ─────────────────────────────────────────────────────────

def build_rf(X_train, y_train, fold=None):
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


# ── 4c. CNN (PyTorch) ─────────────────────────────────────────────────────────

class _MelDataset(Dataset):
    def __init__(self, mels, labels, augment=False):
        self.mels    = mels      # (N, H, W) float32
        self.labels  = labels    # (N,) int64
        self.augment = augment

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        m = self.mels[idx].copy()

        if self.augment:
            # Time shift  ±12 frames
            shift = np.random.randint(-12, 13)
            m = np.roll(m, shift, axis=1)
            # Frequency mask
            fw = np.random.randint(0, 10)
            f0 = np.random.randint(0, max(1, N_MELS - fw))
            m[f0:f0 + fw, :] = m.min()
            # Time mask
            tw = np.random.randint(0, 16)
            t0 = np.random.randint(0, max(1, MEL_FRAMES - tw))
            m[:, t0:t0 + tw] = m.min()
            # Amplitude jitter
            m = m * np.random.uniform(0.85, 1.15)

        # Per-sample normalisation
        m = (m - m.mean()) / (m.std() + 1e-6)
        return torch.FloatTensor(m).unsqueeze(0), int(self.labels[idx])


class _BeatboxCNN(nn.Module):
    def __init__(self, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: (1, 64, 128) → (16, 32, 64)
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2: (16, 32, 64) → (32, 16, 32)
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3: (32, 16, 32) → (64, 4, 4)
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_cls),
        )

    def forward(self, x):
        return self.head(self.net(x))


def build_cnn(X_train, y_train, fold=None):
    le    = LabelEncoder().fit(SOUND_LABELS)
    y_int = le.transform(y_train).astype(np.int64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = _MelDataset(X_train.astype(np.float32), y_int, augment=True)
    dl = DataLoader(ds, batch_size=CNN_BATCH, shuffle=True, drop_last=False)

    model     = _BeatboxCNN(n_cls=4).to(device)
    opt       = optim.Adam(model.parameters(), lr=CNN_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CNN_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, CNN_EPOCHS + 1):
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(Xb), yb).backward()
            opt.step()
        scheduler.step()
        if ep % 20 == 0:
            print(f"      epoch {ep}/{CNN_EPOCHS}")

    class _CNNPredictor:
        def predict(self, X):
            model.eval()
            X = X.astype(np.float32)
            ds_te = _MelDataset(X, np.zeros(len(X), dtype=np.int64),
                                augment=False)
            dl_te = DataLoader(ds_te, batch_size=32)
            preds = []
            with torch.no_grad():
                for Xb, _ in dl_te:
                    out = model(Xb.to(device))
                    preds.extend(out.argmax(1).cpu().numpy())
            return le.inverse_transform(preds)

    return _CNNPredictor()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _per_sound_acc(y_true, y_pred):
    out = {}
    for s in SOUND_LABELS:
        idx = [i for i, t in enumerate(y_true) if t == s]
        if idx:
            out[s] = accuracy_score(
                [y_true[i] for i in idx],
                [y_pred[i] for i in idx]) * 100
        else:
            out[s] = float('nan')
    return out


def plot_confusion_matrix(y_true, y_pred, title, fname):
    cm  = confusion_matrix(y_true, y_pred, labels=SOUND_LABELS)
    acc = accuracy_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SOUND_LABELS, yticklabels=SOUND_LABELS,
                linewidths=.5, ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True',      fontsize=11)
    ax.set_title(f'{title} — LOPO-CV\nAccuracy: {acc*100:.1f}%', fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_feature_importance(importances, feature_names, fname):
    top = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.array(feature_names)[top], importances[top],
            color='steelblue', edgecolor='black', linewidth=.5)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Top 20 Random Forest Feature Importances\n(trained on full dataset)',
                 fontsize=12)
    ax.grid(axis='x', alpha=.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_lopo_per_participant(accs_dict, fname):
    """accs_dict: {model_name: {participant: accuracy}}"""
    parts  = sorted(next(iter(accs_dict.values())).keys())
    models = list(accs_dict.keys())
    x      = np.arange(len(parts))
    w      = 0.22
    colors = ['#4c72b0', '#dd8452', '#55a868']

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (m, c) in enumerate(zip(models, colors)):
        vals = [accs_dict[m][p] * 100 for p in parts]
        ax.bar(x + i * w, vals, w, label=m, color=c,
               edgecolor='black', linewidth=.5)

    ax.axhline(76.1, color='red', linestyle='--', linewidth=1.2,
               label='Phase 1 baseline (76.1%)')
    ax.set_xticks(x + w)
    ax.set_xticklabels([f'P{p}' for p in parts], fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('LOPO-CV: Per-Participant Accuracy by Model', fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_per_sound_comparison(results_dict, fname):
    """results_dict: {model_name: (y_true, y_pred)}"""
    models_order = ['Phase 1\n(K-Means)'] + list(results_dict.keys())

    # Build accuracy table
    pa = {
        'Phase 1\n(K-Means)': [PHASE1[s] for s in SOUND_LABELS]
    }
    for m, (yt, yp) in results_dict.items():
        ps = _per_sound_acc(yt, yp)
        pa[m] = [ps[s] for s in SOUND_LABELS]

    x      = np.arange(len(SOUND_LABELS))
    w      = 0.20
    colors = ['#aaaaaa', '#4c72b0', '#dd8452', '#55a868']

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (m, c) in enumerate(zip(models_order, colors)):
        ax.bar(x + i * w, pa[m], w, label=m, color=c,
               edgecolor='black', linewidth=.5)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(SOUND_LABELS, fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Per-Sound Accuracy: Phase 1 vs Phase 2', fontsize=12)
    ax.set_ylim(0, 112)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  PHASE 2: SUPERVISED BEATBOX CLASSIFICATION")
    print("=" * 65)
    print()

    records = load_dataset()
    print()

    # ── MODEL 1: SVM ──────────────────────────────────────────────────────
    print("─" * 65)
    print("MODEL 1 / 3 — SVM (RBF kernel, GridSearchCV)")
    print("─" * 65)
    yt_svm, yp_svm, files_svm, pacc_svm = lopo_cv(
        records, build_svm, use_mel=False, label="SVM")
    print(f"\n  Overall accuracy: {accuracy_score(yt_svm, yp_svm)*100:.1f}%")
    print(classification_report(yt_svm, yp_svm,
                                 target_names=SOUND_LABELS, digits=3))
    plot_confusion_matrix(yt_svm, yp_svm, 'SVM (RBF)',
                          'phase2_confusion_svm.png')
    print()

    # ── MODEL 2: Random Forest ────────────────────────────────────────────
    print("─" * 65)
    print("MODEL 2 / 3 — Random Forest (200 trees)")
    print("─" * 65)
    yt_rf, yp_rf, files_rf, pacc_rf = lopo_cv(
        records, build_rf, use_mel=False, label="RF")
    print(f"\n  Overall accuracy: {accuracy_score(yt_rf, yp_rf)*100:.1f}%")
    print(classification_report(yt_rf, yp_rf,
                                  target_names=SOUND_LABELS, digits=3))
    plot_confusion_matrix(yt_rf, yp_rf, 'Random Forest',
                          'phase2_confusion_rf.png')

    # Feature importance on full dataset (more stable than single fold)
    print("  Computing feature importances on full dataset …")
    X_all = np.array([r['features'] for r in records])
    y_all = [r['sound'] for r in records]
    rf_full = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf_full.fit(X_all, y_all)
    plot_feature_importance(rf_full.feature_importances_,
                            get_feature_names(),
                            'phase2_feature_importance.png')
    print()

    # ── MODEL 3: CNN ──────────────────────────────────────────────────────
    print("─" * 65)
    print("MODEL 3 / 3 — CNN on Mel-Spectrograms (PyTorch)")
    print("─" * 65)
    yt_cnn, yp_cnn, files_cnn, pacc_cnn = lopo_cv(
        records, build_cnn, use_mel=True, label="CNN")
    print(f"\n  Overall accuracy: {accuracy_score(yt_cnn, yp_cnn)*100:.1f}%")
    print(classification_report(yt_cnn, yp_cnn,
                                  target_names=SOUND_LABELS, digits=3))
    plot_confusion_matrix(yt_cnn, yp_cnn, 'CNN (Mel-Spectrogram)',
                          'phase2_confusion_cnn.png')
    print()

    # ── Comparison plots ──────────────────────────────────────────────────
    print("─" * 65)
    print("GENERATING COMPARISON PLOTS")
    print("─" * 65)
    plot_lopo_per_participant(
        {'SVM': pacc_svm, 'Random Forest': pacc_rf, 'CNN': pacc_cnn},
        'phase2_lopo_per_participant.png')
    plot_per_sound_comparison(
        {'SVM':           (yt_svm, yp_svm),
         'Random Forest': (yt_rf,  yp_rf),
         'CNN':           (yt_cnn, yp_cnn)},
        'phase2_per_sound_comparison.png')

    # ── Save CSV ──────────────────────────────────────────────────────────
    rf_lut  = {f: (t, p) for f, t, p in zip(files_rf,  yt_rf,  yp_rf)}
    cnn_lut = {f: (t, p) for f, t, p in zip(files_cnn, yt_cnn, yp_cnn)}
    rows = []
    for f, t, p in zip(files_svm, yt_svm, yp_svm):
        row = {'file': f, 'true': t,
               'svm_pred': p, 'svm_correct': t == p}
        if f in rf_lut:
            row['rf_pred']    = rf_lut[f][1]
            row['rf_correct'] = rf_lut[f][0] == rf_lut[f][1]
        if f in cnn_lut:
            row['cnn_pred']    = cnn_lut[f][1]
            row['cnn_correct'] = cnn_lut[f][0] == cnn_lut[f][1]
        rows.append(row)

    pd.DataFrame(rows).to_csv('phase2_results.csv', index=False)
    print("  Saved: phase2_results.csv")

    # ── Final summary table ───────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  PHASE 2 SUMMARY")
    print("=" * 65)
    hdr = f"{'Model':<30} {'Overall':>8}  " + \
          "  ".join(f"{s:>6}" for s in SOUND_LABELS)
    print(hdr)
    print("-" * 65)

    rows_summary = [
        ("Phase 1 — K-Means Overlap",
         PHASE1['overall'], {s: PHASE1[s] for s in SOUND_LABELS}),
        ("Phase 2 — SVM (RBF)",
         accuracy_score(yt_svm, yp_svm) * 100, _per_sound_acc(yt_svm, yp_svm)),
        ("Phase 2 — Random Forest",
         accuracy_score(yt_rf,  yp_rf)  * 100, _per_sound_acc(yt_rf,  yp_rf)),
        ("Phase 2 — CNN (Mel-Spec)",
         accuracy_score(yt_cnn, yp_cnn) * 100, _per_sound_acc(yt_cnn, yp_cnn)),
    ]
    for name, ov, ps in rows_summary:
        sound_cols = "  ".join(f"{ps[s]:>5.1f}%" for s in SOUND_LABELS)
        print(f"{name:<30} {ov:>7.1f}%  {sound_cols}")

    print("=" * 65)
    print()
    print("Output files generated:")
    for f in ['phase2_confusion_svm.png', 'phase2_confusion_rf.png',
              'phase2_confusion_cnn.png', 'phase2_feature_importance.png',
              'phase2_lopo_per_participant.png',
              'phase2_per_sound_comparison.png',
              'phase2_results.csv']:
        print(f"  {f}")


if __name__ == "__main__":
    main()
