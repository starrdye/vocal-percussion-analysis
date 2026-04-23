# Beatbox Classification Project

This project focuses on the classification and analysis of beatbox sounds.

## Project Structure

The codebase is organized into several phases of analysis, each represented by corresponding Python scripts and generated reports.

## 📝 Research Report

For a detailed analysis of the methodology, results, and discussion, please refer to the full [Research Report](report.md).

## 🚀 Key Findings

- **CNN Superiority**: The Convolutional Neural Network (CNN) on Mel-Spectrograms achieved an overall accuracy of **94.2%**, significantly outperforming baseline models.
- **Feature Significance**: MFCC features proved decisive in resolving acoustic ambiguities between bass kicks `{b}` and snare clicks `{k}`.
- **LOPO Generalization**: Leave-One-Participant-Out (LOPO) cross-validation confirmed that the model generalizes well across different individuals, despite variations in vocal production.

## Audio Data Structure

The `audio_data` directory is the central location for all audio files used in this project. It is structured to support participant recordings across different phases.

```text
audio_data/
├── 1/                     # Participant 1
│   ├── Phase 1/           # Individual sound clips (e.g., P1-b-01.wav)
│   └── Phase 2/           # Patterns and full recordings
├── 2/                     ... (same structure for participants 2-11)
└── ...
```

### Participant Data
- **Phase 1**: Contains individual sound clips organized by sound class (e.g., `b` for bass kick, `k` for snare).
- **Phase 2**: Contains patterns and full recordings used for supervised classification.

---

<div align="center">

## 📊 Project Findings and AI Layer Structure Poster

### Model Performance Findings

The Convolutional Neural Network (CNN) demonstrated a significant accuracy improvement over traditional machine learning and unsupervised clustering.

```mermaid
xychart-beta
    title "Overall Classification Accuracy by Model (%)"
    x-axis ["Phase 1 K-Means", "Phase 2 SVM", "Phase 2 Random Forest", "Phase 2 CNN"]
    y-axis "Accuracy %" 0 --> 100
    bar [76.1, 82.6, 83.7, 94.2]
```

<br>

<p align="center">
  <img src="public/poster.png" width="95%" title="Beatbox Classification CNN Findings Poster">
</p>

### CNN Architecture Structure

The best-performing model extracts Mel-Spectrogram features through three connected spatial convolution blocks.

```mermaid
graph TD
    Input["Input: Log Mel-Spectrogram (1, 64, 128)"] --> Block1
    
    subgraph Block1["Conv2D Block 1"]
        C1["Conv2d (1 to 16, 3x3)"] --> B1["BatchNorm2d"]
        B1 --> R1["ReLU"]
        R1 --> P1["MaxPool2d (2x2)"]
    end
    
    Block1 --> Block2
    
    subgraph Block2["Conv2D Block 2"]
        C2["Conv2d (16 to 32, 3x3)"] --> B2["BatchNorm2d"]
        B2 --> R2["ReLU"]
        R2 --> P2["MaxPool2d (2x2)"]
    end
    
    Block2 --> Block3
    
    subgraph Block3["Conv2D Block 3"]
        C3["Conv2d (32 to 64, 3x3)"] --> B3["BatchNorm2d"]
        B3 --> R3["ReLU"]
        R3 --> P3["AdaptiveAvgPool2d (4x4)"]
    end
    
    Block3 --> Head
    
    subgraph Head["Classifier Head"]
        F1["Flatten"] --> L1["Linear (1024 to 128)"]
        L1 --> RH1["ReLU"]
        RH1 --> D1["Dropout (0.5)"]
        D1 --> L2["Linear (128 to 4)"]
    end
    
    Head --> Output["Output: Softmax (4 Classes)"]
```

</div>

---

## Getting Started

1. **Audio Data**: Ensure the `audio_data/` directory is populated with the necessary recordings.
2. **Classification**: Run `phase2_classification.py` to perform supervised classification using SVM, Random Forest, and CNN models.
3. **Analysis**: Use `rms_analysis.py` for further signal processing and comparative studies.

---
