# TinyBioNet: A lightweight time-frequency network for biomedical signal classification on edge devices

This repository provides a pipeline for training a quantized deep learning model on physiological signals (PPG, ACC, ECG, EDA, EMG) using **STFT preprocessing** and **sliding window segmentation**. The model is designed to support **multiple datasets**, **cross-validation**, **weight quantization**, and **TFLite INT8 deployment**.

---

## Features

- **Multiple datasets supported**:
  - AffectiveROAD (`low`, `medium`, `high`)
  - PPG_ACC (`rest`, `squat`, `step`)
  - WEASAD (`baseline`, `stress`, `amusement`, `meditation`)
- **Flexible input selection**:
  - PPG, ACC, ECG, EDA, EMG, or combined signals
- **Sliding window segmentation** with overlap
  - Automatically handles averaging or majority vote for target labels
- **STFT preprocessing** to transform time-domain signals into time-frequency representations
- **Weight quantization** with custom bitwidth (default 8-bit)
- **Fine-tuning quantized models**
- **Full INT8 TFLite conversion** with representative dataset
- **Stratified K-Fold Cross-Validation** for robust evaluation
- **Metrics**:
  - Accuracy
  - Weighted F1-score
  - Detailed classification report

---

## Installation

```bash
pip install tensorflow tensorflow_model_optimization keras_tuner pandas scipy matplotlib scikit-learn
