# AGENTS.md - Developer & Agent Guide

Welcome to the **Resemble Enhance** codebase. This document is designed to help software engineers and AI agents understand the structure, modules, and workflows of this project.

## Project Overview

Resemble Enhance is an AI-powered audio refinement tool that provides two main capabilities:
1.  **Denoising**: Separating speech from background noise.
2.  **Enhancement**: Boosting perceptual quality by restoring distortions and extending bandwidth (up to 44.1kHz).

## Directory Structure

```text
resemble-enhance/
├── app.py                  # Gradio-based Web UI
├── config/                 # Training configurations (YAML)
├── resemblance_enhance/    # Main package
│   ├── denoiser/           # Denoising model and training
│   ├── enhancer/           # Enhancement model (CFM + Vocoder)
│   ├── utils/              # Training and logging utilities
│   ├── inference.py        # High-level inference (chunking/merging)
│   ├── hparams.py          # Centralized hyperparameters
│   └── melspec.py          # Spectrogram utilities
├── setup.py                # Package installation
└── requirements.txt        # Dependencies
```

## Key Modules

### `resemble_enhance.denoiser`
- **Goal**: Implement a robust speech denoiser.
- **Key Files**:
    - `denoiser.py`: Contains the model architecture, often using Multi-Period Discriminators (MPD) and Multi-Scale Discriminators (MSD).
    - `unet.py`: The core UNet architecture used for feature extraction.
    - `train.py`: Script for "warmup" training of the denoiser.

### `resemble_enhance.enhancer`
- **Goal**: High-fidelity audio restoration.
- **Key Files**:
    - `enhancer.py`: Defines the enhancer model, which typically involves an autoencoder and a vocoder.
    - `lcfm/`: Implementation of Latent Conditional Flow Matching.
    - `univnet/`: Implementation of the UnivNet vocoder for high-quality audio synthesis.
    - `train.py`: Two-stage training script (Stage 1: Autoencoder/Vocoder, Stage 2: CFM).

### `resemble_enhance.inference`
- **Goal**: Orchestrate the inference process.
- **Features**: Handles long audio files by splitting them into chunks, processing them individually, and merging them with cross-fading to avoid artifacts.

## Development Workflows

### Installation
```bash
pip install -e .
```

### Running the Web Demo
```bash
python app.py
```

### Training
1.  **Warmup Denoiser**: `python -m resemble_enhance.denoiser.train --yaml config/denoiser.yaml runs/denoiser`
2.  **Enhancer Stage 1**: `python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage1.yaml runs/enhancer_stage1`
3.  **Enhancer Stage 2**: `python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage2.yaml runs/enhancer_stage2`

## Guidelines for Agents

- **Hyperparameters**: Most architectural and training settings are in `resemble_enhance/hparams.py`.
- **Devices**: Use `torch.cuda.is_available()` to determine the processing device.
- **Audio Loading**: Prefer `torchaudio` for loading and resampling.
- **Inference**: Always use the top-level `inference` function in `resemble_enhance/inference.py` for general use, as it handles audio chunking correctly.
