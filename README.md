# Resemble Enhance

[![PyPI](https://img.shields.io/pypi/v/resemble-enhance.svg)](https://pypi.org/project/resemble-enhance/)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face%20%F0%9F%A4%97-Space-yellow)](https://huggingface.co/spaces/ResembleAI/resemble-enhance)
[![License](https://img.shields.io/github/license/resemble-ai/Resemble-Enhance.svg)](https://github.com/resemble-ai/resemble-enhance/blob/main/LICENSE)
[![Webpage](https://img.shields.io/badge/Webpage-Online-brightgreen)](https://www.resemble.ai/enhance/)

https://github.com/resemble-ai/resemble-enhance/assets/660224/bc3ec943-e795-4646-b119-cce327c810f1

Resemble Enhance is an AI-powered tool that aims to improve the overall quality of speech by performing denoising and enhancement. It consists of two modules: a denoiser, which separates speech from a noisy audio, and an enhancer, which further boosts the perceptual audio quality by restoring audio distortions and extending the audio bandwidth. The two models are trained on high-quality 44.1kHz speech data that guarantees the enhancement of your speech with high quality.

## Usage

### Installation

Install the stable version:

```bash
pip install resemble-enhance --upgrade
```

Or try the latest pre-release version:

```bash
pip install resemble-enhance --upgrade --pre
```

### Enhance

```
resemble-enhance in_dir out_dir
```

### Denoise only

```
resemble-enhance in_dir out_dir --denoise_only
```

### Web Demo

We provide a web demo built with Gradio, you can try it out [here](https://huggingface.co/spaces/ResembleAI/resemble-enhance), or also run it locally:

```
python app.py
```

By default, the script listens on `127.0.0.1`. To make it accessible across your local network, you can modify `app.py` to use `interface.launch(server_name="0.0.0.0")`.

#### Gradio Settings

The web demo provides several advanced settings that directly impact the output quality and speech accuracy:

- **CFM ODE Solver** (`Midpoint`, `RK4`, `Euler`): Selects the numerical solver for the Conditional Flow Matching (CFM) model. `Midpoint` is the default and provides a good balance between speed and quality. `RK4` is more computationally expensive but can yield highly accurate and detailed speech. `Euler` is the fastest but may result in lower quality and more audio artifacts.
- **CFM Number of Function Evaluations (NFE)** (Range: 1-128, Default: 64): Determines the number of steps the solver takes. A higher NFE generally results in better speech accuracy, fewer artifacts, and more natural-sounding audio, though it will take longer to process. Lowering this value speeds up generation but can noticeably degrade clarity.
- **CFM Prior Temperature (Tau)** (Range: 0-1, Default: 0.5): Controls the amount of variance or randomness applied during the enhancement process. A lower value (e.g., closer to 0) makes the output more deterministic and stable, which helps in preserving exact speech patterns and minimizing unwanted artifacts. A higher value introduces more high-frequency details and variability, which can sometimes make the voice sound more natural but risks introducing artifacts or slight alterations in speech accuracy.
- **CFM Denoise Strength / Guidance (Lambda)** (Range: 0-1, Default: 0.1): Controls how much freedom the enhancement model has to generate new features. If your enhanced speech sounds "foreign" or "nonsense", **turn this slider down to 0**. A lower value forces the AI to closely follow the original audio, preserving reality and preventing hallucinated speech sounds on low-volume sources. A higher value (e.g., 0.9) allows the AI to completely replace heavy noise with clean speech, but risks altering the speaker's identity or clarity.

## Train your own model

### Data Preparation

You need to prepare a foreground speech dataset and a background non-speech dataset. In addition, you need to prepare a RIR dataset ([examples](https://github.com/RoyJames/room-impulse-responses)).

```bash
data
├── fg
│   ├── 00001.wav
│   └── ...
├── bg
│   ├── 00001.wav
│   └── ...
└── rir
    ├── 00001.npy
    └── ...
```

### Training

#### Denoiser Warmup

Though the denoiser is trained jointly with the enhancer, it is recommended for a warmup training first.

```bash
python -m resemble_enhance.denoiser.train --yaml config/denoiser.yaml runs/denoiser
```

#### Enhancer

Then, you can train the enhancer in two stages. The first stage is to train the autoencoder and vocoder. And the second stage is to train the latent conditional flow matching (CFM) model.

##### Stage 1

```bash
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage1.yaml runs/enhancer_stage1
```

##### Stage 2

```bash
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage2.yaml runs/enhancer_stage2
```

## Blog

Learn more on our [website](https://www.resemble.ai/enhance/)!
