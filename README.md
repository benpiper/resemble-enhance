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

We provide a web demo built with Plotly Dash. You can run it locally:

```
python app.py
```

By default, the script listens on `0.0.0.0:7861`.

#### Dash Settings

The web demo provides several advanced settings that directly impact the output quality and speech accuracy:

- **Trim Start / End Time:** You can manually crop the input audio by specifying a start and end time (in seconds). This is useful for removing long silences before processing. Leaving End Time at `0` processes until the end of the file.
- **CFM ODE Solver** (`Midpoint`, `RK4`, `Euler`): Selects the numerical solver for the Conditional Flow Matching (CFM) model. `Midpoint` is the default and provides a good balance between speed and quality. `RK4` is more computationally expensive but can yield highly accurate and detailed speech. `Euler` is the fastest but may result in lower quality and more audio artifacts.
- **CFM Number of Function Evaluations (NFE)** (Range: 1-128, Default: 64): Determines the number of steps the solver takes. A higher NFE generally results in better speech accuracy, fewer artifacts, and more natural-sounding audio, though it will take longer to process. Lowering this value speeds up generation but can noticeably degrade clarity.
- **CFM Prior Temperature (Tau)** (Range: 0-1, Default: 0.5): Controls the amount of variance or randomness applied during the enhancement process. A lower value (e.g., closer to 0) makes the output more deterministic and stable, which helps in preserving exact speech patterns and minimizing unwanted artifacts. A higher value introduces more high-frequency details and variability, which can sometimes make the voice sound more natural but risks introducing artifacts or slight alterations in speech accuracy.
- **Denoising Strength** (Range: 0-1, Default: 0.5): Specifies the interpolation balance between the original audio signal and the enhanced output. A lower value keeps more of the source signal mixed in, while a higher value leans more heavily on the model's generated enhancements. If your original audio has artifacts you want completely removed, increase this value.
    - *Note:* Setting Denoising Strength to `0` does **not** bypass the model entirely. While `0` disables the pre-enhancement UNet denoiser, your audio is still passed through the Latent Conditional Flow Matching (LCFM) model and neural Vocoder. Thus, the "Enhanced Audio" output will still sound materially different (and generative) compared to the original, noisy audio or the standalone "Denoised Audio" output.

#### Handling Low SNR Audio

If your source audio has a very low Signal-to-Noise Ratio (SNR)—for example, the speech is very quiet compared to loud background traffic or wind noise—the enhancement model may struggle to isolate and reconstruct the voice accurately, potentially hallucinating speech or generating artifacts.

For best results with low SNR audio:
1. **Pre-clean the audio:** It is highly recommended to run the audio through a dedicated noise reduction utility first (such as a traditional spectral noise gate or a specialized dialogue isolator) to remove large, continuous bands of background noise before passing it to Resemble Enhance.
2. **Normalize and Compress:** If the speech volume is simply very low, normalizing the audio to a standard level (e.g., -3 dBFS) and applying gentle dynamic range compression can help the model "hear" the speech better without over-amplifying the noise floor. 
3. **Use the Denoiser First:** You can use the `resemble-enhance --denoise_only` command to run the first-stage denoiser on its own, evaluate the standalone cleanup, and then run the enhancer on that output if satisfied.

**Pre-cleaning vs. Built-in Denoiser:**
Which approach is better depends heavily on the specific noise profile of your audio:
* **Use Pre-cleaning (Dedicated external utilities)** for extreme continuous noise (e.g., severe traffic, heavy tape hiss) where the speech is barely audible. Pre-cleaning prevents the generative enhancer model from hallucinating speech sounds out of heavy noise.
* **Use the Built-in Denoiser** for moderate or highly variable noise (e.g., a typical imperfect room recording, keyboard clicks, background chatter). The built-in denoiser was explicitly trained alongside the enhancer, so they have a native synergy that prevents weird "underwater" phase artifacts often introduced by aggressive third-party spectral subtraction tools.

#### Troubleshooting

**What to do if the enhanced audio sounds like gibberish, even with optimal settings?**
The Latent Conditional Flow Matching (LCFM) model relies heavily on the structure of human speech. If the model outputs metallic, scrambled, or "gibberish" audio despite using ideal slider settings (e.g., Denoising Strength = 0.5, Tau = 0.5), it is usually due to the input signal confusing the generative process:

1. **Overlapping Speech or Background Voices:** Resemble Enhance is designed to enhance a single primary speaker. If multiple people are talking at once, or if there is distinct background chatter, the model may attempt to "merge" the voices or enhance a background conversation, resulting in scrambled, gibberish output.
    * **How to resolve:** Use a dedicated stem-separation or dialogue isolation tool to isolate the primary speaker's voice before processing it with Resemble Enhance.
2. **Extreme Clipping (Distortion):** If the input audio is severely clipped (peaking above 0 dBFS, which creates a flattened square wave), the generative model will struggle to interpret the distorted formants and will output strange artifacts.
    * **How to resolve:** Apply a dedicated de-clipper utility to restore the waveform peaks *before* running the audio through Resemble Enhance.
3. **Non-Speech Audio:** Resemble Enhance is strictly a *speech* enhancer. If you feed it music, singing, or isolated environmental noise (like a running faucet) without discernible speech, the model will still try to aggressively force the audio into a speech-like structure, leading to bizarre sounds.
    * **How to resolve:** Ensure you are only feeding spoken word/dialogue into the model. If a clip contains sections of pure music or noise, cut or mute those sections beforehand.
4. **Severely Degraded Intelligibility:** If the source audio is so degraded (e.g., 8kHz telephone audio with heavy compression) that you can barely understand the words yourself, the model lacks the structural data it needs as a guide and will frequently guess incorrectly (hallucination).
    * **How to resolve:** Lower the **Denoising Strength** slider (e.g. to `0.1` or `0.2`) to mask the hallucinations by blending the original signal back in. Pre-cleaning with external EQ or noise reduction before enhancement is also highly recommended.

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
