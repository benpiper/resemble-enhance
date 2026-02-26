import gradio as gr
import torch
import torchaudio

from resemble_enhance.enhancer.inference import denoise, enhance

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def _fn(audio, start_seq, end_seq, solver, nfe, tau, lambd):
    if audio is None:
        return None, None

    sr, y = audio

    # Calculate actual start and end frames
    start_frame = int(start_seq * sr)
    
    if y.ndim == 1:
        if end_seq > 0:
            end_frame = int(end_seq * sr)
            y = y[start_frame:end_frame]
        else:
            y = y[start_frame:]
        y = y[None, :]
    else:
        if end_seq > 0:
            end_frame = int(end_seq * sr)
            y = y[start_frame:end_frame, :]
        else:
            y = y[start_frame:, :]
        y = y.T

    if y.size == 0 or y.shape[-1] == 0:
        return None, None
        
    dwav = torch.from_numpy(y)

    # Convert commonly returned int formats to standard float32 [-1.0, 1.0]
    if dwav.dtype == torch.int16:
        dwav = dwav.float() / 32768.0
    elif dwav.dtype == torch.int32:
        dwav = dwav.float() / 2147483648.0
    else:
        dwav = dwav.float()
        
    solver = solver.lower()
    nfe = int(nfe)

    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    wav1 = wav1.cpu().numpy()
    wav2 = wav2.cpu().numpy()

    return (new_sr, wav1), (new_sr, wav2)


def main():
    inputs: list = [
        gr.Audio(type="numpy", label="Input Audio", editable=False),
        gr.Number(value=0.0, label="Trim Start Time (seconds)"),
        gr.Number(value=0.0, label="Trim End Time (seconds, 0 = end)"),
        gr.Dropdown(choices=["Midpoint", "RK4", "Euler"], value="Midpoint", label="CFM ODE Solver"),
        gr.Slider(minimum=1, maximum=128, value=64, step=1, label="CFM Number of Function Evaluations"),
        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="CFM Prior Temperature"),
        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="Denoising Strength (Does not affect Output Denoised Audio)"),
    ]

    outputs: list = [
        gr.Audio(label="Output Denoised Audio"),
        gr.Audio(label="Output Enhanced Audio"),
    ]

    interface = gr.Interface(
        fn=_fn,
        title="Resemble Enhance",
        description="AI-driven audio enhancement for your audio files, powered by Resemble AI.",
        inputs=inputs,
        outputs=outputs,
    )

    interface.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
