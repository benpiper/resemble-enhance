import io
import streamlit as st
import torch
import torchaudio
import scipy.io.wavfile
import numpy as np



from resemble_enhance.enhancer.inference import denoise, enhance

# Must be the first Streamlit command
st.set_page_config(page_title="Resemble Enhance", layout="wide")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def process_audio(dwav, sr, start_time, end_time, solver, nfe, tau, lambd):
    # 1. Trim Audio
    # Convert seconds to frames
    start_frame = int(start_time * sr)
    end_frame = int(end_time * sr) if end_time > 0 else dwav.shape[-1]
    
    if dwav.ndim == 1:
        dwav = dwav[start_frame:end_frame]
    else: # (channels, frames)
        dwav = dwav[:, start_frame:end_frame]
            
    # Downmix to mono if stereo
    if dwav.ndim > 1:
        dwav = dwav.mean(dim=0)
    
    # Int tensor conversion to float
    if dwav.dtype == torch.int16:
        dwav = dwav.float() / 32768.0
    elif dwav.dtype == torch.int32:
        dwav = dwav.float() / 2147483648.0
    else:
        dwav = dwav.float()

    if dwav.numel() == 0:
        return None, None
        
    # 2. Process
    solver = solver.lower()
    nfe = int(nfe)
    
    with st.spinner("Denoising audio..."):
        wav1, new_sr = denoise(dwav, sr, device)
        
    with st.spinner("Enhancing audio (this may take a moment)..."):
        wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)
        
    return (new_sr, wav1.cpu().numpy()), (new_sr, wav2.cpu().numpy())

def create_wav_buffer(wav_np, sr):
    buffer = io.BytesIO()
    if wav_np.ndim > 1:
        wav_np = wav_np.T
    scipy.io.wavfile.write(buffer, sr, wav_np)
    buffer.seek(0)
    return buffer

def main():
    st.title("Resemble Enhance")
    st.markdown("AI-driven audio enhancement for your audio files, powered by Resemble AI.")

    # Sidebar settings
    with st.sidebar:
        st.header("Enhancement Settings")
        
        solver = st.selectbox(
            "CFM ODE Solver",
            options=["Midpoint", "RK4", "Euler"],
            index=0,
            help="Selects the numerical solver for the CFM model."
        )
        
        nfe = st.slider(
            "CFM Number of Function Evaluations",
            min_value=1, max_value=128, value=64, step=1,
            help="Determines the number of steps the solver takes. Higher = better quality but slower."
        )
        
        tau = st.slider(
            "CFM Prior Temperature",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="Controls variance/randomness. Lower = more stable and deterministic."
        )
        
        lambd = st.slider(
            "Denoising Strength",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="Does not affect Output Denoised Audio directly."
        )

    # Main layout
    st.subheader("Input Audio")
    
    tab1, tab2 = st.tabs(["Upload File", "Record Microphone"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "ogg"])
        
    with tab2:
        recorded_file = st.audio_input("Record an audio clip")
        
    audio_source = uploaded_file or recorded_file
    
    if audio_source is not None:
        try:
            # We must parse it to determine duration and pass to the plugin
            file_bytes = audio_source.read()
            buffer = io.BytesIO(file_bytes)
            dwav, sr = torchaudio.load(buffer, backend="soundfile")
            duration = dwav.shape[-1] / sr
            
            # The Advanced Audio component handles the UI and Trimming interactions
            st.markdown("**Trim Audio:** Drag the red region on the waveform below.")
            
            # We render the audio component using Streamlit's HTML component with an embedded wavesurfer instance
            # Since streamlit-advanced-audio sometimes has issues with base64 size limits, we'll use Streamlit's native st.audio
            # with custom session state bounds if we want a pure Python approach, but the user requested an integrated waveform trimmer.
            # Thus we use the newly added st.audio_input component from Streamlit 1.39!
            
            # Wait, st.audio_input is for RECORDING. For UPLOADED files, we need a custom html wrapper or session state.
            
            # Since the user specifically requested an integrated "Visual Audio Trimmer Component" 
            # and we are on Streamlit, we will use the `streamlit-advanced-audio` package (imported as `audix` usually, but we need to check exactly how it exposes itself).
            
            # Let's import the component dynamically
            from streamlit_advanced_audio import audix
            
            # Display the advanced audio player which includes regions
            audio_result = audix(
                data=file_bytes, 
                key="source_audio_player",
            )
            
            # Extract trim bounds
            start_time = 0.0
            end_time = duration
            
            if audio_result and 'selectedRegion' in audio_result and audio_result['selectedRegion']:
                # The plugin returns dictionary containing selectedRegion if the user created one
                region = audio_result['selectedRegion']
                start_time = region.get('start', 0.0)
                end_time = region.get('end', duration)
                st.info(f"Selected Region: {start_time:.2f}s to {end_time:.2f}s")
            else:
                st.info(f"Full Track length: {duration:.2f}s")

            if st.button("Enhance Audio", type="primary", use_container_width=True):
                denoised_result, enhanced_result = process_audio(
                    dwav, sr, start_time, end_time, solver, nfe, tau, lambd
                )
                
                if denoised_result and enhanced_result:
                    st.divider()
                    st.subheader("Outputs")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Output Denoised Audio**")
                        denoised_sr, denoised_wav = denoised_result
                        denoised_buffer = create_wav_buffer(denoised_wav, denoised_sr)
                        st.audio(denoised_buffer, format="audio/wav")
                        
                    with col2:
                        st.markdown("**Output Enhanced Audio**")
                        enhanced_sr, enhanced_wav = enhanced_result
                        enhanced_buffer = create_wav_buffer(enhanced_wav, enhanced_sr)
                        st.audio(enhanced_buffer, format="audio/wav")

        except Exception as e:
            st.error(f"Error loading audio: {e}")

if __name__ == "__main__":
    main()
