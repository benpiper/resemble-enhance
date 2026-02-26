import base64
import io
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import torch
import torchaudio
import scipy.io.wavfile
import numpy as np
import plotly.graph_objects as go

from resemble_enhance.enhancer.inference import denoise, enhance

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Resemble Enhance"

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Resemble Enhance", className="text-center mt-4 mb-2"))),
    dbc.Row(dbc.Col(html.P("AI-driven audio enhancement for your audio files, powered by Resemble AI.", className="text-center text-muted mb-4"))),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Input Audio & Settings"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-audio',
                        children=html.Div(['Drag and Drop or ', html.A('Select an Audio File')]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '15px'
                        }
                    ),
                    html.Div(id='upload-status', className="text-success mb-3", style={'fontWeight': 'bold'}),
                    
                    html.Div(id='waveform-container', style={'display': 'none'}, children=[
                        html.Label("Trim Audio:"),
                        dcc.Graph(id='waveform-graph', config={'displayModeBar': False}, style={'height': '150px'}),
                        dcc.RangeSlider(
                            id='trim-slider',
                            min=0, max=100, step=0.01,
                            value=[0, 100],
                            tooltip={"placement": "bottom", "always_visible": True},
                            marks=None,
                            className="mb-4"
                        )
                    ]),
                    
                    html.Label("CFM ODE Solver"),
                    dcc.Dropdown(
                        id='solver-dropdown',
                        options=[
                            {'label': 'Midpoint', 'value': 'Midpoint'},
                            {'label': 'RK4', 'value': 'RK4'},
                            {'label': 'Euler', 'value': 'Euler'}
                        ],
                        value='Midpoint',
                        className="mb-3"
                    ),
                    
                    html.Label("CFM Number of Function Evaluations"),
                    dcc.Slider(id='nfe-slider', min=1, max=128, step=1, value=64,
                               tooltip={"placement": "bottom", "always_visible": True}, className="mb-3"),
                    
                    html.Label("CFM Prior Temperature"),
                    dcc.Slider(id='tau-slider', min=0, max=1, step=0.01, value=0.5,
                               marks={0: '0', 0.5: '0.5', 1: '1'},
                               tooltip={"placement": "bottom", "always_visible": True}, className="mb-3"),
                               
                    html.Label("Denoising Strength (Does not affect Output Denoised Audio)"),
                    dcc.Slider(id='lambd-slider', min=0, max=1, step=0.01, value=0.5,
                               marks={0: '0', 0.5: '0.5', 1: '1'},
                               tooltip={"placement": "bottom", "always_visible": True}, className="mb-4"),
                               
                    dbc.Button("Enhance Audio", id='submit-button', color="primary", className="w-100", n_clicks=0),
                ])
            ], className="mb-4")
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Outputs"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-outputs",
                        type="circle",
                        children=[
                            html.H5("Output Denoised Audio", className="mt-2"),
                            html.Audio(id='output-denoised', controls=True, style={'width': '100%'}, className="mb-4"),
                            
                            html.H5("Output Enhanced Audio"),
                            html.Audio(id='output-enhanced', controls=True, style={'width': '100%'})
                        ]
                    )
                ])
            ])
        ], md=6)
    ])
], fluid=True, style={'maxWidth': '1200px'})

def parse_audio(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Load audio from memory buffer
    buffer = io.BytesIO(decoded)
    dwav, sr = torchaudio.load(buffer)
    return dwav, sr

def create_data_uri(wav_tensor, sr):
    buffer = io.BytesIO()
    # Scipy expects (frames, channels) and a numpy array
    wav_np = wav_tensor.cpu().numpy()
    if wav_np.ndim == 1:
        pass # already 1D
    else:
        wav_np = wav_np.T # usually torchaudio gives (channels, frames) so transpose back if needed but enhance gives 1D
        
    scipy.io.wavfile.write(buffer, sr, wav_np)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('ascii')
    return f"data:audio/wav;base64,{encoded}"

@app.callback(
    [Output('upload-status', 'children'),
     Output('waveform-container', 'style'),
     Output('waveform-graph', 'figure'),
     Output('trim-slider', 'min'),
     Output('trim-slider', 'max'),
     Output('trim-slider', 'value')],
    Input('upload-audio', 'contents'),
    State('upload-audio', 'filename')
)
def update_waveform(contents, filename):
    if not contents:
        return "", {'display': 'none'}, dash.no_update, 0, 100, [0, 100]
        
    try:
        dwav, sr = parse_audio(contents)
        
        # Determine total duration
        duration = dwav.shape[-1] / sr
        
        # Prepare waveform for fast plotting
        # Downmix to mono and convert to numpy
        if dwav.ndim > 1:
            y = dwav.mean(dim=0).cpu().numpy()
        else:
            y = dwav.cpu().numpy()
            
        # Target 2000 points max for plotting speed
        n_points = 2000
        if len(y) > n_points:
            # Envelope block max pooling
            chunk_size = len(y) // n_points
            y_pad = np.pad(y, (0, chunk_size - len(y) % chunk_size), mode='constant')
            y_pool = y_pad.reshape(-1, chunk_size).max(axis=1)
        else:
            y_pool = y
            
        time_axis = np.linspace(0, duration, len(y_pool))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_axis, y=y_pool, 
            mode='lines', 
            fill='tozeroy',
            line=dict(color='#007bff', width=1),
            fillcolor='rgba(0,123,255,0.2)'
        ))
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title=None,
            yaxis_title=None,
            xaxis=dict(showgrid=False, zeroline=False, fixedrange=True),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        return f"Loaded: {filename} ({duration:.1f}s)", {'display': 'block'}, fig, 0.0, duration, [0.0, duration]
    
    except Exception as e:
        print(f"Error drawing waveform: {e}")
        return f"Error loading {filename}", {'display': 'none'}, dash.no_update, 0, 100, [0, 100]


@app.callback(
    [Output('output-denoised', 'src'),
     Output('output-enhanced', 'src')],
    [Input('submit-button', 'n_clicks')],
    [State('upload-audio', 'contents'),
     State('trim-slider', 'value'),
     State('solver-dropdown', 'value'),
     State('nfe-slider', 'value'),
     State('tau-slider', 'value'),
     State('lambd-slider', 'value')]
)
def process_audio(n_clicks, audio_contents, trim_values, solver, nfe, tau, lambd):
    # Only run if button was clicked and file is uploaded
    if n_clicks == 0 or audio_contents is None:
        return dash.no_update, dash.no_update
        
    try:
        # 1. Parse base64 audio
        dwav, sr = parse_audio(audio_contents)
        
        # 2. Trim audio
        start_seq, end_seq = trim_values[0], trim_values[1]
        start_frame = int(start_seq * sr)
        end_frame = int(end_seq * sr)
        
        if dwav.ndim == 1:
            dwav = dwav[start_frame:end_frame]
        else: # (channels, frames)
            dwav = dwav[:, start_frame:end_frame]
                
        # Downmix to mono if stereo
        dwav = dwav.mean(dim=0)
        
        if dwav.numel() == 0:
            return dash.no_update, dash.no_update

        # 3. Process
        solver = solver.lower()
        nfe = int(nfe)
        
        wav1, new_sr = denoise(dwav, sr, device)
        wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)
        
        # 4. Encode outputs
        src_denoised = create_data_uri(wav1, new_sr)
        src_enhanced = create_data_uri(wav2, new_sr)
        
        return src_denoised, src_enhanced
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7861)
