from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import soundfile as sf
import librosa
import joblib
import tempfile
import torch
import numpy as np
import torchaudio
from CNN import SpectrogramCNN  


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up FastAPI app")
    app.state.model = SpectrogramCNN()
    app.state.model.load_state_dict = torch.load("mejor_modelo_eer_LA_ES_REG_AUG_ANALISIS.pth", map_location="cpu")
    app.state.model.eval()
    yield
    
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "https://audioclassfrontendpi2.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_audio_to_mel(audio_path, sample_rate=16000, duration=5, n_mels=65, n_fft=512, hop_length=200):
    y, sr = librosa.load(audio_path, sr=sample_rate)
    waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    expected_len = int(sample_rate * duration)
    if waveform.shape[1] > expected_len:
        waveform = waveform[:, :expected_len]
    else:
        padding = expected_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    waveform = waveform / waveform.abs().max()
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    mel = mel_transform(waveform)
    mel_db = db_transform(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    return mel_db.unsqueeze(0)  # shape: (1, n_mels, time)

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    try:
        print("Received request in /predict")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as temp_file:
            temp_file.write(await audio.read())
            temp_file_path = temp_file.name

        
        mel_db = preprocess_audio_to_mel(temp_file_path)  # shape: (1, n_mels, time)
        print("mel_db shape after preprocessing:", mel_db.shape)
        # Add batch dimension for CNN: (batch, channel, n_mels, time)
        input_tensor = mel_db  # shape: (1, 1, n_mels, time)
        print("input_tensor shape for model:", input_tensor.shape)
        
        with torch.no_grad():
            output = app.state.model(input_tensor).squeeze()
        print("Model output:", output) 
        
        if output < -1.71:
            result = "Artificial"
        else:
            result = "Natural"
        
        print("Prediction result:", result)
        return {"prediction": result}
    except Exception as e:
        print("Exception in /predict", e)
        raise


@app.get("/ping")
async def ping():
    print("Ping received")
    return {"message": "pong"}
