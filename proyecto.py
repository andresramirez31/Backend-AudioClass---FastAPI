from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import soundfile as sf
import librosa
import joblib
import tempfile

appMain = FastAPI()

@appMain.get("/")
def read_root():
    return {"message": "Welcome to the Audio Processing API"}