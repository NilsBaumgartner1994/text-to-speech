from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import uuid
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text-to-Speech Service")

# Create necessary directories
OUTPUT_DIR = Path("/tmp/tts_outputs")
UPLOAD_DIR = Path("/tmp/tts_uploads")
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Global model variables
model = None
tokenizer = None

def load_model():
    """Load the Qwen TTS model"""
    global model, tokenizer
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        if device == "cpu":
            model = model.to(device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Request models
class TTSRequest(BaseModel):
    text: str
    voice_preset: str = "default"
    speed: float = 1.0
    pitch: float = 1.0

class VoiceCloneRequest(BaseModel):
    text: str
    speed: float = 1.0

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    html_file = Path("/app/static/index.html")
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse(content="<h1>TTS Service Running</h1>", status_code=200)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

@app.post("/api/tts/design")
async def voice_design(request: TTSRequest):
    """Generate speech using voice design parameters"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        logger.info(f"Generating TTS for text: {request.text[:50]}...")
        
        # Generate unique filename
        output_id = str(uuid.uuid4())
        output_file = OUTPUT_DIR / f"{output_id}.wav"
        
        # Prepare input
        inputs = tokenizer(request.text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate audio
        with torch.no_grad():
            # This is a simplified version - actual Qwen TTS API might differ
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                max_length=1024
            )
        
        # Save audio file
        # Note: This is a placeholder - actual audio generation depends on Qwen TTS API
        sample_rate = 12000  # 12Hz as mentioned in model name
        
        # For now, create a simple tone (will be replaced by actual model output)
        duration = 2.0
        frequency = 440.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
        
        torchaudio.save(str(output_file), audio_tensor, sample_rate)
        
        logger.info(f"Audio saved to: {output_file}")
        
        return {
            "success": True,
            "audio_id": output_id,
            "download_url": f"/api/download/{output_id}"
        }
    except Exception as e:
        logger.error(f"Error in voice_design: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts/clone")
async def voice_clone(
    text: str = File(...),
    audio_file: UploadFile = File(...)
):
    """Generate speech using voice cloning"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        logger.info(f"Voice cloning for text: {text[:50]}...")
        
        # Save uploaded audio file
        upload_id = str(uuid.uuid4())
        upload_path = UPLOAD_DIR / f"{upload_id}_{audio_file.filename}"
        
        with open(upload_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Generate unique filename for output
        output_id = str(uuid.uuid4())
        output_file = OUTPUT_DIR / f"{output_id}.wav"
        
        # Load reference audio
        reference_audio, ref_sr = torchaudio.load(str(upload_path))
        
        # Generate audio with voice cloning
        # Note: This is a placeholder - actual voice cloning depends on Qwen TTS API
        logger.info("Generating cloned voice audio...")
        
        # For now, create a simple tone (will be replaced by actual model output)
        duration = 2.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440.0 * t)
        audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
        
        torchaudio.save(str(output_file), audio_tensor, sample_rate)
        
        logger.info(f"Cloned audio saved to: {output_file}")
        
        return {
            "success": True,
            "audio_id": output_id,
            "download_url": f"/api/download/{output_id}"
        }
    except Exception as e:
        logger.error(f"Error in voice_clone: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{audio_id}")
async def download_audio(audio_id: str):
    """Download generated audio file"""
    audio_file = OUTPUT_DIR / f"{audio_id}.wav"
    if not audio_file.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_file, media_type="audio/wav", filename=f"{audio_id}.wav")

@app.get("/api/voices")
async def list_voices():
    """List available voice presets"""
    return {
        "voices": [
            {"id": "default", "name": "Default Voice"},
            {"id": "female", "name": "Female Voice"},
            {"id": "male", "name": "Male Voice"},
            {"id": "child", "name": "Child Voice"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
