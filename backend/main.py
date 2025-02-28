import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisperx
import torch
import requests  

# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend communication (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load WhisperX Model for Speech-to-Text (STT)
device = "cuda" if torch.cuda.is_available() else "cpu"
stt_model = whisperx.load_model("large-v2", device, compute_type="float32")

# Ensure temp_audio directory exists
TEMP_AUDIO_DIR = "temp_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)  # Creates folder if not exists

@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {"message": "Welcome to the Sowing Advisory Chatbot API!"}


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Receives an audio file, saves it, and transcribes it using WhisperX.
    """

    try:
        audio_path = os.path.join(TEMP_AUDIO_DIR, file.filename)

        # Save the uploaded file
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())

        # ✅ Debugging: Check if file is saved
        print(f"File saved at: {audio_path}")

        # ✅ Debugging: Check if file exists
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail=f"File {audio_path} does not exist.")

        # Transcribe the audio
        audio = whisperx.load_audio(audio_path)
        result = stt_model.transcribe(audio)
        transcribed_text = result["text"]

        return {"text": transcribed_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mistral")
async def process_text(data: dict):
    """
    Placeholder API for Mistral (RAG Model).
    Replace this with actual Mistral API call when available.
    """
    user_input = data.get("text", "No input provided")

    # 🛑 Temporary response (Replace this when Mistral is ready)
    rag_response = f"[MISTRAL-RAG RESPONSE] Processing: {user_input}"

    return {"reply": rag_response}


@app.post("/api/tts")
async def text_to_speech(data: dict):
    """
    Placeholder API for Text-to-Speech (TTS).
    Replace this with actual TTS code when ready.
    """
    response_text = data.get("text", "No text provided")

    # 🛑 Temporary response (Replace this when TTS is ready)
    fake_audio_url = f"http://127.0.0.1:8000/static/fake_audio.wav"

    return {"audio_url": fake_audio_url, "text": response_text}
