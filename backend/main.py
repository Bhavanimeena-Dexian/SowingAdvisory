import os
import base64
import subprocess
import uuid  # Importing uuid to generate unique filenames
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import whisperx
import torch
import requests
from gtts import gTTS

# Initializing FastAPI application
app = FastAPI()

# Enabling CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allowing all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensuring necessary directories exist
TEMP_AUDIO_DIR = "temp_audio"
STATIC_AUDIO_DIR = "static"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
os.makedirs(STATIC_AUDIO_DIR, exist_ok=True)

# Mounting the static directory to serve generated audio files
app.mount("/static", StaticFiles(directory=STATIC_AUDIO_DIR), name="static")

# Loading the WhisperX model for speech-to-text processing
device = "cuda" if torch.cuda.is_available() else "cpu"
stt_model = whisperx.load_model("large-v2", device, compute_type="float32")


@app.get("/")
async def root():
    """Handling the root endpoint to verify if the API is running."""
    return {"message": "Sowing Advisory Chatbot API is running"}


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Handling audio file uploads, converting them to WAV format if necessary, 
    and transcribing them using the WhisperX model.
    """
    try:
        # Defining allowed audio file formats
        allowed_formats = (".wav", ".mp3", ".m4a")
        if not file.filename.endswith(allowed_formats):
            raise HTTPException(status_code=400, detail="Unsupported file format. Allowed formats: .wav, .mp3, .m4a.")

        # Defining file paths
        input_audio_path = os.path.join(TEMP_AUDIO_DIR, file.filename)
        converted_audio_path = os.path.join(TEMP_AUDIO_DIR, "converted.wav")

        # Saving the uploaded file locally
        with open(input_audio_path, "wb") as buffer:
            buffer.write(await file.read())

        # Converting the file to WAV format using FFmpeg for compatibility
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_audio_path, "-ar", "16000", "-ac", "1", converted_audio_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {e}")

        # Checking if the converted file exists
        if not os.path.exists(converted_audio_path):
            raise HTTPException(status_code=500, detail="Converted audio file is missing after FFmpeg conversion.")

        # Transcribing the audio using WhisperX
        audio = whisperx.load_audio(converted_audio_path)
        result = stt_model.transcribe(audio)

        # Extracting transcribed text from the output
        transcribed_text = " ".join([segment["text"] for segment in result.get("segments", [])])

        # Raising an error if the transcription is empty
        if not transcribed_text.strip():
            raise HTTPException(status_code=500, detail="WhisperX returned an empty transcription.")

        return {"text": transcribed_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mistral")
async def process_mistral(data: dict):
    """
    Handling text input and returning a placeholder response 
    as a part of the Mistral AI integration.
    """
    user_input = data.get("text", "No input provided")

    # Defining a placeholder response
    mistral_response = f"Mistral AI placeholder response for: {user_input}"

    # Sending the response to the text-to-speech API
    tts_response = requests.post(
        "http://127.0.0.1:8000/api/tts",
        json={"text": mistral_response}
    ).json()

    return {
        "reply": mistral_response,
        "tts_audio_url": tts_response.get("audio_url", ""),
    }


@app.post("/api/tts")
async def text_to_speech(data: dict):
    """
    Handling text-to-speech conversion using gTTS and returning an audio URL.
    """
    response_text = data.get("text", "No text provided")

    # Generating a unique filename for the output audio file
    unique_filename = f"tts_{uuid.uuid4().hex}.mp3"
    audio_output_path = os.path.join(STATIC_AUDIO_DIR, unique_filename)

    try:
        # Converting text to speech using gTTS
        tts = gTTS(text=response_text, lang="en")
        tts.save(audio_output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text-to-speech conversion failed: {e}")

    # Returning the generated audio file URL
    return {"audio_url": f"http://127.0.0.1:8000/static/{unique_filename}", "text": response_text}
