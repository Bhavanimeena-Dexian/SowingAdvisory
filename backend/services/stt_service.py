import os
import subprocess
import whisperx
import torch
from fastapi import HTTPException

# Setting up directories
TEMP_AUDIO_DIR = "temp_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Load WhisperX model
device = "cpu"
stt_model = whisperx.load_model("small", device, compute_type="float32")

def convert_speech_to_text(audio_file):
    """
    Processes an audio file and transcribes it using WhisperX.
    """
    try:
        allowed_formats = (".wav", ".mp3", ".m4a")
        if not audio_file.filename.endswith(allowed_formats):
            raise HTTPException(status_code=400, detail="Unsupported file format. Allowed formats: .wav, .mp3, .m4a.")

        # Define file paths
        input_audio_path = os.path.join(TEMP_AUDIO_DIR, audio_file.filename)
        converted_audio_path = os.path.join(TEMP_AUDIO_DIR, "converted.wav")

        # Save uploaded file locally
        with open(input_audio_path, "wb") as buffer:
            buffer.write(audio_file.file.read())

        # Convert to WAV format using FFmpeg for compatibility
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_audio_path, "-ar", "16000", "-ac", "1", converted_audio_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {e}")

        if not os.path.exists(converted_audio_path):
            raise HTTPException(status_code=500, detail="Converted audio file is missing.")

        # Transcribe using WhisperX
        audio = whisperx.load_audio(converted_audio_path)
        result = stt_model.transcribe(audio)

        transcribed_text = " ".join([segment["text"] for segment in result.get("segments", [])])
        if not transcribed_text.strip():
            raise HTTPException(status_code=500, detail="WhisperX returned an empty transcription.")

        return {"text": transcribed_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
