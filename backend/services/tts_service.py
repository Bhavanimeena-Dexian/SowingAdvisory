import os
from gtts import gTTS
import uuid

# Ensure the static/tts directory exists
OUTPUT_DIR = "static/tts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_speech(text: str):
    """
    Converts text to speech using gTTS and saves the audio file.
    Returns the file URL.
    """
    try:
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Generate speech
        tts = gTTS(text)
        tts.save(filepath)

        # Return the URL to access the generated speech file
        return {"audio_url": f"http://127.0.0.1:8000/static/tts/{filename}", "text": text}
    
    except Exception as e:
        return {"error": str(e)}
