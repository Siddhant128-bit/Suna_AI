import whisper

# Load the Whisper model (use 'base', 'small', 'medium', or 'large' for varying accuracy)
model = whisper.load_model("small")  # "base" is lightweight, "large" is more accurate but slower

def transcribe_and_italicize(audio_path='test.wav'):
    """
    Transcribes speech to text and returns an italicized English transcription.
    """
    # Transcribe the audio
    result = model.transcribe(audio_path, task="translate")  # Translates to English
    transcription = result["text"]

    # Return the italicized version of the transcription
    return f"*{transcription}*"

