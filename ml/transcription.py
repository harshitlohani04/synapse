import modal
import whisper

modal_app = modal.App("whisper-transcriber-for-audio")
model = whisper.load_model("base")

@modal_app.function(gpu="A10")
def createTranscriptions(audio_path: str) -> str:
    try:
        result = model.transcribe(audio=audio_path)
        return result["text"]
    except Exception as e:
        print(f"Encountered issue in transcribing the audio : {e}")
        return None

