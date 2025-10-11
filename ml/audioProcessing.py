## MODAL MICROSERVICE FILE ##
import modal
import whisper
from pytubefix import YouTube
from pydantic import BaseModel
from uuid import uuid4, UUID
from db import VideoDB, ProcessedAudioData

modal_app = modal.App("whisper-transcriber-for-audio")

'''
ENTIRE AUDIO PROCESSING FILE NEEDS TO BE DEPLOYED ON MODAL AND USED AS A MICROSERVICE
'''

modalAppimage = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements-whisper.txt")
    .apt_install("ffmpeg")
    .add_local_python_source("db")
)
volume = modal.Volume.from_name("audio-processor-volume", create_if_missing=True)

class AudioData(BaseModel):
    audioLink: str
    userId: int
    audioId: UUID

@modal_app.cls(
    image=modalAppimage,
    secrets=[modal.Secret.from_name("aws-credentials")],
    volumes={"/audioFiles": volume}
)
class AudioProcessing:
    def __init__(self, audioInfo: AudioData):
        self.audioLink = audioInfo.audioLink
        self.userId = audioInfo.userId
        self.audioId = audioInfo.audioId
        self.videoDB = VideoDB()

    @modal.enter()
    def startupLoader(self): # function that loads the model one time globally
        self.model = whisper.load_model("base")

    @property
    def downloadVideo(self):
        yt = YouTube(self.audioLink)
        try:
            stream = yt.streams.filter(only_audio=True).first() # taking the first stream from the set of StreamQuery objects
            print(stream)
            try:
                audio_path = f"/audioFiles/audio_{self.userId}"
                filename = f"audio-{self.audioId}.mp4"
                stream.download(audio_path, filename=filename)
                return audio_path+f"/{filename}"
            except Exception as e:
                print(f"error downloading the video to volume : {e}")
                return None
        except Exception as e:
            print(f"error in loading the video : {e}")
            return None

    @modal_app.function(gpu="A10")
    def finalProcessingAndPushing(self) -> str:
        audioPath = self.downloadVideo
        try:
            result = self.model.transcribe(audio=audioPath)
            response = result["text"]
            try: # Trying to push into aws storage
                payload = {
                    "userId":self.userId,
                    "videoLink": self.audioLink,
                    "audioId": self.audioId,
                    "transcriptions": response
                }
                finalPayload = ProcessedAudioData(**payload)
                flagPushData = self.videoDB.flagPushDatainDB(finalPayload)
                if flagPushData:
                    print(f"Pushing the data to DB successful")
                    return payload
            except Exception as e:
                print(f"Encountered issue in the final pushing : {e}")
                return None
        except Exception as e:
            print(f"Encountered issue in transcribing the audio : {e}")
            return None
        


