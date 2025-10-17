# importing dependencies
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
import requests
import boto3
from pytubefix import YouTube
from points_generation import KeywordAndSentencesExtractor, TrelloIntegration
import whisper
from celery import Celery
from db import Users, CeleryTaskDB, Data
import uuid
from uuid import UUID
from ml.audioProcessing import createTranscriptions
import modal
from pydantic import BaseModel

# creating s3 client
s3 = boto3.resource(
        's3',
        region="ap-south-1"
    )
# global paths
AUDIO_PATH = "video-links"
TRANSCRIPTS_PATH = "transcripts"

# celery app instantiation
celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

class AudioData(BaseModel):
    audioLink: str
    userId: UUID
    audioId: UUID

# whisper model initialization
model_w = whisper.load_model("base")

# uploading the transcriptions from local to s3
def file_upload(audio_path: str, file_id: str):
    try:
        s3.upload_file(Bucket = "utube-proj-v1", Filename = audio_path, Key = f"audio/audio-{file_id}.mp4")
        return True
    except Exception as e:
        print(f"Error in file uploading : {e}")
        return False

# downloading the video to local
def download_video(video_link: str):
    yt = YouTube(video_link)
    try:
        stream = yt.streams.filter(only_audio=True, subtype="mp3").first() # taking the first stream from the set of StreamQuery objects
        print(stream)
        try:
            audio_path = "./audio"
            filename = "audio-v1.mp4"
            stream.download(audio_path, filename=filename)
            return audio_path+f"/{filename}"
        except Exception as e:
            print(f"error downloading the video to local : {e}")
            return None
    except Exception as e:
        print(f"error in loading the video : {e}")
        return None

# processing the audio
def process_audio(audio_path: str):
    try:
        result = model_w.transcribe(audio=audio_path)
        return result["text"]
    except Exception as e:
        print(f"Fucked up again : {e}") # Correct this line of code😂
        return None

web_app = FastAPI()

# env variables fetching
load_dotenv()
trello_key = os.getenv("TRELLO_KEY")
trello_token = os.getenv("TRELLO_TOKEN")

# this function will a background process
@celery_app.task()
def processTranscriptions(transcriptions: str, callback_url: str = None):
    extractor = KeywordAndSentencesExtractor(transcriptions)
    board_desc = extractor.text_extraction()
    keywords = extractor.keyword_extraction()
    board_data = {
        "board_desc": board_desc,
        "keywords": keywords
    }
    # CHECK FOR CALLBACK URLS
    final_board_data = Data(**board_data)
    return final_board_data

@web_app.post("/upload-video", response_class=JSONResponse)
async def upload_video(request: Request):
    data = await request.json()
    video_link = data.get("video-link")
    user_id = uuid.uuid4()
    audio_id = uuid.uuid4()

    user_payload = {
        "audioLink": video_link,
        "userId": user_id,
        "audioId": audio_id
    }

    # offloading to modal occurs here
    try:
        AudioProcessor = modal.Cls.lookup("whisper-transcriber-for-audio", "AudioProcessing")
        processor = AudioProcessor(AudioData(**user_payload))
        try:
            processed_audio_details = processor.finalProcessingAndPushing.remote()
        except Exception as e:
            print(f"Encountered issue in processing audio in modal; message : {e}")
            processed_audio_details = {"transcriptions": ""}
    except Exception as e:
        print(f"Encountered issue in starting the modal server; message: {e}")
        processed_audio_details = {"transcriptions": ""}

    # offloading the transcriptions to the celery worker
    transcriptions = processed_audio_details.get("transcriptions")
    if transcriptions != "":
        # offloading task here
        processTranscriptions(transcriptions)
    else:
        raise ValueError("No transcriptions found for the video")

    boardName = data.get('board-name')
    user_context = data.get("user-context")
    print(video_link)
    audio_path = download_video(video_link)

    fileId = str(uuid.uuid4())

    print(f"{audio_path}")
    transcriptions = createTranscriptions(audio_path)
    # NLP class initialization
    extractor = KeywordAndSentencesExtractor(transcriptions)
    desc = extractor.text_extraction()
    keywords = extractor.keyword_extraction()
    # trello class init
    trello = TrelloIntegration(name=boardName, desc=desc, token=trello_token, api_key=trello_key, default_lists=False, n=3)
    await trello._add_cards_to_list(keywords, user_context) # naming is not the encouraged way
    board_url = trello._get_board_url

    return JSONResponse(content={"url for the board": board_url}, status_code=200)

@web_app.post("/fetch-board", response_class=JSONResponse)
async def fetch_board_details(request: Request):
    data = await request.json()
    boardName = data.get('board-name')
    user_context = data.get("user-context")

    user_data = {
        "board-name": boardName,
        "user-context": user_context
    }
    return JSONResponse(content=user_data, status_code=200)

# callback endpoint
@web_app.post("/video/callback")
async def task_callback(request: Request, internal_secret: str = Header(None)):
    data = await request.json()
    # authorization
    secret = data.get("internal_secret")
    if secret!=os.getenv("INTERNAL-SECRET"):
        raise HTTPException(status=403, detail="forbidden access")
    else:
        task_id = data.get('task-id')
        # CODE FOR PUSHING THE DATA INTO THE DB



