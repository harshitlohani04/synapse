# importing dependencies
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import boto3
from pytubefix import YouTube
from points_generation import KeywordAndSentencesExtractor, TrelloIntegration
import whisper

# creating s3 client
s3 = boto3.client('s3')
LINK_PATH = "video-links"
TRANSCRIPTS_PATH = "transcripts"

# whisper model initialization
model_w = whisper.load_model("base")

# uploading the transcriptions from local to s3
def file_upload(impPoints: str, link: str):
    file_path = LINK_PATH + f"/{link}"
    try:
        s3.put_object(Bucket = "utube-proj-v1", Key = file_path, Body = link)
        return True
    except Exception as e:
        print(f"Error in file uploading : {e}")
        return False
# downloading the video to local
def download_video(video_link: str):
    yt = YouTube(video_link)
    try:
        stream = yt.streams.filter(only_audio=True).first() # taking the first stream from the set of StreamQuery objects
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
        print(f"Fucked up again : {e}") # Correct this line of code
        return None

web_app = FastAPI()

# env variables fetching
load_dotenv()
trello_key = os.getenv("TRELLO_KEY")
trello_token = os.getenv("TRELLO_TOKEN")

# temporary endpoint for prototyping
# to be separated later for upload and processing
@web_app.post("/upload-video", response_class=JSONResponse)
async def upload_video(request: Request):
    data = await request.json()
    video_link = data.get("video-link")
    boardName = data.get('board-name')
    user_context = data.get("user-context")
    print(video_link)
    audio_path = download_video(video_link)
    print(f"{audio_path}")
    transcriptions = process_audio(audio_path)
    # NLP class initialization
    extractor = KeywordAndSentencesExtractor(transcriptions)
    desc = extractor.text_extraction()
    keywords = extractor.keyword_extraction()
    # trello class init
    trello = TrelloIntegration(name=boardName, desc=desc, token=trello_token, api_key=trello_key, default_lists=False, n=3)
    await trello._add_cards_to_list(keywords, user_context) # naming is not the encouraged way
    board_url = trello._get_board_url

    return JSONResponse(content={"url for the board": board_url}, status_code=200)

