from fastapi import UploadFile, FastAPI, WebSocket
import os
import torch
import requests
import shutil
import nltk
import time



from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration
import google.generativeai as genai
from bardapi import Bard
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from controllers import *
from config import BARD_KEY

app = FastAPI()

# origins = [
#     "*"
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

os.environ["GOOGLE_API_KEY"] = BARD_KEY
genai.configure(api_key= os.environ['GOOGLE_API_KEY'])

llmmodel = genai.GenerativeModel('gemini-pro')

CAPTION_URL = "https://88fc-34-125-148-121.ngrok-free.app/prompt"

class Caption(BaseModel):
    caption: str


class EmotionParameter(BaseModel):
    image_description: str
    emotion: str

class Prompt(BaseModel):
    prompt: str

async def send_notification():
    print("Sending a notification before starting the FastAPI server......")


@app.websocket("/notify")
async def notify(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Server is ready to use")


@app.get("/")
async def index():
    return {"message": "Hello World"}


def get_image_url(image_path):
    response = requests.get(image_path)
    return response.url


@app.post("/caption")
async def UploadImage(file: UploadFile):
    start_time = time.time()
    upload_dir = os.path.join(os.getcwd(), "uploads")
    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # get the destination path
    destination = os.path.join(upload_dir, file.filename)  # type: ignore
    print(destination)

    # copy the file contents
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    caption = predict_step([destination])
    caption2 = creatingCaption(caption)
    hashtag = hashtags(caption)
    print("--- %s seconds ---" % (time.time() - start_time))
    return {
        'image-description': caption,
        'hash-tags': hashtag,
        'captions': caption2
    }

@app.post("/prompt")
async def prompting(prompt: Prompt):
    response = llmmodel.generate_content(prompt.prompt)
    return response.text

    

def predict_step(image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        # unconditional image captioning
        inputs = processor(images[0], return_tensors="pt")
        out = model.generate(**inputs, num_beams=5, max_new_tokens=30, # type: ignore
                             repetition_penalty=1.0, length_penalty=1.0, temperature=1)  
        return processor.decode(out[0], skip_special_tokens=True)

def creatingCaption(caption):
    response = llmmodel.generate_content(str(f'Create a few captions with emojis and hashtags for the {caption}'))
    return response.text



@app.post("/poetic")
def poeticCaption(caption: Caption):
    response = llmmodel.generate_content(str(f'Make this caption poetic {caption.caption}'))
    return response.text

    


@app.post("/quote")
def createQuote(caption: Caption):
    # return Bard().get_answer(str(f'Find a few quotes related to this caption {caption.caption}'))['content']
    response = llmmodel.generate_content(str(f'Find a few quotes related to this caption {caption.caption}'))
    return response.text


@app.post("/emotion")
def emotionCaption(emotion: EmotionParameter):
    # return Bard().get_answer(str(f'Create a few captions with emojis and hashtags for the {emotion.image_description}, make sure that the captions convey {emotion.emotion} emotion'))['content']
    response = llmmodel.generate_content(str(f'Create a few captions with emojis and hashtags for the {emotion.image_description}, make sure that the captions convey {emotion.emotion} emotion'))
    return response.text


@app.post('/questions')
def populateQuestions(caption: Caption):
    # return Bard().get_answer(str(f'Give some interesting question on {caption.caption}'))['content']
    response = llmmodel.generate_content(str(f'Give some interesting question on {caption.caption}'))
    return response.text


@app.post("/recheckCaption")
def recheckCaption(file: UploadFile):
    upload_dir = os.path.join(os.getcwd(), "uploads")
    # Create the upload directory if it doesn't exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # get the destination path

    destination = os.path.join(upload_dir, file.filename)  # type: ignore
    print(destination)

    # copy the file contents
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = open(destination, "rb").read()
    bard_answer = Bard().ask_about_image(
        'Create a few captions with emojis and hashtags for the', image)
    print(bard_answer['content'])
    caption2 = creatingCaption(bard_answer['content'])
    return {
        'caption': bard_answer['content']
    }