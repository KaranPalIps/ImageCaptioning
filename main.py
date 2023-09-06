from fastapi import FastAPI, File , UploadFile
import os
import torch
import requests
import shutil
import nltk
import time

from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration
from bardapi import Bard
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from controllers import *

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

os.environ["_BARD_API_KEY"] = "agiRO1PpVlNxxwgojTuuXNgJfc6gqkT9rxNmcidbN-dPeniAreOC6dGiCqE0t4rkUFjNYA."

class Caption(BaseModel):
   caption: str

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
    dest = os.path.join(upload_dir, file.filename)
    print(dest)

    # copy the file contents
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    def predict_step(image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        # conditional image captioning
        text = "a photography of"
        inputs = processor(images[0], text, return_tensors="pt")

        out = model.generate(**inputs)

        # unconditional image captioning
        inputs = processor(images[0], return_tensors="pt")

        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)

    caption = predict_step([dest])
    caption2 = creatingCaption(caption)
    hashtag = hashtags(caption)
    print("--- %s seconds ---" % (time.time() - start_time))
    return {
      'image-description': caption, 
      'hash-tags': hashtag,
      'captions':caption2
    }   






def creatingCaption(caption):
   return Bard().get_answer(str(f'Create a few captions with emojis and hastags for the {caption}'))['content']

@app.post("/poetic")
def poeticCaption(caption: Caption):
   return Bard().get_answer(str(f'Make this caption poetic {caption.caption}'))['content']

@app.post("/qoute")
def createQuote(caption: Caption):
   return Bard().get_answer(str(f'Find a few qoutes related to this caption {caption.caption}'))['content']
