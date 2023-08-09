from fastapi import FastAPI, File , UploadFile
import os
import torch
import requests
import shutil
app = FastAPI()
import nltk
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('book')
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")



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

    hashtag = hashtags(caption)
    print("--- %s seconds ---" % (time.time() - start_time))
    return {
      'caption': caption, 
      'hash-tags': hashtag
    }   



def hashtags(caption):
  hashtag = []
  stop_words = set(stopwords.words('english'))
  word_tokens = word_tokenize(caption)
  filtered_sentence = [w for w in word_tokens if not w in stop_words]
  filtered_sentence = []
  for w in word_tokens:
    if w not in stop_words:
      filtered_sentence.append(w)
  X = len(filtered_sentence)
  for i in filtered_sentence:
    hashtag.append('#'+i)
  return hashtag