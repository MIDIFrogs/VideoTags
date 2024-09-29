
# from  natasha import *
# from ipymarkup import show_dep_ascii_markup, show_span_ascii_markup

# def sepSound(input: str) -> str:
#     if (not os.path.exists("temp")):
#         os.mkdir("temp")
#     stream = ff.input(input)
#     output_file = f'temp/audio.mp3'
#     stream = ff.output(stream, output_file, acodec='mp3')
#     ff.run(stream, overwrite_output=True)
#     return output_file

# def extractText(audioName: str) -> str:
#     model  = whisper.load_model("base")
#     result = model.transcribe(audioName)
#     return str(result["text"])

# def getKeywords(text):
#     nlp = spacy.load("ru_core_news_md")
#     doc = nlp(text["text"])
#     spans = [(_.start_char, _.end_char, _.label_) for _ in doc.ents ]
#     show_span_ascii_markup(doc.text, spans)

# inp = "/home/user1/WorkFolder/input/video.mp4"
# sepSound(inp)
# t = extractText()

# def predict(videoPath: str) -> list[tuple[str, float]]:
#     sound = sepSound(videoPath)
#     t = extractText(sound)
#     return TextProcessor.predict(t)
import whisper
import spacy
from ipymarkup import show_dep_ascii_markup, show_span_ascii_markup
import ffmpeg as ff
import os
import uuid
import TextProcessor
import torch

if (torch.cuda.is_available()):
    global cuda
    if (torch.cuda.is_available()):
        device = "cuda"
    else:
        device = "cpu"
    global model
    model  = whisper.load_model("tiny").to(device)

def sepSound(input):
    if not os.path.exists("temp"):
        os.mkdir("temp")
    streamIn = ff.input(input)
    unique_id = str(uuid.uuid4())
    output_file = f'temp/audio_{unique_id[:5]}.mp3'
    streamOut = ff.output(streamIn, output_file, acodec='mp3')
    ff.run(streamOut, overwrite_output=True)
    return output_file

def ExtractText(audioName):
    if (torch.cuda.is_available):
        with torch.cuda.device(device):
            result = model.transcribe(audioName, language="ru")
    else:
        result = model.transcribe(audioName, language="ru")
    return result["text"]

def getKeywords(text):
    nlp = spacy.load("ru_core_news_md")
    doc = nlp(text)
    spans = [(_.start_char, _.end_char, _.label_) for _ in doc.ents ]
    show_span_ascii_markup(doc.text, spans)

def predict(videoPath: str, labels: list[str]) -> list[tuple[str, float]]:
    sound = sepSound(videoPath)
    t = ExtractText(sound)
    return TextProcessor.predict(t, labels)

def process_videos_in_folder(folder_path: str):

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4") : 
            video_path = os.path.join(folder_path, filename)
            print(f"Обрабатываем видео: {video_path}")
            tags = predict(video_path)
            print(f"Теги для видео {filename}: {tags}")
# ВАНЯ, РАЗБУДИ МЕНЯ, КОГДА ПРОСНЁШЬСЯ. РАССКАЖУ, ЧТО ИСПРАВИЛ. Я В ЗАЛЕ СПЛЮ