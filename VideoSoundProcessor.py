import whisper
# import spacy
# from  natasha import *
# from ipymarkup import show_dep_ascii_markup, show_span_ascii_markup
import ffmpeg as ff
import os
import TextProcessor

def sepSound(input: str) -> str:
    if (not os.path.exists("temp")):
        os.mkdir("temp")
    stream = ff.input(input)
    output_file = f'temp/audio.mp3'
    stream = ff.output(stream, output_file, acodec='mp3')
    ff.run(stream, overwrite_output=True)
    return output_file

def extractText(audioName: str) -> str:
    model  = whisper.load_model("base")
    result = model.transcribe(audioName)
    return str(result["text"])

# def getKeywords(text):
#     nlp = spacy.load("ru_core_news_md")
#     doc = nlp(text["text"])
#     spans = [(_.start_char, _.end_char, _.label_) for _ in doc.ents ]
#     show_span_ascii_markup(doc.text, spans)

# inp = "/home/user1/WorkFolder/input/video.mp4"
# sepSound(inp)
# t = extractText()

def predict(videoPath: str) -> list[tuple[str, float]]:
    sound = sepSound(videoPath)
    t = extractText(sound)
    return TextProcessor.predict(t)
