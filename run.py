from argparse import ArgumentParser
import numpy as np
import pandas as pd
import TextProcessor, ImageProcessor, VideoSoundProcessor
import os
import torch

torch.cuda.init()
labels = TextProcessor.loadLabels()
print(len(labels))
# ImageProcessor.prepare()

def processVideo(path: str, title: str, description: str) -> list[str]:
    # В результате получаем [(p1,s1), (p2, s2), ..., (pn, sn)],
    # где pn - это тег под указанным номером, sn - его вероятность
    titleTags = TextProcessor.predict(title, labels)
    descriptionTags = TextProcessor.predict(description, labels)
    # imageTags = ImageProcessor.predict(path, labels)
    soundTags = VideoSoundProcessor.predict(path, labels)
    tags = [x for x in titleTags + descriptionTags + imageTags + soundTags]
    topTags = {}
    for tag in tags:
        if (tag[0] in topTags.keys()):
            topTags[tag] += tag[1]
        topTags[tag] = tag[1]
    return tags


def process(file: str, videoDir: str, output: str) -> None:
    # print(file, videoDir, output)
    data = pd.read_csv(file)[['video_id', 'title', 'description']]
    # Другая программа как-то читает откуда-то векторы
    ids = pd.Series()
    tags = pd.Series()
    for i, row in data.iterrows():
        if (hash(i) % 10 == 0):
            print(f"Processing video {i}...")
        i += 1
        video_id = str(row['video_id'])
        videoPath = os.path.join(videoDir, video_id + ".mp4")
        if (not os.path.exists(videoPath)):
            continue
        topTags = processVideo(videoPath, str(row['title']), str(row['description']))
        ids[i] = video_id
        tags[i] = topTags
    result = pd.DataFrame(ids + tags, columns=["video_id", "predicted_tags"])
    result.to_csv(output)


# Парсим аргументы командной строки и запускаем программу
if __name__ == "__main__":
    parser = ArgumentParser(description="VideoTags v.1.0. Performs video tagging using deep learning technologies.")
    parser.add_argument('-f', '--file', type=str, help="Path to a video CSV index file.")
    parser.add_argument('-d', '--video-directory', type=str, default='Videos/', help="Path to a video directory.")
    parser.add_argument('-o', '--output', type=str, default="sample_submission.csv", help="Path to a file with prediction results.")
    args = parser.parse_args()
    print("CUDA availability: ", torch.cuda.is_available())
    process(args.file, args.video_directory, args.output)
