from argparse import ArgumentParser
import numpy as np
import pandas as pd
import TextProcessor, ImageProcessor, VideoSoundProcessor

def processVideo(path: str, title: str, description: str) -> list[str]:
    # В результате получаем [(p1,s1), (p2, s2), ..., (pn, sn)],
    # где pn - это тег под указанным номером, sn - его вероятность
    titleTags = TextProcessor.predict(title)
    descriptionTags = TextProcessor.predict(description)
    imageTags = ImageProcessor.predict(path)
    soundTags = VideoSoundProcessor.predict(path)
    return [x[0] for x in titleTags + descriptionTags + imageTags + soundTags]


def process(file: str, videoDir: str, output: str) -> None:
    TextProcessor.setupIndex()
    # print(file, videoDir, output)
    data = pd.read_csv(file)[['video_id', 'title', 'description']]
    # Другая программа как-то читает откуда-то векторы
    ids = pd.Series()
    tags = pd.Series()
    for i, row in data.iterrows():
        if (hash(i) % 10 == 0):
            print(f"Processing video {i}...")
        video_id = str(row['video_id'])
        videoPath = videoDir + video_id + ".mp4"
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
    process(args.file, args.video_directory, args.output)
