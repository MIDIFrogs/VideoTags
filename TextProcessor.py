import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.autonotebook import tqdm
from torch import Tensor

topN = 5

model = SentenceTransformer('DeepPavlov/rubert-base-cased-sentence', )
dim = 768
index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)

def get_tags() -> list:
    taxonomy = pd.read_csv("IAB_tags.csv")
    tags = {}
    for i, row in tqdm(taxonomy.iterrows()):
        if isinstance(row['Уровень 1 (iab)'], str):
            tags[row['Уровень 1 (iab)']] = 0
        if isinstance(row['Уровень 2 (iab)'], str):
            tags[row['Уровень 1 (iab)']+ ": "+row['Уровень 2 (iab)']] = 0
        if isinstance(row['Уровень 3 (iab)'], str):
            tags[row['Уровень 1 (iab)']+ ": "+row['Уровень 2 (iab)']+": "+row['Уровень 3 (iab)']] = 0
    return list(tags.keys())

def loadLabels() -> str:
    tags_list = get_tags()
    with open("VTags.np", "rb") as vtags:
        tagsVector = np.load(vtags)
        index.add(tagsVector)
    return tags_list

def predict(text: str, labels: list[str]) -> list[tuple[str, float]]:
    vector = model.encode(text, convert_to_tensor=True).cpu().numpy()
    titleScores, titlePrediction = index.search(np.array([vector]), topN)
    titleScores = titleScores[0]
    titlePrediction = titlePrediction[0]
    results = []
    for i, pred in enumerate(titlePrediction):
        label = labels[titlePrediction[i]]
        score = titleScores[i]
        results.append((label, score))
    return results