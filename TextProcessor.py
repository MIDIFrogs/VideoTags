import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

topN = 5

model = SentenceTransformer('DeepPavlov/rubert-base-cased-sentence', )
dim = 768
index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)

def setupIndex():
    with open("VTags.np", "rb") as vtags:
        tagVectors = np.load(vtags)
        index.add(tagVectors)

def predict(text: str) -> list[tuple[str, float]]:
    vector = model.encode(text, convert_to_tensor=True).cpu().numpy()
    titleScores, titlePrediction = index.search(np.array([vector]), topN)
    return list(zip(titleScores, titlePrediction))