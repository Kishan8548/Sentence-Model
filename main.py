from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="DropZone AI Matching Service",
    description="Semantic text similarity for lost & found items",
    version="1.0"
)

 
model = SentenceTransformer("all-MiniLM-L6-v2")

 

class FoundItem(BaseModel):
    postId: str
    description: str


class MatchRequest(BaseModel):
    lost_text: str
    found_items: List[FoundItem]
 

class MatchResult(BaseModel):
    postId: str
    item: str
    match_percent: float


@app.get("/")
def root():
    return {"status": "DropZone AI backend is running"}


@app.post("/match", response_model=List[MatchResult])
def match_items(data: MatchRequest):
 
    lost_embedding = model.encode([data.lost_text])[0]
 
    found_descriptions = [item.description for item in data.found_items]
    found_embeddings = model.encode(found_descriptions)
 
    scores = cosine_similarity(
        [lost_embedding],
        found_embeddings
    )[0]
 
    results = []
    for idx, score in enumerate(scores):
        results.append({
            "postId": data.found_items[idx].postId,
            "item": data.found_items[idx].description,
            "match_percent": round(float(score * 100), 2)
        })
 
    results.sort(key=lambda x: x["match_percent"], reverse=True)

    return results[:5]
