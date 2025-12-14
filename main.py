from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="DropZone AI Matching Service",
    description="Semantic text similarity for lost & found items",
    version="1.0"
)

# Load model once (important)
model = SentenceTransformer("all-MiniLM-L6-v2")

class MatchRequest(BaseModel):
    lost_text: str
    found_items: list[str]

class MatchResult(BaseModel):
    item: str
    match_percent: float
    
@app.get("/")
def root():
    return {"status": "DropZone AI backend is running"}

@app.post("/match", response_model=list[MatchResult])
def match_items(data: MatchRequest):
    lost_embedding = model.encode([data.lost_text])[0]
    found_embeddings = model.encode(data.found_items)

    scores = cosine_similarity(
        [lost_embedding],
        found_embeddings
    )[0]

    results = []
    for i, score in enumerate(scores):
        results.append({
            "item": data.found_items[i],
            "match_percent": round(float(score * 100), 2)
        })

    results.sort(key=lambda x: x["match_percent"], reverse=True)
    return results[:5]
