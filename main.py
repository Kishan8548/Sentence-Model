from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

class MatchRequest(BaseModel):
    lost_text: str
    found_items: list[str]

@app.post("/match")
def match_items(data: MatchRequest):
    lost_embedding = model.encode([data.lost_text])
    found_embeddings = model.encode(data.found_items)

    scores = cosine_similarity(lost_embedding, found_embeddings)[0]

    results = []
    for item, score in zip(data.found_items, scores):
        results.append({
            "item": item,
            "match_percent": round(float(score * 100), 2)
        })

    results.sort(key=lambda x: x["match_percent"], reverse=True)
    return results[:5]


# ✅ ADD ONLY THIS PART
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
