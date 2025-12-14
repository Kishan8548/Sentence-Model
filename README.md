# 🚨 DropZone AI Matching Service

An AI-powered semantic matching backend for a **Lost & Found application**.
This service compares a lost-item description with found-item descriptions
and returns the most relevant matches using **sentence embeddings**.

---

## 🧠 AI Concept Used

- **Sentence Transformers** (`all-MiniLM-L6-v2`)
- **Cosine Similarity**
- Converts text descriptions into embeddings
- Finds semantic similarity (not just keyword matching)

Example:
> "Black wallet near canteen"  
will match  
> "Found a dark leather wallet near college food court"

---

## 🛠 Tech Stack

- **FastAPI**
- **SentenceTransformers**
- **scikit-learn**
- **Python 3.10+**

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository
```bash
  git clone https://github.com/your-username/dropzone-ai-matching.git
  cd dropzone-ai-matching
```
### 2️⃣ Create virtual environment (recommended)
```bash
  python -m venv venv
  source venv/bin/activate   # Linux / macOS
  # venv\Scripts\activate    # Windows
```
### 3️⃣ Install dependencies
```bash
  pip install -r requirements.txt
```
### 4️⃣ Run the server
```bash
  uvicorn main:app --reload
```
Server will run at:
  http://127.0.0.1:8000

Swagger UI:
  http://127.0.0.1:8000/docs

## 🔁 API Usage

### POST `/match`

#### 📥 Request Body
```json
{
  "lost_text": "Black wallet lost near hostel gate",
  "found_items": [
    "Found brown leather wallet near main gate",
    "Lost phone near library",
    "Wallet found near hostel entrance"
  ]
}
```
### 📤 Response
```json
[
  {
    "item": "Wallet found near hostel entrance",
    "match_percent": 87.32
  }
]
```
