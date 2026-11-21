
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# 1. Load the knowledge base
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"

with open(KB_PATH, "r", encoding="utf-8") as file:
    passages = json.load(file)

if not isinstance(passages, list) or not passages:
    raise ValueError("knowledge_base.json must contain a non-empty JSON list.")

if not all(isinstance(p, str) for p in passages):
    raise ValueError("Every knowledge-base passage must be a string.")


# --------------------------------------------------
# 2. Load the embedding model and build the index
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Creating passage embeddings...")

passage_embeddings = model.encode(
    passages,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# BM25 keyword index
def tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9_.-]+", text.lower())


tokenized_passages = [tokenize(p) for p in passages]
bm25 = BM25Okapi(tokenized_passages)

print(f"Loaded {len(passages)} passages successfully.")


# --------------------------------------------------
# 3. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(title="RAG Colosseum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 4. Request and response formats
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float


# --------------------------------------------------
# 5. Hybrid retrieval: semantic + keyword search
# --------------------------------------------------

def retrieve(question: str, top_k: int):
    # Semantic similarity
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    semantic_scores = np.dot(
        passage_embeddings,
        query_embedding
    )

    # Convert cosine similarities to the 0-1 range
    semantic_scores = np.clip(
        semantic_scores,
        0.0,
        1.0
    )

    # BM25 keyword scores
    keyword_scores = np.array(
        bm25.get_scores(tokenize(question)),
        dtype=float
    )

    # Normalize BM25 scores
    max_bm25 = float(keyword_scores.max())

    if max_bm25 > 0:
        keyword_scores = keyword_scores / max_bm25
    else:
        keyword_scores = np.zeros(len(passages))

    # Combine both retrieval methods
    combined_scores = (
        0.70 * semantic_scores +
        0.30 * keyword_scores
    )

    # Sort passage indices by descending score
    ranked_indices = np.argsort(combined_scores)[::-1]

    selected_indices = ranked_indices[:top_k]

    results = [
        {
            "index": int(i),
            "score": float(combined_scores[i])
        }
        for i in selected_indices
    ]

    return results


# --------------------------------------------------
# 6. Extract relevant sentences from retrieved passages
# --------------------------------------------------

STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and",
    "or", "of", "to", "in", "on", "for", "with", "by",
    "what", "which", "who", "when", "where", "how",
    "does", "do", "did", "can", "could", "would",
    "should", "about", "from", "that", "this", "it",
    "its", "their", "they", "be", "as", "at", "than",
    "between", "compare", "explain", "describe"
}


def important_words(text: str):
    return {
        word for word in tokenize(text)
        if word not in STOPWORDS and len(word) > 1
    }


def extract_answer(question: str, retrieved):
    query_words = important_words(question)

    if not query_words:
        return None

    candidates = []

    for item in retrieved:
        passage_index = item["index"]
        passage = passages[passage_index]

        # Split each passage into individual sentences
        sentences = re.split(r"(?<=[.!?])\s+", passage.strip())

        for sentence in sentences:
            sentence_words = important_words(sentence)

            if not sentence_words:
                continue

            overlap = query_words.intersection(sentence_words)

            # Reward sentences containing more query terms
            coverage = len(overlap) / len(query_words)

            # Penalize sentences with very little query relevance
            if not overlap:
                continue

            sentence_score = (
                0.75 * coverage +
                0.25 * item["score"]
            )

            candidates.append({
                "sentence": sentence.strip(),
                "score": sentence_score,
                "source": passage_index,
                "coverage": coverage
            })

    if not candidates:
        return None

    # Highest-scoring sentence first
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Keep relevant sentences, avoiding duplicates
    selected = []
    seen = set()

    for candidate in candidates:
        sentence = candidate["sentence"]

        if sentence.lower() in seen:
            continue

        if candidate["coverage"] < 0.15:
            continue

        selected.append(candidate)
        seen.add(sentence.lower())

        if len(selected) >= 3:
            break

    if not selected:
        return None

    # Return one or more evidence sentences.
    # This also allows simple cross-passage synthesis.
    return " ".join(item["sentence"] for item in selected)


# --------------------------------------------------
# 7. Query endpoint
# --------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):

    question = request.question.strip()

    if not question:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.0
        )

    # Retrieve relevant passages
    retrieved = retrieve(question, request.top_k)

    if not retrieved:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.0
        )

    # Extract an answer from the retrieved evidence
    answer = extract_answer(question, retrieved)

    # Confidence is a retrieval-based estimate, not a
    # statistically calibrated probability.
    best_score = retrieved[0]["score"]

    confidence = float(
        np.clip(best_score, 0.0, 1.0)
    )

    # Abstain when retrieval is weak or no answer was found
    if answer is None or best_score < 0.35:
        answer = None
        confidence = min(confidence, 0.30)

    return QueryResponse(
        answer=answer,
        sources=[item["index"] for item in retrieved],
        scores=[
            round(item["score"], 4)
            for item in retrieved
        ],
        confidence=round(confidence, 4)
    )


# --------------------------------------------------
# 8. Health-check endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "RAG Colosseum API is ready",
        "passages_loaded": len(passages)
    }
