import json
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# Configuration and knowledge base
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"

with KB_PATH.open("r", encoding="utf-8") as f:
    passages = json.load(f)

if not isinstance(passages, list) or len(passages) != 50:
    raise ValueError("knowledge_base.json must contain exactly 50 passages.")
if not all(isinstance(p, str) and p.strip() for p in passages):
    raise ValueError("Every knowledge-base passage must be a non-empty string.")

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.I)

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "from", "with", "without",
    "and", "or", "but", "as", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "when", "where", "why", "how", "does", "do",
    "did", "can", "could", "would", "should", "explain", "describe", "tell",
    "me", "about", "between", "compare", "according", "passage"
}

def content_terms(text: str) -> set[str]:
    return {
        t for t in tokenize(text)
        if t not in STOPWORDS and len(t) > 1
    }

# Known terminology variants help bridge wording differences without
# introducing facts that are not in the supplied passages.
ALIASES = {
    "postgres": {"postgresql", "postgres"},
    "postgresql": {"postgresql", "postgres"},
    "innodb": {"innodb", "mysql"},
    "mysql": {"mysql", "innodb"},
    "raft": {"raft"},
    "paxos": {"paxos", "multi-paxos"},
    "adamw": {"adamw", "adam"},
    "sgd": {"sgd", "stochastic", "gradient", "descent"},
    "quic": {"quic", "http3", "http", "3"},
    "http3": {"http3", "quic"},
    "tls13": {"tls", "1.3", "tls13"},
    "tls12": {"tls", "1.2", "tls12"},
    "jwt": {"jwt", "json", "web", "tokens"},
    "pkce": {"pkce", "proof", "key", "code", "exchange"},
    "waf": {"waf", "firewall"},
    "api": {"api", "gateway"},
}

def expanded_terms(text: str) -> set[str]:
    terms = content_terms(text)
    expanded = set(terms)
    for term in list(terms):
        expanded.update(ALIASES.get(term, ()))
    return expanded


# ------------------------------------------------------------
# Build retrieval indexes once at process startup
# ------------------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Embedding 50 knowledge passages...")
passage_embeddings = model.encode(
    passages,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False,
).astype(np.float32)

bm25 = BM25Okapi([tokenize(p) for p in passages])
passage_terms = [expanded_terms(p) for p in passages]
