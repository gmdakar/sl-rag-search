#!/usr/bin/env python3
"""
API de recherche hybride générique :
  - matching par nom (exact, tokens, flou) — sans liste d'alias codés en dur
  - recherche vectorielle Pinecone

  uvicorn search_api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "sl-rag-athletes")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
API_SECRET = os.getenv("API_SECRET", "")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
NAMES_INDEX_FILE = Path(__file__).resolve().parent / "names_index.json"

FUZZY_RATIO_MIN = float(os.getenv("NAME_FUZZY_RATIO", "0.78"))
TOKEN_MIN_LEN = int(os.getenv("NAME_TOKEN_MIN_LEN", "3"))

app = FastAPI(title="SL RAG Hybrid Search", version="1.3.0-fastembed")

_model = None
_index = None
_names_index: dict[str, list[dict]] = {}
_names_flat: list[tuple[str, dict]] = []


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def significant_tokens(norm: str) -> list[str]:
    """Tokens utiles : longueur >= TOKEN_MIN_LEN, ou chiffres (ex. '2' dans Boy Niang 2)."""
    out = []
    for t in norm.split():
        if t.isdigit() or len(t) >= TOKEN_MIN_LEN:
            out.append(t)
    return out


def name_digits(norm: str) -> set[str]:
    return {t for t in norm.split() if t.isdigit()}


def edit_distance(a: str, b: str) -> int:
    """Distance de Levenshtein (petites chaînes : noms de lutteurs)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def load_names_index() -> dict[str, list[dict]]:
    if not NAMES_INDEX_FILE.exists():
        return {}
    try:
        return json.loads(NAMES_INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_model():
    """Charge fastembed (léger, sans torch CUDA) — même modèle 384-d que l'ingestion."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        # Même famille que sentence-transformers paraphrase-multilingual-MiniLM-L12-v2
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
        print(f"fastembed chargé : {EMBEDDING_MODEL}")
    return _model


def embed_query(query: str) -> list[float]:
    model = get_model()
    vec = next(model.embed([query]))
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return [float(x) for x in vec]


def get_index():
    global _index
    if _index is None:
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY manquant")
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = pc.Index(PINECONE_INDEX)
    return _index


@app.on_event("startup")
def on_startup():
    global _names_index, _names_flat
    _names_index = load_names_index()
    flat: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for norm, items in _names_index.items():
        for it in items:
            key = f"{it.get('id')}|{norm}"
            if key in seen:
                continue
            seen.add(key)
            flat.append((norm, it))
    _names_flat = sorted(flat, key=lambda x: len(x[0]), reverse=True)
    print(
        f"names_index chargé : {len(_names_index)} clés, "
        f"{len(_names_flat)} entrées depuis {NAMES_INDEX_FILE}"
    )


def check_secret(authorization: Optional[str]) -> None:
    if not API_SECRET:
        return
    if not authorization or authorization != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _add_match(results, matched_ids, item, score, how):
    iid = item.get("id")
    if not iid or iid in matched_ids:
        return
    matched_ids.add(iid)
    results.append({
        "id": iid,
        "nom": item.get("nom"),
        "drupal_id": item.get("drupal_id"),
        "name_score": round(score, 4),
        "match_how": how,
    })


def find_name_matches(query: str) -> list[dict]:
    """Matching générique strict :
    - exact / substring / tokens d'abord
    - les chiffres du nom (ex. 2) doivent être dans la question
    - fuzzy seulement si assez de tokens en commun
    """
    if not _names_flat:
        return []

    q = normalize_name(query)
    if not q:
        return []
    q_compact = q.replace(" ", "")
    q_tokens = set(significant_tokens(q))
    q_tok_list = significant_tokens(q)
    q_digits = name_digits(q)

    results: list[dict] = []
    matched_ids: set[str] = set()

    def digits_ok(norm: str) -> bool:
        """Si le nom porte un numéro (2, 3…), la question doit le contenir."""
        nd = name_digits(norm)
        if not nd:
            return True
        return nd.issubset(q_digits) or all(d in q for d in nd)

    # --- Pass 1 : exact, substring, tokens (haute confiance) ---
    for norm, item in _names_flat:
        if len(norm) < 2:
            continue
        if not digits_ok(norm):
            continue

        if re.search(rf"(?:^|\s){re.escape(norm)}(?:\s|$)", q):
            _add_match(results, matched_ids, item, 1.0, "exact")
            continue

        tokens = significant_tokens(norm)
        if len(tokens) >= 2 and norm in q:
            _add_match(results, matched_ids, item, 0.98, "substring")
            continue

        # Tous les tokens du nom dans la question (y compris chiffres)
        if len(tokens) >= 2 and all(t in q_tokens for t in tokens):
            _add_match(results, matched_ids, item, 0.95, "tokens")
            continue

        # Tokens avec 1 faute max (hors chiffres exacts)
        if len(tokens) >= 2 and len(tokens) <= 5:
            if len(q_tok_list) >= len(tokens):
                used = [False] * len(q_tok_list)
                ok = True
                for nt in tokens:
                    found = False
                    for qi, qt in enumerate(q_tok_list):
                        if used[qi]:
                            continue
                        if nt.isdigit() or qt.isdigit():
                            if nt == qt:
                                used[qi] = True
                                found = True
                                break
                            continue
                        if abs(len(nt) - len(qt)) > 1:
                            continue
                        if edit_distance(nt, qt) <= 1:
                            used[qi] = True
                            found = True
                            break
                    if not found:
                        ok = False
                        break
                if ok:
                    _add_match(results, matched_ids, item, 0.94, "tokens_edit1")
                    continue

    high_conf = [r for r in results if r["name_score"] >= 0.94]
    # Tokens déjà « couverts » par un match exact (évite Ama Bou Malika si General Malika match)
    covered_tokens: set[str] = set()
    for r in high_conf:
        nom = normalize_name(str(r.get("nom") or ""))
        covered_tokens.update(significant_tokens(nom))

    # --- Pass 2 : fuzzy / edit1 plus strict ---
    for norm, item in _names_flat:
        iid = item.get("id")
        if not iid or iid in matched_ids:
            continue
        if len(norm) < 4:
            continue
        if not digits_ok(norm):
            continue

        name_toks = significant_tokens(norm)
        if not name_toks:
            continue

        # Au moins la moitié des tokens du nom doivent matcher la question
        overlap = 0
        for nt in name_toks:
            if nt in q_tokens:
                overlap += 1
                continue
            if nt.isdigit():
                continue
            for qt in q_tok_list:
                if qt.isdigit():
                    continue
                if abs(len(nt) - len(qt)) <= 1 and edit_distance(nt, qt) <= 1:
                    overlap += 1
                    break
        if overlap < max(1, (len(name_toks) + 1) // 2):
            continue

        # Si un match exact a déjà pris tous les tokens discriminants, skip
        if covered_tokens and name_toks:
            # ex. "malika" seul déjà couvert par "general malika"
            non_generic = [t for t in name_toks if t not in {"boy", "fils", "bou", "de", "le", "la"}]
            if non_generic and all(t in covered_tokens for t in non_generic):
                continue

        # Fuzzy SequenceMatcher sur fenêtres
        q_words = q.split()
        windows = []
        for n in range(1, min(5, len(q_words) + 1)):
            for i in range(len(q_words) - n + 1):
                windows.append(" ".join(q_words[i : i + n]))
        best = 0.0
        for w in windows:
            if abs(len(w) - len(norm)) > max(3, len(norm) // 2):
                continue
            r = SequenceMatcher(None, w, norm).ratio()
            if r > best:
                best = r
        # Seuil plus strict pour noms multi-mots
        min_ratio = max(FUZZY_RATIO_MIN, 0.85 if len(name_toks) >= 2 else FUZZY_RATIO_MIN)
        if best >= min_ratio:
            _add_match(results, matched_ids, item, best, "fuzzy")

    results.sort(key=lambda x: (-x["name_score"], -len(str(x.get("nom") or ""))))
    return results


def fetch_by_ids(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    index = get_index()
    ns = NAMESPACE or None
    fetched = index.fetch(ids=ids, namespace=ns)
    vectors = getattr(fetched, "vectors", None) or fetched.get("vectors") or {}
    hits = []
    for vid, data in vectors.items():
        meta = dict(getattr(data, "metadata", None) or data.get("metadata") or {})
        text = str(meta.pop("text", "") or meta.get("text_preview", "") or "")
        hits.append({
            "id": str(vid),
            "score": 1.0,
            "metadata": meta,
            "text": text,
            "match_type": "name",
        })
    return hits


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)
    filters: Optional[dict[str, Any]] = None
    namespace: Optional[str] = None


class SearchHit(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any]
    text: str = ""
    match_type: str = "vector"
    match_how: str = ""


class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: list[SearchHit]
    name_matches: int = 0
    vector_matches: int = 0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": EMBEDDING_MODEL,
        "index": PINECONE_INDEX,
        "names_index_size": len(_names_index),
        "names_flat_size": len(_names_flat),
        "fuzzy_ratio_min": FUZZY_RATIO_MIN,
        "backend": "fastembed",
    }


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, authorization: Optional[str] = Header(default=None)):
    check_secret(authorization)

    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query vide")

    name_items = find_name_matches(query)
    name_hits_raw = fetch_by_ids([x["id"] for x in name_items])
    how_by_id = {x["id"]: x for x in name_items}
    name_hits = []
    for h in name_hits_raw:
        info = how_by_id.get(h["id"], {})
        h["score"] = float(info.get("name_score") or 1.0)
        h["match_how"] = str(info.get("match_how") or "name")
        name_hits.append(h)
    seen_ids = {h["id"] for h in name_hits}

    index = get_index()
    vector = embed_query(query)
    ns = body.namespace if body.namespace is not None else NAMESPACE

    result = index.query(
        vector=vector,
        top_k=body.top_k,
        include_metadata=True,
        namespace=ns or None,
        filter=body.filters or None,
    )

    vector_hits = []
    matches = getattr(result, "matches", None) or result.get("matches", [])
    for m in matches:
        mid = str(getattr(m, "id", None) or m.get("id"))
        if mid in seen_ids:
            continue
        score = float(getattr(m, "score", None) or m.get("score") or 0)
        meta = dict(getattr(m, "metadata", None) or m.get("metadata") or {})
        text = str(meta.pop("text", "") or meta.get("text_preview", "") or "")
        vector_hits.append({
            "id": mid,
            "score": score,
            "metadata": meta,
            "text": text,
            "match_type": "vector",
            "match_how": "vector",
        })
        seen_ids.add(mid)

    merged = (name_hits + vector_hits)[: body.top_k]
    hits = [
        SearchHit(
            id=h["id"],
            score=h["score"],
            metadata=h["metadata"],
            text=h["text"],
            match_type=h.get("match_type", "vector"),
            match_how=h.get("match_how", ""),
        )
        for h in merged
    ]

    return SearchResponse(
        query=query,
        top_k=body.top_k,
        hits=hits,
        name_matches=len(name_hits),
        vector_matches=len(vector_hits),
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    uvicorn.run("search_api:app", host=host, port=port, reload=False)
