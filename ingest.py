#!/usr/bin/env python3
"""
Ingestion locale des fiches lutteurs vers Pinecone + index des noms (hybride).

Usage:
  python ingest.py --file all_athletes.jsonl
  python ingest.py --file all_athletes.jsonl --limit 50
  python ingest.py --file all_athletes.jsonl --recreate
  python ingest.py --file all_athletes.jsonl --names-only   # régénère names_index.json sans ré-embedder
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "sl-rag-athletes")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
BATCH_SIZE = 64
NAMES_INDEX_FILE = Path(__file__).resolve().parent / "names_index.json"


def die(msg: str) -> None:
    print(f"ERREUR: {msg}", file=sys.stderr)
    sys.exit(1)


def normalize_name(name: str) -> str:
    """Normalise un nom pour matching (minuscules, sans accents, espaces simples)."""
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
            if limit and len(docs) >= limit:
                break
    return docs


def build_names_index(docs: list[dict]) -> dict:
    """
    Mappe nom normalisé -> liste d'entrées {id, nom, drupal_id}.
    Inclut aussi nom_technique.
    """
    index: dict[str, list[dict]] = {}
    for doc in docs:
        meta = doc.get("metadata") or {}
        doc_id = str(doc.get("id") or f"athlete_{meta.get('drupal_id', '')}")
        entries = []
        for key in ("nom", "nom_technique"):
            raw = str(meta.get(key) or "").strip()
            if not raw:
                continue
            norm = normalize_name(raw)
            if len(norm) < 2:
                continue
            entries.append(
                {
                    "id": doc_id,
                    "nom": str(meta.get("nom") or raw),
                    "drupal_id": int(meta.get("drupal_id") or 0),
                    "match_on": key,
                    "normalized": norm,
                }
            )
        for e in entries:
            index.setdefault(e["normalized"], []).append(
                {"id": e["id"], "nom": e["nom"], "drupal_id": e["drupal_id"]}
            )
    return index


def save_names_index(names_index: dict) -> None:
    # Dédupliquer les listes par id
    clean = {}
    for norm, items in names_index.items():
        seen = set()
        uniq = []
        for it in items:
            if it["id"] in seen:
                continue
            seen.add(it["id"])
            uniq.append(it)
        clean[norm] = uniq
    NAMES_INDEX_FILE.write_text(
        json.dumps(clean, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )
    print(f"Index des noms écrit : {NAMES_INDEX_FILE} ({len(clean)} clés)")


def ensure_index(pc, dim: int, recreate: bool = False):
    from pinecone import ServerlessSpec

    existing = {idx.name for idx in pc.list_indexes()}
    if recreate and PINECONE_INDEX in existing:
        print(f"Suppression de l'index existant '{PINECONE_INDEX}'…")
        pc.delete_index(PINECONE_INDEX)
        existing.discard(PINECONE_INDEX)
        time.sleep(2)

    if PINECONE_INDEX not in existing:
        print(f"Création de l'index '{PINECONE_INDEX}' (dim={dim})…")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while True:
            desc = pc.describe_index(PINECONE_INDEX)
            if getattr(desc.status, "ready", False) or (
                isinstance(desc.status, dict) and desc.status.get("ready")
            ):
                break
            print("  Index en cours de création…")
            time.sleep(2)
        print("Index prêt.")
    else:
        print(f"Index '{PINECONE_INDEX}' déjà existant.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion JSONL → Pinecone + names_index")
    parser.add_argument("--file", "-f", required=True, help="Fichier JSONL exporté par Drupal")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de documents")
    parser.add_argument("--recreate", action="store_true", help="Recréer l'index Pinecone")
    parser.add_argument("--namespace", default="", help="Namespace Pinecone")
    parser.add_argument(
        "--names-only",
        action="store_true",
        help="Ne régénère que names_index.json (pas de ré-embedding)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        die(f"Fichier introuvable: {path}")

    print(f"Chargement de {path}…")
    docs = load_jsonl(path, args.limit)
    print(f"  → {len(docs)} document(s)")
    if not docs:
        die("Aucun document.")

    names_index = build_names_index(docs)
    save_names_index(names_index)

    if args.names_only:
        print("Mode --names-only : terminé (Pinecone non modifié).")
        return

    if not PINECONE_API_KEY:
        die("PINECONE_API_KEY manquant dans .env")

    print(f"Chargement du modèle d'embeddings '{EMBEDDING_MODEL}'…")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"  → dimension = {dim}")

    from pinecone import Pinecone

    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, dim, recreate=args.recreate)
    index = pc.Index(PINECONE_INDEX)

    texts = [d.get("text") or "" for d in docs]
    ids = [str(d.get("id") or d.get("metadata", {}).get("drupal_id") or i) for i, d in enumerate(docs)]

    print(f"Vectorisation de {len(texts)} textes (batch={BATCH_SIZE})…")
    vectors = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    )

    print("Envoi vers Pinecone…")
    upserts = []
    for i, (doc_id, vec, doc) in enumerate(zip(ids, vectors, docs)):
        meta = doc.get("metadata") or {}
        clean_meta = {
            "drupal_id": int(meta["drupal_id"]) if meta.get("drupal_id") is not None else 0,
            "nom": str(meta.get("nom") or "")[:500],
            "ecurie": str(meta.get("ecurie") or "")[:200],
            "victoires": int(meta.get("victoires") or 0),
            "defaites": int(meta.get("defaites") or 0),
            "total_combats": int(meta.get("total_combats") or 0),
            "is_king": bool(meta.get("is_king") or False),
            "text_preview": (doc.get("text") or "")[:1000],
        }
        full_text = doc.get("text") or ""
        if len(full_text) < 35000:
            clean_meta["text"] = full_text
        upserts.append({"id": doc_id, "values": vec.tolist(), "metadata": clean_meta})

        if len(upserts) >= BATCH_SIZE:
            index.upsert(vectors=upserts, namespace=args.namespace)
            print(f"  upsert {i + 1}/{len(docs)}")
            upserts = []

    if upserts:
        index.upsert(vectors=upserts, namespace=args.namespace)
        print(f"  upsert {len(docs)}/{len(docs)}")

    stats = index.describe_index_stats()
    print("Terminé.")
    print(f"Stats index: {stats}")


if __name__ == "__main__":
    main()
