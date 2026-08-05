"""Offline indexer. Run once (or whenever the corpus changes):

    python -m ingest.build_index --books organon,boericke,allen

Deliberately separate from the serving path. The app should never embed a
book at request time; it should open a prebuilt, versioned index and start
answering. Commit data/chroma/ to the Space (it's small) or rebuild in a
Space startup hook.

Tiers are a `--books` flag, not a code edit: tier 1 (Organon + Boericke +
Allen) indexes and evals in minutes so calibration and bugs surface cheap,
before the other ~90k chunks (Kent is never in this list — task 4 keeps it
out of Chroma entirely and serves it from SQLite instead).
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL
from ingest.build_chunks import BUILDERS, build

# book -> subject filter value. This is what rag/retriever.py's
# `where={"subject": ...}` gate matches on, so "quiz me on Repertory" can
# never return Materia Medica.
BOOK_SUBJECT = {
    "organon": "Organon of Medicine",
    "boericke": "Materia Medica",
    "allen": "Materia Medica",
    "clarke": "Materia Medica",
    "hering": "Materia Medica",
    "nash": "Materia Medica",
}

ALL_BOOKS = list(BOOK_SUBJECT)


def _chunk_id(source: str, locator: str, text: str) -> str:
    # Content-hashed IDs make re-indexing idempotent: unchanged chunks keep
    # their id, so an interrupted build can safely be re-run.
    return hashlib.sha1(f"{source}|{locator}|{text}".encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--books",
        default=",".join(ALL_BOOKS),
        help=f"comma-separated subset of: {','.join(ALL_BOOKS)}",
    )
    args = ap.parse_args()
    books = [b.strip() for b in args.books.split(",") if b.strip()]

    unknown = set(books) - set(BOOK_SUBJECT)
    if unknown:
        print(f"Unknown book(s): {sorted(unknown)}. Known: {sorted(BOOK_SUBJECT)}")
        return 1

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Rebuild from scratch: a stale chunk that no longer exists in the
    # current corpus build is worse than a slow rebuild, because it can
    # still be cited.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    coll = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        # search_ef default (10) makes HNSW skip recall for speed -- fine at
        # web scale, but it means the *same* query can non-deterministically
        # return a different top-k near a boundary, run to run (verified
        # directly: "Nux vomica" found 1 vs 2 non-sparse candidates in the
        # top-30 across repeated calls). At 6,448 vectors the cost of a much
        # higher ef is negligible and buys back-to-back-reproducible results
        # for a confidence gate that's supposed to be a stable threshold.
        metadata={"hnsw:space": "cosine", "hnsw:search_ef": 400},
    )

    chunks_by_book = build(books)

    total = 0
    for book, chunks in chunks_by_book.items():
        subject = BOOK_SUBJECT[book]
        ids, docs, metas = [], [], []
        for c in chunks:
            ids.append(_chunk_id(c["source"], c["locator"], c["text"]))
            docs.append(c["text"])
            meta = {
                "book": c["source"],
                "subject": subject,
                "locator": c["locator"],
                "layer": c["layer"],
                "url": c["url"],
                "sparse": bool(c.get("sparse", False)),
            }
            if c.get("remedy"):
                meta["remedy"] = c["remedy"]
            metas.append(meta)

        # Chroma chokes on very large single adds; batch it.
        for i in range(0, len(ids), 256):
            coll.add(ids=ids[i : i + 256], documents=docs[i : i + 256], metadatas=metas[i : i + 256])

        print(f"  {book}: {len(ids)} chunks")
        total += len(ids)

    print(f"\nIndexed {total} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")
    if total == 0:
        print("No chunks produced — check ingest/build_chunks.py and out/*.json")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
