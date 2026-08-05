"""Retrieval: the half of anti-hallucination that happens before the model runs.

Three jobs:
  1. Fetch passages, optionally filtered by subject (so "quiz me on Repertory"
     never pulls Materia Medica).
  2. Gate on confidence. If the corpus doesn't really cover the topic, we say so
     rather than letting the model improvise.
  3. Mine *near-miss* passages to seed believable distractors.
"""

from __future__ import annotations

import re

import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DISTRACTOR_RANKS,
    EMBED_MODEL,
    MIN_MARGIN,
    MIN_RETRIEVAL_SIMILARITY,
    OUT_OF_DOMAIN_TERMS,
    RETRIEVE_K,
    SOFT_SIM_FLOOR,
)
from rag.schemas import Chunk

# \b on each side of the whole alternation still anchors correctly per
# multi-word phrase -- it matches against whichever alternative the engine
# picked, not the group as a unit -- so "brachial plexus" and single words
# like "acupuncture" both get proper word-boundary matching, not a substring
# match that could fire inside an unrelated word.
_OUT_OF_DOMAIN_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in OUT_OF_DOMAIN_TERMS) + r")\b", re.IGNORECASE
)

_coll = None


def collection():
    global _coll
    if _coll is None:
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _coll = client.get_collection(COLLECTION_NAME, embedding_function=embedder)
    return _coll


class NotInCorpus(Exception):
    """Raised when retrieval confidence is too low to answer honestly.

    This is a feature. A quiz bot that invents a plausible-sounding aphorism is
    worse than one that admits the book doesn't cover the topic.
    """


def search(query: str, subject: str | None = None, k: int = RETRIEVE_K) -> list[Chunk]:
    where = {"subject": subject} if subject else None
    res = collection().query(query_texts=[query], n_results=k, where=where)

    chunks: list[Chunk] = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        chunks.append(
            Chunk(
                chunk_id=cid,
                text=doc,
                book=meta["book"],
                subject=meta["subject"],
                locator=meta["locator"],
                similarity=1.0 - float(dist),  # cosine distance -> similarity
                sparse=bool(meta.get("sparse", False)),
            )
        )
    return chunks


# DISTRACTOR_RANKS=(2,3,4,5) originally indexed straight into `hits` assuming
# rank-0 was always the answer source: skip rank 1 (too similar to the source
# to make a good distractor), take the next 4. Skipping sparse chunks for the
# *source* means the source isn't always hits[0] any more, so this is now
# expressed relative to whichever hit gets chosen — same skip-1-take-4 shape,
# applied to whatever's left after removing the source.
_NEIGHBOUR_SKIP = min(DISTRACTOR_RANKS) - 1
_NEIGHBOUR_COUNT = len(DISTRACTOR_RANKS)

# Sparse chunks are common enough (~49% of Boericke fields) that RETRIEVE_K=8
# often contains zero non-sparse hits — searching only that shallow would turn
# the sparse-source exclusion into a false-refusal generator. Search deeper
# for a source candidate; neighbours still come from the original RETRIEVE_K
# proximity window so distractor quality (topically *close*) doesn't drift.
_SOURCE_SEARCH_K = max(RETRIEVE_K * 4, 30)


def _non_sparse(hits: list[Chunk]) -> list[Chunk]:
    return [h for h in hits if not h.sparse]


def effective_top(query: str, subject: str | None = None) -> tuple[Chunk | None, list[Chunk]]:
    """The closest *non-sparse* hit, deliberately not gated on
    MIN_RETRIEVAL_SIMILARITY — evaluation/run_eval.py needs the real
    similarity value even when it's below threshold, to calibrate that
    threshold in the first place. Returns (top_or_None, all_hits_searched).

    Single-line fields like Dose/Relationship ("Complementary: Sulph.")
    don't carry enough propositional content to build a defensible question
    on, and letting them win rank-0 is what was driving the generator to
    invent an explanation the passage doesn't actually support — so they're
    skipped here, not just downstream. They're still fair game as
    neighbours: exactly the kind of "related but different" material that
    makes a wrong option tempting rather than obviously absurd.
    """
    hits = search(query, subject=subject, k=_SOURCE_SEARCH_K)
    ns = _non_sparse(hits)
    return (ns[0] if ns else None), hits


def margin_for(query: str, subject: str | None = None) -> tuple[Chunk | None, float | None, list[Chunk]]:
    """(top, margin, hits) — margin is top-1 minus the next-best non-sparse
    candidate's similarity, or None if there isn't a second candidate to
    compare against. See config.SOFT_SIM_FLOOR / MIN_MARGIN for how this is
    used as a second accept path, and README "Known limits" for why it's
    *not* a clean fix for every case (a real remedy can have as thin a
    margin as a genuine leak, at this corpus's embedding resolution)."""
    hits = search(query, subject=subject, k=_SOURCE_SEARCH_K)
    ns = _non_sparse(hits)
    if not ns:
        return None, None, hits
    margin = (ns[0].similarity - ns[1].similarity) if len(ns) > 1 else None
    return ns[0], margin, hits


def out_of_domain_term(query: str) -> str | None:
    """The matched OUT_OF_DOMAIN_TERMS phrase, or None.

    A deterministic veto checked before retrieval: some out-of-corpus queries
    (other CAM systems, anatomy/pathology terminology) cosine-match this
    corpus's vocabulary closely enough to clear MIN_RETRIEVAL_SIMILARITY
    outright -- a real similarity, wrong domain, so no threshold catches it.
    See config.OUT_OF_DOMAIN_TERMS for the list and why each entry is safe.
    """
    m = _OUT_OF_DOMAIN_RE.search(query)
    return m.group(1) if m else None


def retrieve_for_question(query: str, subject: str | None = None) -> tuple[Chunk, list[Chunk]]:
    """Return (answer_source, distractor_sources).

    Checks the out-of-domain veto first -- free, and catches leaks similarity
    alone cannot (see out_of_domain_term). Otherwise accepts on either of two
    paths: top-1 clears MIN_RETRIEVAL_SIMILARITY outright, or it clears the
    lower SOFT_SIM_FLOOR *and* leads the next-best non-sparse candidate by
    MIN_MARGIN. The margin path exists to rescue real topics that score just
    under the hard floor without lowering that floor itself (which would also
    re-admit already-refused leaks).
    """
    term = out_of_domain_term(query)
    if term is not None:
        raise NotInCorpus(f"'{query}' names an out-of-domain term ('{term}') this corpus doesn't cover.")

    top, margin, hits = margin_for(query, subject=subject)
    if not hits:
        raise NotInCorpus(f"No passages indexed for '{query}'.")
    if top is None:
        raise NotInCorpus(f"Only sparse passages found for '{query}'.")

    hard_ok = top.similarity >= MIN_RETRIEVAL_SIMILARITY
    soft_ok = top.similarity >= SOFT_SIM_FLOOR and margin is not None and margin >= MIN_MARGIN
    if not (hard_ok or soft_ok):
        margin_s = f"{margin:.3f}" if margin is not None else "n/a"
        raise NotInCorpus(
            f"Closest passage scored {top.similarity:.2f} (margin {margin_s}), below the "
            f"{MIN_RETRIEVAL_SIMILARITY} confidence floor and the "
            f"{SOFT_SIM_FLOOR}-floor/{MIN_MARGIN}-margin fallback."
        )

    top_idx = next(i for i, h in enumerate(hits) if not h.sparse)  # same selection margin_for made
    # Neighbour candidates: the original close-proximity window, source
    # removed. If the source came from beyond that window (deep search found
    # it, but it wasn't among the closest RETRIEVE_K), the window is already
    # source-free.
    window = hits[:RETRIEVE_K]
    candidates = [h for i, h in enumerate(window) if i != top_idx]
    neighbours = candidates[_NEIGHBOUR_SKIP : _NEIGHBOUR_SKIP + _NEIGHBOUR_COUNT]
    return top, neighbours


def format_sources(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.locator}]\n{c.text}" for c in chunks)
