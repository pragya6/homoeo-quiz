"""Central configuration. Everything tunable lives here, nothing else reads os.environ."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # populates os.environ from .env if present; real env vars (Space secrets, shell exports) still win

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
CHROMA_DIR = ROOT / "data" / "chroma"
REPERTORY_DB = ROOT / "data" / "repertory.db"  # Kent: a database, not a retrieval corpus — see ingest/build_repertory.py
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------- Station 1
# Scale target: 2-10 concurrent students. Sized for the Gemini free tier.
# Flip FREE_TIER_MODE off once billing is enabled and the RPD ceiling lifts.
FREE_TIER_MODE = os.environ.get("FREE_TIER_MODE", "1") == "1"

MODEL_FAST = "gemini-2.5-flash-lite"   # 1,000 req/day free, $0.10/$0.40 paid
MODEL_SMART = "gemini-2.5-flash"       # ~250 req/day free, $0.30/$2.50 paid

# Task -> model. This table IS the cost story: only the two tasks that need
# real reasoning get the expensive model, and on free tier even those fall back.
ROUTES = {
    "mcq_generation": MODEL_FAST,
    "distractor_generation": MODEL_FAST if FREE_TIER_MODE else MODEL_SMART,
    "grounding_verification": MODEL_FAST if FREE_TIER_MODE else MODEL_SMART,
    "coaching": MODEL_FAST,
}

# Per-model request budget. Docs say free tier is 15 RPM; measured against the
# live API (2026-07) Flash-Lite's free_tier_requests quota is actually 10 RPM
# ("quotaValue": "10" on a 429), so we throttle at 8 for real headroom rather
# than the 12 the docs would justify — a 429 costs a retry (and a student's
# patience) more than a 200ms wait.
RATE_LIMITS = {
    MODEL_FAST: {"rpm": 8 if FREE_TIER_MODE else 60},
    MODEL_SMART: {"rpm": 8 if FREE_TIER_MODE else 60},
}

MAX_RETRIES = 6
BACKOFF_BASE_SECONDS = 1.5

# ---------------------------------------------------------------- Station 2
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, runs on CPU, no API quota
COLLECTION_NAME = "homeo_canon"

RETRIEVE_K = 8            # pull 8, use rank-0 as answer source
DISTRACTOR_RANKS = (2, 3, 4, 5)   # "related but not the answer" -> confusable options

# Confidence gate. Below this cosine similarity we refuse instead of bluffing.
# Calibrated 2026-07-24/25 on the tier-1 index (Organon+Boericke+Allen), then
# cross-validated against a full evaluation.run_eval pass (not just retrieval
# similarity): refusal_accuracy 0.6->0.83, false_refusal_rate 0.61->0.30 after
# also excluding sparse single-line fields from rank-0 eligibility (see
# rag/retriever.py). Full before/after table and the one residual gap
# ("monoclonal antibody dosing" still slips through) are in README "Known
# limits". Revisit as the goldset grows past 15 cases.
MIN_RETRIEVAL_SIMILARITY = 0.65

# Second, orthogonal accept path for real topics that score just under
# MIN_RETRIEVAL_SIMILARITY (Nux vomica 0.63, Lycopodium 0.59, Arsenicum
# album 0.61 were all being refused): accept if top-1 clears this *lower*
# floor AND leads the next-best non-sparse candidate by MIN_MARGIN. Does not
# touch MIN_RETRIEVAL_SIMILARITY itself, so it can't reopen a leak that's
# already accepted on the hard floor alone (e.g. "monoclonal antibody
# dosing", top-1 0.66, already above 0.65 regardless of margin).
#
# Calibrated on a zero-cost retrieval-only sweep of the goldset (no LLM
# calls -- see README "Known limits" for the full table):
#   SOFT_SIM_FLOOR doesn't matter in 0.55-0.60 -- every real candidate in
#     that band sits above all three values tested.
#   MIN_MARGIN has a real, narrow sweet spot: 0.015-0.020 rescues Nux vomica
#     (margin 0.024) without wrongly accepting CRISPR (margin 0.0104,
#     the closest refuse-probe). Below 0.015 it accepts CRISPR too; above
#     0.02 it rescues nothing. This is a partial fix, not a full one --
#     Arsenicum album (margin 0.0052) and Lycopodium (no second non-sparse
#     candidate to measure a margin against) aren't rescuable this way.
SOFT_SIM_FLOOR = 0.55
MIN_MARGIN = 0.02

# Deterministic, zero-cost veto checked *before* any retrieval or LLM call
# (rag/retriever.py::retrieve_for_question). The 62-case goldset expansion
# (see evaluation/goldset.jsonl, README "Known limits") surfaced 8 refuse
# probes that cleared MIN_RETRIEVAL_SIMILARITY outright -- e.g. "anatomy of
# the brachial plexus" (0.697), "acupuncture point selection..." (0.713) --
# because the corpus genuinely contains adjacent-sounding vocabulary (remedy
# body-region fields, generic Organon aphorisms). That's a relevance problem,
# not a confidence problem: no threshold fixes it, because the similarity is
# real and the domain is wrong.
#
# Every entry here is a term that is *exclusively* out-of-domain -- never a
# word that appears in normal remedy indications. Deliberately excludes
# generic anatomy/symptom words like "cirrhosis" or "cardiac": Carduus
# marianus is genuinely indicated for liver cirrhosis in Boericke, and
# "cardiac" alone appears throughout legitimate remedy "Heart" sections
# (e.g. "cardiac dropsy"), so blocking either would raise false_refusal_rate
# on real homoeopathy queries. The specific multi-word phrases below
# ("brachial plexus", "conduction system", "monoclonal antibody", ...) don't
# have that problem -- they're precise enough to name the leak without
# catching generic body-part language.
OUT_OF_DOMAIN_TERMS = [
    # other complementary/alternative medicine systems -- not homoeopathy,
    # however adjacent-sounding
    "acupuncture", "meridian theory", "meridian point", "unani", "tibb",
    "naturopathic", "naturopathy", "ayurvedic", "ayurveda", "dosha",
    "chiropractic", "traditional chinese medicine",
    # modern biomedicine / biotech vocabulary this 19th/early-20th-century
    # corpus predates
    "monoclonal antibody", "crispr", "mrna vaccine", "lipid nanoparticle",
    "pcr-based", "polymerase chain reaction", "statin", "chemotherapy",
    "mri scan", "ct scan", "ecg", "ekg", "laparoscopic",
    # clinical/academic terminology, not homoeopathic symptom language
    "histopathology", "brachial plexus", "conduction system",
    "pharmacokinetics", "antibiotic resistance",
    # unrelated technical/legal/finance
    "gdpr", "bitcoin", "gradient descent",
]

# Chunking
MAX_CHUNK_CHARS = 2200
CHUNK_OVERLAP_CHARS = 180

# ---------------------------------------------------------------- Station 5
GENERATION_TEMPERATURE = 0.35   # some variety in phrasing
VERIFY_TEMPERATURE = 0.0        # judging must be deterministic
METRICS_LOG = LOG_DIR / "metrics.jsonl"

# Target share of served questions routed through rag/deterministic.py
# (Kent grade-3 rubric lookups, Hering important/clinical symptoms) instead
# of the generate-then-verify LLM pipeline: zero hallucination risk by
# construction, and it's the only way "Repertory" gets served at all now
# that Kent is deliberately kept out of the vector index (task 4).
DETERMINISTIC_ROUTE_RATE = 0.40

# Subjects the deterministic generator can actually speak to -- it only ever
# produces Kent-rubric or Hering-symptom questions, nothing about Organon
# doctrine or philosophy. This gates the random DETERMINISTIC_ROUTE_RATE
# roll: without it, the roll fired regardless of topic, and since the
# deterministic path never touches retrieval or a confidence gate, it served
# a confident, unrelated answer for out-of-corpus probes too (verified
# directly: "Bitcoin consensus mechanism" served a real Hering fever
# question). "Repertory" still always routes here regardless of the roll --
# there's no other way to serve it.
DETERMINISTIC_ELIGIBLE_SUBJECTS = {"Repertory", "Materia Medica"}

EXAMS = ["UPSC (Homoeopathy)", "Other / General"]

SUBJECTS = [
    "Organon of Medicine",
    "Materia Medica",
    "Repertory",
    "Homoeopathic Philosophy",
]
