#!/usr/bin/env python3
"""
extract_homeoint.py — Extract Boericke, Clarke, and Nash from homeoint.org (Médi-T)
into structured JSON per remedy, for a RAG quiz-bot.

WHY THIS over the Archive OCR: homeoint serves clean HTML (not OCR), so it kills the
Boericke "BAD" grade at the source. We crawl politely, decode cp1252 (fixes the �),
strip the FrontPage/Médi-T boilerplate, normalize text, and segment into fields.

POLITE BY DESIGN: ~1 req/sec, and every page is cached to cache/ on first fetch so
re-runs hit disk, not the volunteer-run server. Run on YOUR machine.

    pip install requests beautifulsoup4 charset-normalizer
    python3 extract_homeoint.py --book boericke     # or clarke / nash / all
    python3 extract_homeoint.py --book all
    python3 extract_homeoint.py --book boericke --limit 5   # smoke test, 5 pages

Output: out/<book>.json  — list of {remedy, source, keynotes[], modalities{},
relations{}, fields{}, raw_text, url}.

NOTE: An LLM cleanup pass is deliberately NOT included. Per the plan, only run one
later, field-by-field, on residual gaps — never as a bulk "clean this book" job, or
you risk hallucinated symptoms, which is a correctness disaster in an exam bot.
"""

from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from urllib.parse import urljoin

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("pip install requests beautifulsoup4 charset-normalizer")

CACHE = Path("cache"); OUT = Path("out")
RATE_SECONDS = 1.0
LETTERS = "abcdefghijklmnopqrstuvwxyz"

FAILED: set[str] = set()          # URLs that failed to fetch this run (see fetch())
session = requests.Session()      # reused connection + headers for polite crawling
session.headers.update({"User-Agent": "extract_homeoint.py (personal RAG project; contact via homeoint.org courtesy)"})

# Boilerplate phrases/link-texts that repeat on every Médi-T page.
BOILER = re.compile(
    r"presented by m[eé]di-?t|copyright|m[eé]di-?t \d{4}|buy a copy|advertising|"
    r"^\s*main\s*$|^\s*home\s*$|^h\.?i\.?$|hom.opathic materia medica|"
    r"a dictionary of practical materia medica|leaders in homoeopathic therapeutics|"
    r"keynotes?( and characteristics)?( by)? h\.?\s*c\.? allen|"
    r"^\s*keynotes\s*$|by\s+h\.?\s*c\.?\s+allen|"
    r"the guiding symptoms|constantine hering|b\.?\s*jain|homoeopathe international|"
    r"^\s*\*+\s*[A-Z]\s*\*|catalogue|>{3,}|<{3,}|"
    r"translated by (dudgeon|boericke)|^\s*organon of medicine\s*$|"
    r"pr[eé]sent[eé] par|robert s[eé]ror|sylvain cazalet|num[eé]risation|"
    r"mise en page|lectures on homoeopathic philosophy|james tyler kent|"
    r"dr samuel hahnemann|^\s*main\s*$|^\s*lecture \d+\s*$",
    re.I)

# Known OCR/spelling normalizations (extend as you spot more).
TYPO_MAP = {
    "\u0153": "oe", "\u00e6": "ae", "\u0152": "OE", "\u00c6": "AE",  # œ æ ligatures
    "Hom\u0153opath": "Homoeopath", "hom\u0153opath": "homoeopath",
}


# --- fetch (cp1252-correct, cached, rate-limited) ---------------------------

def _alt_host(url: str) -> str:
    """homeoint answers on BOTH www.homeoint.org and homeoint.org, and one host
    sometimes 403s while the other serves fine. Swap as a last resort."""
    if "://www.homeoint.org" in url:
        return url.replace("://www.homeoint.org", "://homeoint.org")
    if "://homeoint.org" in url:
        return url.replace("://homeoint.org", "://www.homeoint.org")
    return url


def fetch(url: str, book: str, attempts: int = 5) -> str:
    """Cached, rate-limited GET with backoff, Referer, and host fallback.

    homeoint intermittently 403s under sustained crawling (it cost 4 Organon
    pages = 79 aphorisms on the first full run). Failures are NEVER cached, so
    re-running a book re-fetches only what failed.
    """
    key = CACHE / book / (re.sub(r"[^\w.-]", "_", url.split("://", 1)[-1]) + ".html")
    if key.exists():
        return key.read_text(encoding="utf-8", errors="replace")

    base = url.rsplit("/", 1)[0] + "/"
    last = ""
    for i in range(attempts):
        time.sleep(RATE_SECONDS if i == 0 else RATE_SECONDS * (3 ** i))  # 1,3,9,27,81s
        target = _alt_host(url) if i >= attempts - 2 else url   # last 2 tries: other host
        try:
            r = session.get(target, timeout=60, headers={"Referer": base})
            if r.status_code in (403, 429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                continue
            r.raise_for_status()
            try:
                html = r.content.decode("cp1252")
            except UnicodeDecodeError:
                html = r.content.decode(r.apparent_encoding or "utf-8", errors="replace")
            key.parent.mkdir(parents=True, exist_ok=True)
            key.write_text(html, encoding="utf-8")
            FAILED.discard(url)
            return html
        except requests.RequestException as e:
            last = str(e)
    FAILED.add(url)
    raise RuntimeError(f"{last} after {attempts} attempts")


# HTML entities like &#150; mean codepoint U+0096, which is a C1 CONTROL char,
# not an en-dash. Browsers silently remap them per the HTML5 spec; BeautifulSoup
# does not. This is why Clarke's "Characteristics. -" headings never matched any
# dash regex while printing correctly in a terminal.
C1_MAP = {
    "\u0091": "'", "\u0092": "'", "\u0093": '"', "\u0094": '"',
    "\u0095": "*", "\u0096": "\u2013", "\u0097": "\u2014",
    "\u0085": "...", "\u0082": ",", "\u0084": '"', "\u0099": "(TM)",
    "\u008b": "<", "\u009b": ">", "\u0086": "+", "\u0087": "+",
}


def normalize(text: str) -> str:
    for a, b in C1_MAP.items():
        text = text.replace(a, b)
    for a, b in TYPO_MAP.items():
        text = text.replace(a, b)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)     # "Acon ." -> "Acon."
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_boiler(line: str) -> bool:
    return bool(BOILER.search(line.strip()))


def italics(soup) -> list[str]:
    """Multi-word italic runs = Boericke/Clarke keynote (characteristic) symptoms.
    Single Titlecase tokens are remedy cross-refs, not keynotes — dropped here."""
    out = []
    for it in soup.find_all(["i", "em"]):
        t = normalize(it.get_text(" ", strip=True))
        if len(t.split()) >= 2 and not re.fullmatch(r"([A-Z][a-z-]+\.?\s*){1,3}", t):
            out.append(t)
    # dedupe, keep order
    seen, uniq = set(), []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower()); uniq.append(t)
    return uniq


# --- link discovery ---------------------------------------------------------

def discover_links(base: str, index_pages: list[str], book: str, pattern: re.Pattern) -> list[str]:
    urls: list[str] = []
    seen = set()
    for page in index_pages:
        try:
            soup = BeautifulSoup(fetch(urljoin(base, page), book), "html.parser")
        except Exception as e:
            print(f"  ! index {page}: {e}"); continue
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0]          # strip #fragment BEFORE matching
            if not href:
                continue
            full = urljoin(base, href)
            if pattern.search(href) and full not in seen:
                seen.add(full); urls.append(full)
    return urls


# --- per-book parsers -------------------------------------------------------

HEADING = re.compile(r"^(.{1,25}?)\.--\s*$")            # Boericke: "Mind.--"

# Clarke: bolded modality marks / stray punctuation that are NOT headings.
CLARKE_NOISE = re.compile(r"[\s<>_.,;:*\u2013\u2014\u2020\u2021+=|/\\-]{1,3}")
# Real Clarke heading, allowing an optional schema number and a TRAILING dash:
#   "Characteristics. -"   "Clinical."   "1 Mind."   "Relations. -"
# A Clarke heading occupies its own line and ALWAYS ends "<words>. <dash>":
#   "Characteristics. -"  "Relations. -"  "SYMPTOMS. 1. Mind. -"  "14. Urinary Organs. -"
DASH = "\u2010\u2011\u2012\u2013\u2014\u2015\u0096\u0097\u2212-"
CLARKE_LINE_HEAD = re.compile(
    r"(?m)^[ \t]*((?:SYMPTOMS\.[ \t]*)?(?:\d{1,2}\.[ \t]*)?"
    r"[A-Z][A-Za-z][A-Za-z ,'&()/-]{1,38}\.)[ \t]*[" + DASH + r"][ \t]*$")
# Fallback: the same headings WITHOUT a trailing dash, used only if the dash
# form yields nothing, so a markup change can never zero the whole book again.
CLARKE_LINE_HEAD_NODASH = re.compile(
    r"(?m)^[ \t]*((?:SYMPTOMS\.[ \t]*)?(?:\d{1,2}\.[ \t]*)?"
    r"(?:Clinical|Characteristics?|Relations?|Causation|Mind|Head|Eyes|Ears|Nose|"
    r"Face|Teeth|Mouth|Throat|Appetite|Stomach|Abdomen|Stool and Anus|"
    r"Urinary Organs|Skin|Sleep|Fever|Generalities)\.)[ \t]*$")


def parse_boericke(soup, url):
    title = normalize(soup.title.get_text()) if soup.title else ""
    latin = title.split(" - ")[0].strip()
    keynotes = italics(soup)
    # Turn <b>Field.--</b> headings into split markers, then segment.
    for b in soup.find_all(["b", "strong"]):
        m = HEADING.match(b.get_text(" ", strip=True))
        if m:
            b.replace_with(f"\x00{m.group(1).strip()}\x00")
    body = normalize(soup.get_text("\n"))
    body = "\n".join(l for l in body.splitlines() if not is_boiler(l))
    parts = body.split("\x00")
    intro = normalize(parts[0])
    fields = {}
    for i in range(1, len(parts) - 1, 2):
        fields[parts[i].strip()] = re.sub(r"\s+", " ", normalize(parts[i + 1]))
    rel = fields.get("Relationship", "") or fields.get("Relations", "")
    mod = _split_modalities(fields.get("Modalities", ""))
    return [{
        "remedy": latin, "source": "Boericke, Pocket Manual (1922)", "url": url,
        "intro": intro, "keynotes": keynotes, "fields": fields,
        "modalities": mod,
        "relations": {"raw": rel, "remedies": _remedy_tokens(rel)},
        "raw_text": normalize(re.sub(r"\x00", " ", body)),
    }]


def _merge_punct_lines(text: str) -> str:
    """Fold lines that are only punctuation (".", "-", en/em dash) into the line
    above, undoing node-level line splitting."""
    out: list[str] = []
    for l in text.splitlines():
        s = l.strip()
        if out and s and re.fullmatch(r"[.:" + DASH + r"]{1,3}", s):
            out[-1] = out[-1].rstrip() + " " + s
        else:
            out.append(l)
    return "\n".join(out)


def parse_clarke(soup, url):
    r"""Clarke: split on the heading pattern in FLATTENED TEXT, not on tags.

    Verified via --diagnose on s/sul.htm: the page is <font>/<br> soup with 915
    <b> tags, 761 of them very short and 121 pure symbols ("<" = worse, ">" =
    better). Tag-based heading detection kept failing on it. But every real
    heading ends with a period + en-dash, on its own line:

        Clinical. -       Characteristics. -      Relations. -
        Causation. -      SYMPTOMS. 1. Mind. -    2. Head. -   14. Urinary Organs. -

    So we flatten to text and split on that, which is immune to the tag layout.
    """
    title = normalize(soup.title.get_text()) if soup.title else ""
    latin = re.split(r" - |\.", title)[0].strip()

    # Clinical list = the italics (disease names), captured BEFORE flattening.
    clinical = []
    for i in soup.find_all(["i", "em"]):
        c = normalize(i.get_text(" ", strip=True))
        if 3 < len(c) < 80 and not CLARKE_NOISE.fullmatch(c or " "):
            clinical.append(c)

    text = normalize(soup.get_text("\n"))
    text = "\n".join(l for l in text.splitlines() if not is_boiler(l))
    # CRITICAL: get_text("\n") puts each text NODE on its own line, so a heading
    # rendered as <b>Characteristics. <font>-</font></b> arrives as TWO lines:
    #   "Characteristics."  /  "-"
    # (--diagnose used get_text(" ") and showed it as one string, which hid this.)
    # Merge punctuation-only lines back onto the previous line before splitting.
    text = _merge_punct_lines(text)

    parts = CLARKE_LINE_HEAD.split(text)
    if len(parts) < 3:                       # dash form found nothing -> fallback
        parts = CLARKE_LINE_HEAD_NODASH.split(text)
    fields = {}
    for i in range(1, len(parts) - 1, 2):
        k = re.sub(r"^SYMPTOMS\.\s*", "", parts[i].strip()).rstrip(". ")
        v = re.sub(r"\s+", " ", parts[i + 1]).strip()
        if k and v:
            fields[k] = (fields.get(k, "") + " " + v).strip()

    rel = next((v for k, v in fields.items() if k.lower().startswith("relation")), "")
    mod = {}
    for k, v in fields.items():
        kl = k.lower()
        if kl.startswith(("modalit", "aggrav")):
            mod.setdefault("worse", v)
        elif kl.startswith("amelior"):
            mod.setdefault("better", v)
    return [{
        "remedy": latin, "source": "Clarke, Dictionary of Practical Materia Medica",
        "url": url, "fields": fields,
        "clinical": clinical[:200],
        "keynotes": [],              # Clarke italics are clinical terms, not keynotes
        "modalities": mod,
        "relations": {"raw": rel, "remedies": _remedy_tokens(rel)},
        "raw_text": re.sub(r"\s+", " ", text).strip(),
    }]


def parse_nash(soup, url):
    """Nash chapter pages hold several remedies.

    Verified via --diagnose on mmh05.htm: there ARE <a name="BELLADONNA"> anchors
    (4 on that page) and bold ALL-CAPS headings ("BELLADONNA.", "HYOSCYAMUS
    NIGER."). The 213 <i> tags are remedy names Nash compares against - harvested
    as `remedy_mentions`, which is why Nash's low `relations` count is expected:
    it is prose therapeutics, not a structured materia medica.
    """
    italics = [normalize(i.get_text(" ", strip=True)) for i in soup.find_all(["i", "em"])]
    italics = [t for t in italics if 2 < len(t) < 90]

    text = normalize("\n".join(l for l in soup.get_text("\n").splitlines()
                               if not is_boiler(l)))
    recs = _nash_split(text, url, soup)
    # attach remedy mentions (whole-page granularity; good enough for retrieval)
    seen, mentions = set(), []
    for t in italics:
        for tok in re.split(r"[,;]", t):
            tok = tok.strip(" .")
            if tok and tok.lower() not in seen and re.match(r"^[A-Z][A-Za-z.\- ]{2,}$", tok):
                seen.add(tok.lower()); mentions.append(tok)
    for r in recs:
        r["remedy_mentions"] = mentions[:60]
    return recs


def _nash_split(text, url, soup):
    lines = text.splitlines()
    heads = [i for i, l in enumerate(lines)
             if re.fullmatch(r"[A-Z][A-Z \.\-]{3,40}", l.strip())]
    recs = []
    for j, h in enumerate(heads):
        name = lines[h].strip().rstrip(".").title()
        end = heads[j + 1] if j + 1 < len(heads) else len(lines)
        blob = normalize("\n".join(lines[h + 1:end]))
        if len(blob.split()) < 10:
            continue
        rel = " ".join(re.findall(
            r"(?:compare|complementary|antidote[sd]?|followed by|similar to)\b[^.]{0,200}\.",
            blob, re.I))
        recs.append({"remedy": name,
                     "source": "Nash, Leaders in Homoeopathic Therapeutics",
                     "url": url, "keynotes": [],
                     "relations": {"raw": rel, "remedies": _remedy_tokens(rel)},
                     "raw_text": blob})
    return recs


def _split_modalities(text: str) -> dict:
    if not text:
        return {}
    text = re.sub(r"\s+", " ", text)
    better = re.search(r"better([^.;]*[.;])", text, re.I)
    worse = re.search(r"worse(.*)$", text, re.I)
    return {k: normalize(v) for k, v in
            {"better": better.group(1) if better else "",
             "worse": worse.group(1) if worse else ""}.items() if v}


def _remedy_tokens(text: str) -> list[str]:
    toks = re.findall(r"\b[A-Z][a-z]{2,}\b\.?", text)
    seen, out = set(), []
    for t in toks:
        t = t.rstrip(".")
        if t.lower() not in seen and t.lower() not in {
                "compare", "complementary", "not", "after", "before", "similar",
                "compatible", "incompatible", "like", "followed", "follows",
                "antidotes", "affects", "inimical", "complements", "the", "similarto"}:
            seen.add(t.lower()); out.append(t)
    return out


# --- Allen (chapter files + #anchors, Boericke-style sub-sections) -----------

ALLEN_SECT = re.compile(r"^(Relationship|Relations|Aggravation|Amelioration|Causation)"
                        r"\s*[:.\-]?\s*$", re.I)
# "Abrotanum. Southernwood. (Compositae.)" / "Aconitum Napellus. Monkshood."
ALLEN_HEAD = re.compile(r"^[A-Z][A-Za-z\-]+(\s+[A-Za-z\-]+){0,3}\.\s*\S.{0,90}$")


def parse_allen(soup, url):
    # Mark the section sub-headings, then mark each remedy name via its anchor.
    for b in soup.find_all(["b", "strong"]):
        if ALLEN_SECT.match(b.get_text(" ", strip=True)):
            b.replace_with(f"\x01{re.sub(r'[:.\\-]+$','',b.get_text(' ',strip=True)).strip()}\x01")
    # Strategy A: <a name="Acon"> anchors, marking the bold that follows.
    marked = set()
    for a in soup.find_all("a", attrs={"name": True}):
        nb = a.find_next(["b", "strong"])
        if nb and "\x01" not in nb.get_text() and id(nb) not in marked:
            marked.add(id(nb))
            nb.replace_with(f"\x02{nb.get_text(' ', strip=True)}\x02")
    # Strategy B (fallback / supplement): bold blocks that LOOK like a remedy
    # heading -- "Abrotanum. Southernwood. (Compositae.)". Without this, chapter
    # files with no anchors collapse into ONE record per chapter.
    for b in soup.find_all(["b", "strong"]):
        if id(b) in marked:
            continue
        t = b.get_text(" ", strip=True)
        if ALLEN_HEAD.match(t) and not ALLEN_SECT.match(t) and not is_boiler(t):
            marked.add(id(b))
            b.replace_with(f"\x02{t}\x02")
    text = "\n".join(l for l in normalize(soup.get_text("\n")).splitlines()
                     if not is_boiler(l))
    chunks = text.split("\x02")
    recs = []
    for i in range(1, len(chunks) - 1, 2):
        name, body = chunks[i].strip(), chunks[i + 1]
        latin = re.split(r"\.\s", name, maxsplit=1)[0].strip()
        parts = body.split("\x01")
        keynotes_text = re.sub(r"\s+", " ", normalize(parts[0]))
        fields = {}
        for j in range(1, len(parts) - 1, 2):
            fields[parts[j].strip().title()] = re.sub(r"\s+", " ", normalize(parts[j + 1]))
        mod = {}
        if fields.get("Aggravation"):  mod["worse"] = fields["Aggravation"]
        if fields.get("Amelioration"): mod["better"] = fields["Amelioration"]
        rel = fields.get("Relationship") or fields.get("Relations", "")
        keynotes = [s.strip() for s in re.split(r"(?<=[.;])\s+", keynotes_text)
                    if len(s.split()) >= 3]
        if not latin or not (keynotes_text or fields):
            continue                      # skip stray bold that isn't a remedy
        recs.append({
            "remedy": latin, "name_full": name,
            "source": "Allen, Keynotes and Characteristics",
            "layer": "condensed-keynotes", "url": url,
            "keynotes": keynotes, "keynotes_text": keynotes_text,
            "modalities": mod,
            "relations": {"raw": rel, "remedies": _remedy_tokens(rel)},
            "raw_text": re.sub(r"[\x01\x02]", " ", re.sub(r"\s+", " ", normalize(body))),
        })
    return recs


# --- Hering (one page per remedy; 1-48 section schema; bold = important) -----
# Page anatomy (verified against /hering/a/abrot.htm):
#   * ONE page per remedy = the full text. The "-kn1"/"-kn2"/"-kn3" siblings are
#     FILTERED VIEWS ("Red only" / "Red and blue only") -> EXCLUDE them or every
#     symptom gets ingested 2-3 times.
#   * Sections: "MIND. [1] [Abrot.]" ... "RELATIONS. [48]" = Hering's fixed
#     1-48 schema. The [n] is the SECTION NUMBER, not a grade.
#   * BOLD symptom text = higher importance (the "red" grade in print).
#   * Greek theta separates a symptom from the clinical condition it cured.

HERING_REMEDY = re.compile(r"/hering/[a-z]/[\w-]+\.htm$", re.I)
HERING_KN = re.compile(r"-kn\d*\.htm$", re.I)                 # filtered views: skip
# Real heading: "MIND. [1] [Abrot.]"  ->  NAME comes FIRST, no brackets in it.
# Page-top TOC looks like "[1] Mind. [2] Sensorium." -> number first. Forbidding
# "[" inside the name group is what separates them.
HERING_SECT = re.compile(r"^([^\[\]]{2,60}?)\.?\s*\[(\d{1,2})\]")
THETA = re.compile(r"\s*\u03b8\s*")


def discover_hering(base, book):
    """Letter pages -> {url: (full_name, abbrev)}. Excludes -kn filtered views."""
    found: dict[str, tuple[str, str]] = {}
    for L in LETTERS:
        page = urljoin(base, f"{L}.htm")
        try:
            soup = BeautifulSoup(fetch(page, book), "html.parser")
        except Exception as e:
            print(f"    ! index {L}.htm: {e}")
            continue
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0]
            u = urljoin(page, href)
            if not HERING_REMEDY.search(u) or HERING_KN.search(u):
                continue
            abbrev = normalize(a.get_text(" ", strip=True))
            # The index lists: <b>Full Latin Name</b> ----- <a>Abbrev.</a>
            full = ""
            prev = a.find_previous(["b", "strong"])
            if prev:
                cand = normalize(prev.get_text(" ", strip=True))
                if cand and not cand.lower().startswith(
                        ("main", "the guiding", "presented", "copyright")) and len(cand) < 60:
                    full = cand
            if u not in found or (full and not found[u][0]):
                found[u] = (full or abbrev, abbrev)
    return found


def parse_hering(soup, url, name_hint=("", "")):
    """One Hering remedy page -> schema sections with per-symptom importance."""
    from bs4 import NavigableString
    full_name, abbrev = name_hint
    # The page <title> is the reliable source: "Abrotanum. - THE GUIDING SYMPTOMS..."
    # The letter-index guess is only a fallback (its bold-scan mis-pairs names).
    if soup.title:
        t = normalize(soup.title.get_text()).split(" - ")[0].strip(" .")
        if t and not t.upper().startswith("THE GUIDING"):
            full_name = t

    # Group text nodes by their nearest BLOCK ancestor, so inline <i>/<b> runs
    # inside one sentence don't get split into separate "symptoms".
    BLOCKS = ("p", "div", "li", "td", "blockquote", "body", "tr", "table")

    def block_of(node):
        for pa in node.parents:
            if getattr(pa, "name", "") in BLOCKS:
                return id(pa)
        return 0

    nodes: list[tuple[str, bool]] = []
    cur_key, buf, bold_chars, tot_chars = None, [], 0, 0
    for el in (soup.body or soup).descendants:
        if not isinstance(el, NavigableString):
            continue
        txt = normalize(str(el))
        if not txt:
            continue
        key = block_of(el)
        bold = any(getattr(pa, "name", "") in ("b", "strong") for pa in el.parents)
        if key != cur_key and buf:
            nodes.append((normalize(" ".join(buf)), bold_chars > tot_chars / 2))
            buf, bold_chars, tot_chars = [], 0, 0
        cur_key = key
        buf.append(txt)
        tot_chars += len(txt)
        if bold:
            bold_chars += len(txt)
    if buf:
        nodes.append((normalize(" ".join(buf)), bold_chars > tot_chars / 2))

    sections, cur = [], None
    for txt, bold in nodes:
        if is_boiler(txt):
            continue
        m = HERING_SECT.match(txt)
        if m and not re.match(r"^\s*\[", txt):
            cur = {"no": int(m.group(2)), "name": m.group(1).strip(" .").title(),
                   "symptoms": []}
            sections.append(cur)
            txt = txt[m.end():].strip()
            txt = re.sub(r"^\[?" + re.escape(abbrev.rstrip(".")) + r"\.?\]?", "", txt).strip(" []")
            if not txt:
                continue
        if cur is None:
            continue
        for piece in re.split(r"(?<=[.;])\s+(?=[A-Z(])", txt):
            piece = piece.strip()
            if len(piece) < 5:
                continue
            bits = THETA.split(piece)
            sym = {"text": re.sub(r"\s+", " ", bits[0]).strip(), "important": bold}
            if len(bits) > 1 and bits[1].strip():
                sym["clinical"] = bits[1].strip(" .")
            cur["symptoms"].append(sym)

    sections = [s for s in sections if s["symptoms"]]
    if not sections:
        # Short/irregular entries carry no "[n]" schema headings. Previously they
        # produced zero sections and were dropped entirely (216 of 413 pages lost).
        loose = []
        for txt, bold in nodes:
            if is_boiler(txt) or len(txt) < 12 or HERING_SECT.match(txt):
                continue
            for piece in re.split(r"(?<=[.;])\s+(?=[A-Z(])", txt):
                piece = piece.strip()
                if len(piece) >= 5:
                    bits = THETA.split(piece)
                    sym = {"text": re.sub(r"\s+", " ", bits[0]).strip(),
                           "important": bold}
                    if len(bits) > 1 and bits[1].strip():
                        sym["clinical"] = bits[1].strip(" .")
                    loose.append(sym)
        if loose:
            sections = [{"no": 0, "name": "General", "symptoms": loose}]
    rel = next((s for s in sections if s["name"].lower().startswith("relation")), None)
    rel_raw = " ".join(x["text"] for x in rel["symptoms"]) if rel else ""
    all_text = " ".join(x["text"] for s in sections for x in s["symptoms"])
    return {
        "remedy": full_name, "abbrev": abbrev,
        "source": "Hering, Guiding Symptoms", "layer": "exhaustive", "url": url,
        "sections": sections,
        "n_symptoms": sum(len(s["symptoms"]) for s in sections),
        "relations": {"raw": rel_raw, "remedies": _remedy_tokens(rel_raw)},
        "raw_text": re.sub(r"\s+", " ", all_text).strip(),
    }



# --- Kent (rubric -> graded remedy list; grade lives in bold/italic tags) ----
# Grade 3 = <b> (Title-case), Grade 2 = <i> (lowercase), Grade 1 = plain.
# The rubric HEADING is also bold, so we only read grades AFTER the ':' colon.

KENT_PART = re.compile(r"kent\d+\.htm$", re.I)                 # content pages: kent0320.htm
KENT_CHAPTER_HREF = re.compile(r"kent[a-z]+\.htm", re.I)       # chapter files: kentmind.htm
KENT_FRONT = re.compile(r"kent(pref|cont|reme|userep|repert)\.htm", re.I)


def _grade_runs(block):
    """Yield (text, grade) for each string in a block, grade from nearest b/i ancestor."""
    from bs4 import NavigableString
    for s in block.descendants:
        if isinstance(s, NavigableString):
            g = 1
            for p in s.parents:
                if p is block:
                    break
                nm = getattr(p, "name", "")
                if nm in ("b", "strong"):
                    g = max(g, 3)
                elif nm in ("i", "em"):
                    g = max(g, 2)
            yield str(s), g


def _parse_remedies(runs_after_colon):
    """runs_after_colon: list of (text, grade). Split into {name, grade} entries."""
    out = []
    for text, g in runs_after_colon:
        for tok in text.split(","):
            name = normalize(tok).strip(" .;")
            if not name or name.lower().startswith("see") or "(" in name:
                continue
            if not re.match(r"^[A-Za-z][\w-]*\.?$", name):   # a remedy abbrev, not prose
                continue
            out.append({"name": name.rstrip("."), "grade": g})
    return out


def parse_kent(soup, url):
    blocks = soup.find_all("p") or [soup.body or soup]
    records, chapter, main_rubric, page = [], "", "", ""
    for b in blocks:
        # chapter marker: a link to a kentrep/ chapter file
        a = b.find("a", href=KENT_CHAPTER_HREF)
        if a:
            chapter = normalize(a.get_text()).title()
            pm = re.search(r"p\.\s*(\d+)", b.get_text())
            if pm:
                page = pm.group(1)
        runs = list(_grade_runs(b))
        full = normalize("".join(t for t, _ in runs))
        if ":" not in full:
            # bold-only heading with no remedies -> may set a main rubric context
            if runs and runs[0][1] == 3 and full and "----" not in full and len(full) < 60:
                main_rubric = full.strip()
            continue
        # split label vs remedies at first colon, keeping grade runs aligned
        idx, acc = None, 0
        for k, (t, g) in enumerate(runs):
            if ":" in t:
                idx = k; break
            acc += 1
        if idx is None:
            continue
        label = normalize("".join(t for t, _ in runs[:idx]) + runs[idx][0].split(":")[0])
        first_grade = next((g for t, g in runs if t.strip()), 1)
        after = [(runs[idx][0].split(":", 1)[1], runs[idx][1])] + runs[idx + 1:]
        remedies = _parse_remedies(after)
        if not remedies:
            continue
        if first_grade == 3:                       # new main rubric
            main_rubric = label.strip()
            rubric_full = label.strip()
        else:                                      # sub-rubric under current main
            rubric_full = f"{main_rubric} > {label.strip()}" if main_rubric else label.strip()
        records.append({
            "chapter": chapter, "rubric": rubric_full,
            "remedies": remedies,
            "grade3": [r["name"] for r in remedies if r["grade"] == 3],
            "page": page, "source": "Kent, Repertory", "layer": "repertory", "url": url,
        })
    return records


def discover_kent(base, book):
    idx = BeautifulSoup(fetch(urljoin(base, "index.htm"), book), "html.parser")
    chapters = [urljoin(base, a["href"].split("#")[0]) for a in idx.find_all("a", href=True)
                if KENT_CHAPTER_HREF.search(a["href"].split("#")[0])
                and not KENT_FRONT.search(a["href"].split("#")[0])]
    chapters = sorted(set(chapters))
    print(f"   found {len(chapters)} chapter files")
    pages, seen = [], set()
    for ch in chapters:
        try:
            soup = BeautifulSoup(fetch(ch, book), "html.parser")
        except Exception:
            continue
        for a in soup.find_all("a", href=True):
            u = urljoin(ch, a["href"].split("#")[0])
            if KENT_PART.search(u) and u not in seen:
                seen.add(u); pages.append(u)
    return sorted(seen)


# --- Organon (split on § aphorism number, keep 6th ed, separate footnotes) ---

ORG_HEAD = re.compile(r"^[§\u00a7]\s*(\d+)(?:\s+(Fifth|Sixth)\s+Edition)?\s*$", re.I)
ORG_FOOT = re.compile(r"^\s*(\d{1,2}|\*)\s+\S")    # inline: "1 His mission ..."
ORG_MARK = re.compile(r"(\d{1,2}|\*)\.?")            # marker alone on its own line


def parse_organon(soup, url):
    """Split on the <a name="P7"> / <a name="P6E6"> anchors the index links to.
    Anchors are deterministic; the earlier bold-text match failed whenever a page
    merged the edition header into the same <b> as the first marker (organ001.htm
    lost ALL of aphorisms 1-19 that way)."""
    marked = 0
    for a in soup.find_all("a"):
        nm = (a.get("name") or a.get("id") or "").strip()
        m = re.fullmatch(r"[Pp](\d{1,3})(E[56])?", nm)
        if not m:
            continue
        a.replace_with(f"\x00{m.group(1)}|{(m.group(2) or '').upper()}\x00")
        marked += 1

    if not marked:                       # fallback: old bold-heading scan
        for b in soup.find_all(["b", "strong"]):
            if b.find_parent("a"):
                continue
            mm = ORG_HEAD.search(b.get_text(" ", strip=True))
            if mm:
                b.replace_with(f"\x00{mm.group(1)}|"
                               f"{'E5' if (mm.group(2) or '').lower()=='fifth' else 'E6'}\x00")

    text = "\n".join(l for l in normalize(soup.get_text("\n")).splitlines()
                     if not is_boiler(l))
    parts = text.split("\x00")
    recs = []
    for i in range(1, len(parts) - 1, 2):
        num, edition = (parts[i].split("|") + [""])[:2]
        if edition == "E5":                       # 5th-edition-only variant: skip
            continue
        n = int(num)
        body = parts[i + 1]
        # strip a leading "§ 7" / "§ 6 Sixth Edition" left over next to the anchor
        body = re.sub(r"^\s*[\u00a7§]?\s*\d{1,3}\s*(Fifth|Sixth)?\s*(Edition)?\s*",
                      "", body, flags=re.I)
        lines = [l for l in body.splitlines() if l.strip()]
        main, foots, cur = [], [], None
        for l in lines:
            s = l.strip()
            # Footnote markers appear TWO ways in this HTML:
            #   (a) inline:      "1 His mission is not, however, ..."
            #   (b) marker alone on its own line, text on the next line(s).
            # Only (a) was handled before, which is why footnotes came out at 1%.
            bare = ORG_MARK.fullmatch(s)
            inline = ORG_FOOT.match(l)
            if (bare or inline) and main:          # require the aphorism text first
                if cur:
                    foots.append(cur)
                if bare:
                    cur = {"marker": s.rstrip("."), "text": ""}
                else:
                    mk = re.match(r"^\s*(\d{1,2}|\*)\s+(.*)", l)
                    cur = {"marker": mk.group(1), "text": mk.group(2)}
            elif cur is not None:
                cur["text"] = (cur["text"] + " " + s).strip()
            else:
                main.append(s)
        foots = [f for f in ([*foots, cur] if cur else foots) if f["text"]]
        for f in foots:
            f["text"] = re.sub(r"\s+", " ", f["text"]).strip()
        txt = re.sub(r"\s+", " ", " ".join(main)).strip()
        if not txt and not foots:
            continue
        recs.append({
            "aphorism_no": n, "edition": "6th", "text": txt, "footnotes": foots,
            "source": "Hahnemann, Organon of Medicine, 6th ed. (Boericke tr., 1921)",
            "layer": "organon",
            "url": f"{url}#P{n}" + ("E6" if edition == "E6" else ""),
        })
    # a page can carry both P6E5 and P6E6; keep the richest record per number
    best = {}
    for r in recs:
        k = r["aphorism_no"]
        if k not in best or len(r["text"]) > len(best[k]["text"]):
            best[k] = r
    return list(best.values())


def discover_organon(base, book):
    idx = BeautifulSoup(fetch(urljoin(base, "index.htm"), book), "html.parser")
    pages = set()
    for a in idx.find_all("a", href=True):
        href = a["href"].split("#")[0]              # index links are organ001.htm#P1
        if re.search(r"organ\d+\.htm$", href, re.I):
            pages.add(urljoin(base, href))
    print(f"   found {len(pages)} page files")
    return sorted(pages)


# --- Kent's Lectures on Homoeopathic Philosophy (one prose lecture per page) --
# Structure (verified on books3/kentlect/lect04.htm):
#   * index.htm links lect01.htm .. lect37.htm, one LECTURE per page.
#   * Heading line: 'LECTURE 4 : Organon § 4. "Fixed principles." Law And ...'
#     -> the "§ N" is the Organon cross-reference; captured so a lecture can be
#        linked to organon_6th.json aphorism N.
#   * Body is clean prose (one sentence per <p>). No sub-schema -> Nash-like.
#   * Different presenter (Seror/Cazalet 1999) -> its own boilerplate above.

KENTLECT_HEAD = re.compile(
    r"LECTURE\s+(\d+)\s*:\s*(.*?)(?:\s*$)", re.I | re.S)
KENTLECT_ORG = re.compile(
    r"Organon\s*[\u00a7\u00a7]+\s*(\d+(?:\s*(?:and|,|[-\u2013])\s*\d+)*(?:\s*et seq\.?)?)",
    re.I)


def parse_kentlect(soup, url):
    r"""One prose lecture per page (Seror/Cazalet presentation).

    Heading "LECTURE N : Organon § M. Topic" WRAPS across lines in the HTML
    (Organon on one line, "§ M." on the next), so we glue the run starting at
    "LECTURE N :" before parsing -- otherwise title becomes just "Organon" and the
    § reference is lost (seen as ref='' on the first run).
    """
    ttl = normalize(soup.title.get_text()) if soup.title else ""
    mnum = re.search(r"LECTURE\s+(\d+)", ttl, re.I)
    lecture_no = int(mnum.group(1)) if mnum else None

    lines = [l.strip() for l in normalize(soup.get_text("\n")).splitlines()
             if l.strip() and not is_boiler(l)]

    heading, org_ref, head_idx = "", "", -1
    for i, l in enumerate(lines[:8]):
        if re.match(r"LECTURE\s+\d+\s*:", l, re.I):
            head_idx = i
            joined = l
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j]
                # The first line after "LECTURE N :" IS the title, even when it
                # opens with "The"/"In"/"It" -- so only treat *later* lines as body
                # prose, and only on unambiguous openers or long lines. (fixes the
                # wrapped-heading titles: #13/#17/#24/#33 etc.)
                if j > i + 1 and (nxt.lower().startswith(("dr samuel", "we ")) or len(nxt) > 70):
                    break
                joined += " " + nxt
                if "." in nxt:                     # heading rarely spans a full stop
                    break
            joined = re.sub(r"\s+", " ", joined).strip()
            hm = re.match(r"LECTURE\s+\d+\s*:\s*(.+)", joined, re.I)
            heading = hm.group(1).strip() if hm else ""
            om = KENTLECT_ORG.search(joined)
            if om:
                org_ref = re.sub(r"\s+", " ", om.group(1)).strip().rstrip(".")
            # Strip ONLY the "Organon \u00a7 <ref>" token -- never title words.
            # The old [^."]* was greedy and ate the title whenever the heading was
            # shaped "\u00a7 N <Title>." with no period after N (#2, #11 -> empty).
            # Ref shape mirrors KENTLECT_ORG's capture group (+ optional "(1)").
            heading = re.sub(
                r"Organon\s*[\u00a7\u00a7]+\s*"
                r"\d+(?:\s*(?:and|,|[-\u2013])\s*\d+)*"   # 2 | 10 and 11 | 21-25
                r"(?:\s*\(\d+\))?"                        # (1)
                r"(?:\s*et seq\.?)?"                      # et seq.
                r"\.?\s*",
                "", heading)
            heading = re.sub(r"\[\s*Read\s*\]|\[\s*\]|\bRead\b\s*$", "", heading)
            heading = heading.strip(' ."[]').lstrip('" ').strip(' ."[]')
            break

    # Body = lines after the heading, dropping caption / leftover section-ref.
    body_lines = []
    for l in lines[head_idx + 1:] if head_idx >= 0 else lines:
        if len(body_lines) < 3 and len(l) < 80 and re.match(
                r"(\u00a7|Organon\s*\u00a7|Dr Samuel|et seq|Law And Government|"
                r"Simple substance|[\"\u201c].{0,70}[\"\u201d]\.?$)", l, re.I):
            continue
        body_lines.append(l)
    body = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
    # final sweep: strip a leading "§ 4. \"topic\"." or "et seq." remnant
    body = re.sub(r'^(?:\u00a7\s*[\d,\s]*(?:and\s*\d+)?\.?\s*'
                  r'(?:"[^"]*"|[A-Z][^.]*)?\.?\s*|et seq\.?\s*)+', "", body).strip()

    # aphorism numbers, incl. "10 and 11", "21-25" ranges, comma lists
    aph = []
    if org_ref:
        rng = re.match(r"(\d+)\s*[-\u2013]\s*(\d+)", org_ref)
        if rng:
            aph = list(range(int(rng.group(1)), int(rng.group(2)) + 1))
        else:
            aph = [int(t) for t in re.findall(r"\d+", org_ref)]

    # Also scan the BODY for every "\u00a7 N" / "paragraph N" / "Organon N" mention,
    # so thematic lectures (no heading ref) still link to the aphorisms they discuss.
    mentions = set(aph)
    for m in re.finditer(r"(?:[\u00a7\u00a7]|paragraph|Organon)\s*(\d{1,3})"
                         r"(?:\s*(?:and|,|[-\u2013]|to)\s*(\d{1,3}))?", body, re.I):
        a = int(m.group(1))
        if 1 <= a <= 294:
            mentions.add(a)
            if m.group(2):
                b = int(m.group(2))
                if 1 <= b <= 294 and b - a < 20:
                    mentions.update(range(min(a, b), max(a, b) + 1))
    organon_mentions = sorted(mentions)

    return [{
        "lecture_no": lecture_no,
        "title": heading,
        "organon_ref": org_ref,
        "organon_aphorisms": aph,          # from the heading (authoritative)
        "organon_mentions": organon_mentions,  # every aphorism cited anywhere
        "text": body,
        "n_words": len(body.split()),
        "source": "Kent, Lectures on Homoeopathic Philosophy (1900)",
        "layer": "philosophy", "url": url,
    }]


def discover_kentlect(base, book):
    idx = BeautifulSoup(fetch(urljoin(base, "index.htm"), book), "html.parser")
    pages = set()
    for a in idx.find_all("a", href=True):
        href = a["href"].split("#")[0]
        if re.search(r"lect\d+\.htm$", href, re.I):
            pages.add(urljoin(base, href))
    print(f"   found {len(pages)} lecture pages")
    return sorted(pages)



# --- book registry ----------------------------------------------------------

BOOKS = {
    "boericke": dict(
        base="http://www.homeoint.org/books/boericmm/",
        index_pages=[f"{L}.htm" for L in LETTERS],
        content=re.compile(r"^[a-z]/[\w-]+\.htm$"),      # a/acon.htm
        parser=parse_boericke, out="boericke_materia_medica.json"),
    "clarke": dict(
        base="http://www.homeoint.org/clarke/",
        index_pages=[f"{L}.htm" for L in LETTERS],
        content=re.compile(r"^[a-z]?/?[\w-]+\.htm$"),    # CALIBRATE against a live letter page
        parser=parse_clarke, out="clarke_dictionary.json"),
    "nash": dict(
        base="http://www.homeoint.org/books2/nashtherap/",
        index_pages=["index.htm", ""],
        content=re.compile(r"mmh\d+\.htm"),              # 24 chapter files
        parser=parse_nash, out="nash_leaders.json"),
    "allen": dict(
        base="http://www.homeoint.org/books/allkeyn/",
        index_pages=["index.htm", ""],
        content=re.compile(r"allkey(?!pr)[a-z]+\.htm"),  # allkeyaa..allkeytz (skip preface)
        parser=parse_allen, out="allen_keynotes.json"),
    "hering": dict(
        base="http://www.homeoint.org/hering/",
        special="hering",                                # multi-part BFS pipeline
        out="hering_guiding_symptoms.json"),
    "kent": dict(
        base="http://www.homeoint.org/books/kentrep/",
        special="kent",                                  # rubric->graded remedies
        out="kent_repertory.json"),
    "organon": dict(
        base="http://www.homeoint.org/books/hahorgan/",
        special="organon",                               # split on § aphorism no.
        out="organon_6th.json"),
    "kentlect": dict(
        base="http://www.homeoint.org/books3/kentlect/",
        special="kentlect",                              # one prose lecture per page
        out="kent_lectures_philosophy.json"),
}


# --- dedup: Allen (condensed) vs Hering (exhaustive) ------------------------
# Allen's Keynotes is largely selected VERBATIM from Hering's Guiding Symptoms
# (homeoint's own introduction says practically not a sentence was changed).
# Ingesting both unflagged => the quiz bot asks the same symptom twice under two
# "different" authors. This pass FLAGS the overlap; it deletes nothing.

STOP = set("the a an of in on to and or with is are was were be been it its his her "
           "as at by for from that this these those not no all any".split())


def _toks(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _containment(a, b):
    """|A n B| / min(|A|,|B|).

    The right metric here. Allen's keynotes are semicolon-separated FRAGMENTS
    while Hering's are full sentences; Jaccard punishes that length mismatch and
    scored 95% of pairs under 0.3, hiding a derivation that is real.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _name_key(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def _first_word(s):
    parts = (s or "").split()
    return re.sub(r"[^a-z]", "", parts[0].lower()) if parts else ""


def dedup(threshold=0.7):
    ap, hp = OUT / "allen_keynotes.json", OUT / "hering_guiding_symptoms.json"
    for f in (ap, hp):
        if not f.exists():
            print(f"Missing {f} - build it first (--book allen / --book hering).")
            return
    allen = json.loads(ap.read_text(encoding="utf-8"))
    hering = json.loads(hp.read_text(encoding="utf-8"))

    by_exact, by_first, by_abbrev = {}, {}, {}
    for h in hering:
        by_exact.setdefault(_name_key(h.get("remedy", "")), h)
        by_first.setdefault(_first_word(h.get("remedy", "")), h)
        by_abbrev.setdefault(_name_key(h.get("abbrev", "")), h)

    def match(name):
        k, f = _name_key(name), _first_word(name)
        if k in by_exact:
            return by_exact[k]
        if f in by_first:
            return by_first[f]
        for ab, h in by_abbrev.items():        # "abrot" is a prefix of "abrotanum"
            if ab and f.startswith(ab):
                return h
        return None

    matched = dup_total = key_total = 0
    for a in allen:
        h = match(a.get("remedy", ""))
        a["hering_match"] = h["remedy"] if h else None
        if not h:
            a["dedup"] = {"status": "no_hering_remedy"}
            continue
        matched += 1
        h_syms = [(s["text"], _toks(s["text"]), sec["name"])
                  for sec in h.get("sections", []) for s in sec["symptoms"]]
        h_all = set().union(*[t for _, t, _ in h_syms]) if h_syms else set()
        flagged = []
        for kn in a.get("keynotes", []):
            key_total += 1
            kt = _toks(kn)
            if len(kt) < 3:                 # too short to judge either way
                flagged.append({"keynote": kn, "duplicate": False, "score": 0.0,
                                "note": "too_short", "hering_text": None,
                                "hering_section": None})
                continue
            best, best_score, best_sec, best_jac = None, 0.0, "", 0.0
            for text, ht, secname in h_syms:
                sc = _containment(kt, ht)
                if sc > best_score:
                    best, best_score, best_sec = text, sc, secname
                    best_jac = _jaccard(kt, ht)
            is_dup = best_score >= threshold
            dup_total += is_dup
            flagged.append({"keynote": kn, "duplicate": bool(is_dup),
                            "score": round(best_score, 2),
                            "jaccard": round(best_jac, 2),
                            "in_remedy_text": round(_containment(kt, h_all), 2),
                            "hering_text": best if is_dup else None,
                            "hering_section": best_sec if is_dup else None})
        a["dedup"] = {"status": "matched", "hering_url": h.get("url"),
                      "duplicate_ratio": round(sum(f["duplicate"] for f in flagged) /
                                               max(len(flagged), 1), 2),
                      "keynotes": flagged}

    outp = OUT / "allen_keynotes.dedup.json"
    outp.write_text(json.dumps(allen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAllen records: {len(allen)}  |  matched to a Hering remedy: {matched}")
    print(f"Keynotes compared: {key_total:,}  |  flagged as Hering duplicates: "
          f"{dup_total:,} ({dup_total*100//max(key_total,1)}%)  [threshold {threshold}]")
    print(f"wrote -> {outp}")
    print("\nNothing deleted. Suggested indexing policy:")
    print("  * Allen  = 'condensed-keynotes' layer -> HIGH-YIELD questions")
    print("  * Hering = 'exhaustive' layer       -> DETAIL questions")
    print("  * Where a keynote is flagged duplicate, cite Allen and suppress the")
    print("    Hering twin for that question so students never see it twice.")


# --- stats ------------------------------------------------------------------

def _pct(n, d):
    return f"{n*100//max(d,1)}%"


def stats():
    files = sorted(OUT.glob("*.json"))
    if not files:
        print("No JSON in out/ yet.")
        return
    print(f"{'file':36} {'records':>8}  detail")
    print("-" * 104)
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"{f.name:36} {'ERR':>8}  {e}")
            continue
        if not isinstance(data, list) or not data:
            print(f"{f.name:36} {0:>8}  EMPTY - parser matched nothing")
            continue
        n = len(data)
        k = data[0]
        if "aphorism_no" in k:                                    # Organon
            nums = sorted(r["aphorism_no"] for r in data)
            have = set(nums)
            gaps = [i for i in range(nums[0], nums[-1] + 1) if i not in have]
            withf = sum(1 for r in data if r.get("footnotes"))
            empty = sum(1 for r in data if not r.get("text"))
            detail = (f"S{nums[0]}-S{nums[-1]}  gaps={len(gaps)}{gaps[:5] if gaps else ''}  "
                      f"with-footnotes={_pct(withf, n)}  empty-text={empty}")
        elif "rubric" in k:                                       # Kent
            chaps = len({r.get("chapter", "") for r in data})
            ents = sum(len(r.get("remedies", [])) for r in data)
            g = {1: 0, 2: 0, 3: 0}
            for r in data:
                for x in r.get("remedies", []):
                    g[x.get("grade", 1)] = g.get(x.get("grade", 1), 0) + 1
            detail = (f"chapters={chaps}  remedy-entries={ents:,}  "
                      f"g3={g[3]:,} g2={g[2]:,} g1={g[1]:,}  "
                      f"rubrics-with-g3={_pct(sum(1 for r in data if r.get('grade3')), n)}")
        elif "lecture_no" in k:                                    # Kent Lectures
            words = sum(r.get("n_words", 0) for r in data)
            xref = sum(1 for r in data if r.get("organon_aphorisms"))
            xref_any = sum(1 for r in data
                           if r.get("organon_aphorisms") or r.get("organon_mentions"))
            notitle = sum(1 for r in data if not r.get("title"))
            empty = sum(1 for r in data if r.get("n_words", 0) < 50)
            detail = (f"lectures  total-words={words:,}  avg={words//max(n,1):,}  "
                      f"heading-xref={_pct(xref, n)}  any-xref={_pct(xref_any, n)}  "
                      f"no-title={notitle}  "
                      f"empty-text={empty}")
        elif "sections" in k:                                     # Hering
            syms = sum(r.get("n_symptoms", 0) for r in data)
            imp = sum(1 for r in data for s in r.get("sections", [])
                      for x in s["symptoms"] if x.get("important"))
            clin = sum(1 for r in data for s in r.get("sections", [])
                       for x in s["symptoms"] if x.get("clinical"))
            noname = sum(1 for r in data if not r.get("remedy"))
            detail = (f"symptoms={syms:,}  important={_pct(imp, syms)}  "
                      f"with-clinical={_pct(clin, syms)}  missing-name={noname}")
        else:                                                     # materia medica
            kn = sum(len(r.get("keynotes", [])) for r in data)
            nomod = sum(1 for r in data if not r.get("modalities"))
            norel = sum(1 for r in data if not r.get("relations", {}).get("remedies"))
            noraw = sum(1 for r in data if not r.get("raw_text"))
            dup = sum(1 for r in data
                      if r.get("dedup", {}).get("duplicate_ratio", 0) >= .5)
            detail = (f"keynotes={kn:,} (avg {kn // max(n, 1)})  "
                      f"no-modalities={_pct(nomod, n)}  no-relations={_pct(norel, n)}  "
                      f"empty-raw={noraw}" + (f"  >=50%-dup={dup}" if dup else ""))
        print(f"{f.name:36} {n:>8}  {detail}")
    print("\nRed flags: EMPTY file, big gaps, empty-raw > 0, or no-relations at 100%.")



# --- diagnose ---------------------------------------------------------------
# Answers "what does this page actually look like?" from YOUR cache/ rather than
# guesswork. Run it whenever --stats shows a parser matching nothing.

def diagnose(book, n_show=20):
    d = CACHE / book
    files = sorted(d.glob("*.html")) if d.exists() else []
    if not files:
        print(f"No cached pages for '{book}'. Run --book {book} --limit 3 first.")
        return
    f = max(files, key=lambda x: x.stat().st_size)      # biggest = a content page
    soup = BeautifulSoup(f.read_text(encoding="utf-8", errors="replace"), "html.parser")
    print(f"page: {f.name}  ({f.stat().st_size:,} bytes)\n")

    names = [(a.get("name") or a.get("id")) for a in soup.find_all("a")
             if a.get("name") or a.get("id")]
    print(f"anchors ({len(names)}): {names[:14]}")
    tags = {}
    for t in soup.find_all(True):
        tags[t.name] = tags.get(t.name, 0) + 1
    print("tags:", dict(sorted(tags.items(), key=lambda x: -x[1])[:10]))

    def buckets(tagnames):
        noise, short, real = 0, 0, []
        for t in soup.find_all(tagnames):
            s = t.get_text(" ", strip=True)
            if not s or re.fullmatch(r"[\s<>_.,;:*+=|/\\-]{1,3}", s):
                noise += 1
            elif len(s) < 4:
                short += 1
            else:
                real.append(s)
        return noise, short, real

    for label, tagnames in (("<b>/<strong>", ["b", "strong"]), ("<i>/<em>", ["i", "em"])):
        noise, short, real = buckets(tagnames)
        print(f"\n{label}: {noise} symbol/noise, {short} very-short, {len(real)} substantive")
        for s in real[:n_show]:
            print("   |", s[:88])

    # Anything short, capitalised and repeated is probably a section heading.
    counts = {}
    for t in soup.find_all(["b", "strong", "font"]):
        s = t.get_text(" ", strip=True)
        if 4 <= len(s) <= 44 and s[:1].isupper():
            counts[s] = counts.get(s, 0) + 1
    print("\ncandidate headings (short, capitalised) -- shown as repr() so that")
    print("invisible/control characters (e.g. '\\x96' from &#150;) are visible:")
    for s, c in sorted(counts.items(), key=lambda x: -x[1])[:20]:
        print(f"   x{c:<3} {s[:70]!r}")
    print("\nUse these to fix the book's heading regex.")



def run(book: str, limit: int | None):
    cfg = BOOKS[book]
    if cfg.get("special") == "hering":
        print("== hering: discovering remedy pages (excluding -kn filtered views)…")
        found = discover_hering(cfg["base"], "hering")
        urls = sorted(found)
        if limit:
            urls = urls[:limit]
        print(f"   {len(urls)} remedy pages")
        records = []
        for i, u in enumerate(urls, 1):
            try:
                rec = parse_hering(BeautifulSoup(fetch(u, "hering"), "html.parser"),
                                   u, found[u])
                if rec["n_symptoms"]:
                    records.append(rec)
                if i % 25 == 0 or i == len(urls):
                    print(f"   [{i}/{len(urls)}] records: {len(records)}")
            except Exception as e:
                print(f"   ! {u}: {e}")
        OUT.mkdir(exist_ok=True)
        (OUT / cfg["out"]).write_text(json.dumps(records, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
        tot = sum(r["n_symptoms"] for r in records)
        print(f"   wrote {len(records)} remedies / {tot:,} symptoms -> out/{cfg['out']}")
        return

    if cfg.get("special") == "kent":
        print("== kent: discovering rubric pages…")
        pages = discover_kent(cfg["base"], "kent")
        if limit:
            pages = pages[:limit]
        print(f"   {len(pages)} content pages")
        records = []
        for i, u in enumerate(pages, 1):
            try:
                recs = parse_kent(BeautifulSoup(fetch(u, "kent"), "html.parser"), u)
                records.extend(recs)
                if i % 25 == 0 or i == len(pages):
                    print(f"   [{i}/{len(pages)}] rubrics so far: {len(records)}")
            except Exception as e:
                print(f"   ! {u}: {e}")
        OUT.mkdir(exist_ok=True)
        (OUT / cfg["out"]).write_text(json.dumps(records, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
        print(f"   wrote {len(records)} rubrics -> out/{cfg['out']}")
        return

    if cfg.get("special") == "kentlect":
        print("== kentlect: discovering lecture pages…")
        pages = discover_kentlect(cfg["base"], "kentlect")
        if limit:
            pages = pages[:limit]
        records = []
        for i, u in enumerate(pages, 1):
            try:
                rec = parse_kentlect(BeautifulSoup(fetch(u, "kentlect"), "html.parser"), u)
                records.extend(r for r in rec if r["n_words"] > 50)
                print(f"   [{i}/{len(pages)}] {u.rsplit('/',1)[-1]}: "
                      f"lect {rec[0]['lecture_no']} ({rec[0]['n_words']} words)")
            except Exception as e:
                print(f"   ! {u}: {e}")
        records.sort(key=lambda r: r["lecture_no"] or 0)
        OUT.mkdir(exist_ok=True)
        (OUT / cfg["out"]).write_text(json.dumps(records, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
        print(f"   wrote {len(records)} lectures -> out/{cfg['out']}")
        return

    if cfg.get("special") == "organon":
        print("== organon: discovering pages…")
        pages = discover_organon(cfg["base"], "organon")
        if limit:
            pages = pages[:limit]
        print(f"   {len(pages)} pages")
        records = []
        for i, u in enumerate(pages, 1):
            try:
                recs = parse_organon(BeautifulSoup(fetch(u, "organon"), "html.parser"), u)
                records.extend(recs)
                print(f"   [{i}/{len(pages)}] {u.rsplit('/',1)[-1]}: +{len(recs)} aphorisms")
            except Exception as e:
                print(f"   ! {u}: {e}")
        records.sort(key=lambda r: r["aphorism_no"])
        OUT.mkdir(exist_ok=True)
        (OUT / cfg["out"]).write_text(json.dumps(records, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
        nums = [r["aphorism_no"] for r in records]
        print(f"   wrote {len(records)} aphorisms (§{min(nums)}–§{max(nums)}) -> out/{cfg['out']}"
              if nums else "   wrote 0 aphorisms")
        return

    print(f"== {book}: discovering content pages…")
    urls = discover_links(cfg["base"], cfg["index_pages"], book, cfg["content"])
    # drop the letter-index pages themselves for boericke/clarke
    urls = [u for u in urls if not re.search(r"/[a-z]\.htm$", u)]
    urls = sorted(set(urls))
    if limit:
        urls = urls[:limit]
    print(f"   {len(urls)} pages")
    records = []
    for i, u in enumerate(urls, 1):
        try:
            soup = BeautifulSoup(fetch(u, book), "html.parser")
            recs = cfg["parser"](soup, u)
            records.extend(r for r in recs if r.get("remedy"))
            print(f"   [{i}/{len(urls)}] {u.rsplit('/',1)[-1]}: +{len(recs)}")
        except Exception as e:
            print(f"   ! {u}: {e}")
    OUT.mkdir(exist_ok=True)
    (OUT / cfg["out"]).write_text(json.dumps(records, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    print(f"   wrote {len(records)} records -> out/{cfg['out']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", choices=list(BOOKS) + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="cap pages (smoke test)")
    ap.add_argument("--rate", type=float, default=None,
                    help="seconds between requests (default 1.0; raise if you see 403s)")
    ap.add_argument("--dedup", action="store_true",
                    help="flag Allen keynotes that duplicate Hering symptoms")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="dedup containment cutoff (default 0.7)")
    ap.add_argument("--stats", action="store_true",
                    help="quality report across out/*.json")
    ap.add_argument("--diagnose", metavar="BOOK",
                    help="dump cached page structure for a book (anchors/bold/italics)")
    a = ap.parse_args()
    if a.rate:
        globals()["RATE_SECONDS"] = a.rate
    if a.dedup:
        dedup(a.threshold)
        return
    if a.stats:
        stats()
        return
    if a.diagnose:
        diagnose(a.diagnose)
        return
    for b in (list(BOOKS) if a.book == "all" else [a.book]):
        run(b, a.limit)
    if FAILED:
        print(f"\n!! {len(FAILED)} page(s) never fetched (server refused):")
        for u in sorted(FAILED)[:12]:
            print("   ", u)
        print("   Re-run the SAME command — cached pages are skipped, so only")
        print("   these are retried. If they persist, wait a few minutes first.")


if __name__ == "__main__":
    main()
