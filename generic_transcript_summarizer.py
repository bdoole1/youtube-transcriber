#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generic_transcript_summarizer.py

Summary
-------
A robust, domain-agnostic summarizer for noisy long-form transcripts
(e.g., YouTube auto-captions). Produces a structured, presentation-ready
summary with sections like:
- TL;DR (1–2 lines)
- Core Points (short bullets)
- Themes (mid-level bullets)
- Cause → Effect pairs
- Actionable Takeaways
- Key Phrases
- Open Questions

Design Principles
-----------------
1) **Generic**: No domain lexicons; works for any topic.
2) **Dependency-light**: Standard library only; no network calls.
3) **Deterministic**: Results are stable given the same input/config.
4) **Composable**: Clear Strategy interfaces to swap heuristics later.
5) **Practical**: Handles messy spoken text, filler, timestamps, applause.

Implementation Notes
--------------------
- Sentence segmentation with punctuation heuristics + fallback chunking.
- Hybrid scoring (TF-IDF-ish + TextRank-like graph centrality).
- Redundancy control via Maximal Marginal Relevance (MMR).
- RAKE-like keyphrase extraction (stopword-based).
- Clean Markdown/JSON rendering.

Python: 3.9+ (tested up to 3.13)
Author: You (future-proofed and refactor-friendly)

License: MIT (consider adding your header)
"""

from __future__ import annotations

import re
import math
import json
import textwrap
from dataclasses import dataclass, asdict
from typing import List, Tuple, Iterable, Optional, Dict
from collections import Counter, defaultdict

# =============================================================================
# ---------------------------  CONFIG & CONSTANTS  ----------------------------
# =============================================================================

# A lean stopword set that works “okay” across English transcripts.
# You can swap this out or extend it at runtime.
STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","that","this","those","these",
    "in","on","at","to","from","by","with","for","of","as","is","are","was","were","be",
    "been","being","it","its","into","over","about","after","before","so","such","not",
    "no","nor","do","does","did","doing","will","would","can","could","should","may",
    "might","must","have","has","had","having","you","we","they","i","he","she","him",
    "her","them","our","your","their","me","my","mine","ours","yours","theirs",
    "uh","um","like","you know","kind","sort","sort of","kind of","okay","ok","well","right"
}

# Basic regex helpers (compiled once)
SENTENCE_END_RE = re.compile(r'([.!?…]+)(\s+|$)')
WHITESPACE_RE = re.compile(r'\s+')
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# =============================================================================
# --------------------------  TEXT PRE-PROCESSING  ----------------------------
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize transcript-like text:
    - Remove timestamps, stage directions (e.g., (Applause)).
    - Normalize dash/quote variants and collapse repetitive noise.
    - Canonicalize whitespace.

    The goal is to improve downstream segmentation without losing meaning.
    """
    if not text:
        return ""

    t = text

    # Normalize unicode punctuation commonly seen in captions
    t = (t.replace("—", "-")
           .replace("–", "-")
           .replace("’", "'")
           .replace("“", '"')
           .replace("”", '"'))

    # Remove [00:10], [00:01:23], etc.
    t = re.sub(r'\[(\d{1,2}:){1,2}\d{2}\]', ' ', t)

    # Remove common stage directions / cues
    t = re.sub(r'\((?:applause|laughter|music|cheers|cough|sneeze|inaudible|noise).*?\)', ' ', t, flags=re.I)

    # Collapse long runs of interjections like "Oh! Oh! Oh!"
    t = re.sub(r'(oh!?[\s,.!?]*){3,}', ' ', t, flags=re.I)

    # Normalize whitespace
    t = WHITESPACE_RE.sub(' ', t).strip()
    return t


def split_sentences(text: str, min_words: int = 6) -> List[str]:
    """
    Segment text into "sentences" with heuristics that tolerate spoken-text artifacts.

    Strategy:
    - Primary: split on ., !, ?, …
    - Merge fragments shorter than `min_words` into neighbors.
    - If the whole text has few enders, fall back to fixed-size chunking.

    Returns a list of reasonably complete clauses/sentences.
    """
    if not text:
        return []

    # Try punctuation-based splitting
    parts = []
    start = 0
    for m in SENTENCE_END_RE.finditer(text):
        end = m.end()
        parts.append(text[start:end].strip())
        start = end
    if start < len(text):
        parts.append(text[start:].strip())

    # Fallback if we didn't really split (e.g., no punctuation in captions)
    if len(parts) <= 2:
        # Chunk by ~120-180 chars to create manageable units
        chunks = []
        s = text.strip()
        chunk_size = 160
        while s:
            chunks.append(s[:chunk_size])
            s = s[chunk_size:]
        parts = [c.strip() for c in chunks if c.strip()]

    # Merge too-short fragments to avoid bullet noise
    merged: List[str] = []
    buf = ""
    for s in parts:
        if len(s.split()) < min_words:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                merged.append(buf)
                buf = ""
            merged.append(s)
    if buf:
        merged.append(buf)

    # Final cleanup
    sentences = [s.strip() for s in merged if s.strip()]
    return sentences


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens (very light tokenizer)."""
    return TOKEN_RE.findall(text.lower())


# =============================================================================
# -----------------------  SCORING & SELECTION CORE  --------------------------
# =============================================================================

def tfidf_like_scores(sentences: List[str], stopwords: set = STOPWORDS) -> List[float]:
    """
    A simple TF-IDF-ish sentence importance:
    - Compute term DF at sentence level
    - Score sentence by mean(TF * IDF) over non-stopword tokens
    - Normalize to [0,1]
    This rewards sentences containing discriminative terms repeated across text.
    """
    dfs = Counter()
    sent_tokens = []
    for s in sentences:
        toks = [t for t in tokenize(s) if t not in stopwords]
        sent_tokens.append(toks)
        dfs.update(set(toks))

    n = len(sentences) or 1
    raw = []
    for toks in sent_tokens:
        s_len = max(len(toks), 1)
        tfs = Counter(toks)
        score = 0.0
        for term, tf in tfs.items():
            df = dfs[term]
            idf = math.log((n + 1) / (df + 1)) + 1.0  # smoothed, positive
            score += (tf / s_len) * idf
        # Light length regularization (mid-length sentences tend to summarize better)
        if 12 <= len(toks) <= 40:
            score *= 1.05
        raw.append(score)

    if not raw:
        return [0.0] * len(sentences)
    mx = max(raw) or 1.0
    return [x / mx for x in raw]


def textrank_like_scores(sentences: List[str], stopwords: set = STOPWORDS, iters: int = 20, damping: float = 0.85) -> List[float]:
    """
    TextRank-style centrality using token Jaccard similarity as edge weights.

    Steps:
    - Build sentence token sets
    - Fully-connected graph with edge weights = Jaccard(token_set_i, token_set_j)
    - Power-iterate PageRank with damping

    Notes:
    - This is light-weight (no 3rd-party libs) and domain-agnostic.
    - Jaccard on sets keeps it simple; cosine/TF-IDF would also work if desired.
    """
    N = len(sentences)
    if N == 0:
        return []

    sets = [set(t for t in tokenize(s) if t not in stopwords) for s in sentences]

    # Precompute pairwise Jaccard
    W = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            a, b = sets[i], sets[j]
            if not a or not b:
                w = 0.0
            else:
                w = len(a & b) / len(a | b)
            W[i][j] = W[j][i] = w

    # Normalize rows to make stochastic
    for i in range(N):
        s = sum(W[i]) or 1.0
        for j in range(N):
            W[i][j] /= s

    # Power iteration
    pr = [1.0 / N] * N
    for _ in range(iters):
        new = []
        for i in range(N):
            rank_sum = sum(W[j][i] * pr[j] for j in range(N))
            new.append((1 - damping) / N + damping * rank_sum)
        pr = new

    # Normalize to [0,1]
    mx = max(pr) or 1.0
    return [p / mx for p in pr]


def mmr_select(
    sentences: List[str],
    scores: List[float],
    k: int,
    diversity: float = 0.7,
    stopwords: set = STOPWORDS
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) selector:
    Select k indices that maximize relevance (scores) while minimizing redundancy.

    mmr = λ * relevance - (1-λ) * max_similarity_to_selected

    - `diversity` is λ (0..1): higher = prefer relevance more; lower = prefer diversity.
    - Similarity computed via Jaccard over token sets.

    Returns a list of sentence indices in selection order.
    """
    N = len(sentences)
    if N == 0 or k <= 0:
        return []

    k = min(k, N)
    token_sets = [set(t for t in tokenize(s) if t not in stopwords) for s in sentences]

    selected: List[int] = []
    candidates: set[int] = set(range(N))

    # Seed with the highest-scoring sentence
    first = max(range(N), key=lambda i: scores[i])
    selected.append(first)
    candidates.remove(first)

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    while len(selected) < k and candidates:
        best_i = None
        best_val = -1e9
        for i in list(candidates):
            sim = 0.0
            if selected:
                sim = max(jaccard(token_sets[i], token_sets[j]) for j in selected)
            val = diversity * scores[i] - (1 - diversity) * sim
            if val > best_val:
                best_val = val
                best_i = i
        selected.append(best_i)  # type: ignore[arg-type]
        candidates.remove(best_i)  # type: ignore[arg-type]

    return selected


# =============================================================================
# ----------------------------  KEY PHRASE (RAKE)  ----------------------------
# =============================================================================

def rake_keyphrases(text: str, stopwords: set = STOPWORDS, top_k: int = 12) -> List[str]:
    """
    RAKE-like keyphrase extraction (very lightweight):
    - Split text by stopwords and punctuation to get candidate phrases.
    - Score phrases by word co-occurrence degree/frequency.
    - Return top unique phrases (longer first bias).
    """
    # Split into candidate phrases (sequences of non-stopword tokens)
    words = tokenize(text)
    phrases: List[List[str]] = []
    current: List[str] = []
    for w in words:
        if w in stopwords:
            if current:
                phrases.append(current)
                current = []
        else:
            current.append(w)
    if current:
        phrases.append(current)

    # Compute word degree & frequency
    freq = Counter()
    degree = Counter()
    for ph in phrases:
        unique = set(ph)
        l = len(ph)
        for w in ph:
            freq[w] += 1
            degree[w] += l - 1  # degree excludes the word itself
        # Ensure words seen alone still get degree at least 1
        for w in unique:
            if degree[w] == 0:
                degree[w] = 1

    # Phrase score = sum( (degree(w) / freq(w)) for w in phrase )
    phrase_scores: Dict[Tuple[str, ...], float] = {}
    for ph in phrases:
        score = sum((degree[w] / max(freq[w], 1)) for w in ph)
        phrase_scores[tuple(ph)] = score

    # Rank phrases: score desc, then length desc (favor multi-word), then alphabetic
    ranked = sorted(
        phrase_scores.items(),
        key=lambda kv: (kv[1], len(kv[0]), " ".join(kv[0])),
        reverse=True
    )

    # Return top_k as strings, deduplicated (case-insensitive)
    seen = set()
    keyphrases: List[str] = []
    for ph_tuple, _ in ranked:
        phrase = " ".join(ph_tuple)
        low = phrase.lower()
        if low in seen:
            continue
        seen.add(low)
        keyphrases.append(phrase)
        if len(keyphrases) == top_k:
            break
    return keyphrases


# =============================================================================
# --------------------------  SUMMARY DATA STRUCTURE  -------------------------
# =============================================================================

@dataclass
class StructuredSummary:
    tldr: str
    core_points: List[str]
    themes: List[str]
    cause_effect: List[Tuple[str, str]]
    takeaways: List[str]
    key_phrases: List[str]
    open_questions: List[str]

    def to_markdown(self) -> str:
        """Render to clean Markdown suitable for docs/Obsidian."""
        md: List[str] = []
        md.append(f"**TL;DR**: {self.tldr}\n")

        if self.core_points:
            md.append("## Core Points")
            md.extend([f"- {p}" for p in self.core_points])
            md.append("")

        if self.themes:
            md.append("## Themes")
            md.extend([f"- {t}" for t in self.themes])
            md.append("")

        if self.cause_effect:
            md.append("## Cause → Effect")
            md.extend([f"- **{c}** → {e}" for c, e in self.cause_effect])
            md.append("")

        if self.takeaways:
            md.append("## Actionable Takeaways")
            md.extend([f"- {t}" for t in self.takeaways])
            md.append("")

        if self.key_phrases:
            md.append("## Key Phrases")
            md.extend([f"- {k}" for k in self.key_phrases])
            md.append("")

        if self.open_questions:
            md.append("## Open Questions")
            md.extend([f"- {q}" for q in self.open_questions])
            md.append("")

        return "\n".join(md)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# =============================================================================
# -----------------------------  SUMMARIZER CORE  -----------------------------
# =============================================================================

@dataclass
class SummarizerConfig:
    # Target sizes (tune for your UX)
    max_summary_sentences: int = 40     # upper bound for internal candidate pool
    tldr_sentences: int = 1             # 1–2 is typical
    core_points: int = 5
    themes: int = 6
    cause_effect_pairs: int = 5
    takeaways: int = 5
    key_phrases: int = 10
    open_questions: int = 3

    # Selection controls
    mmr_lambda: float = 0.7             # relevance vs diversity trade-off (0..1)
    min_sentence_words: int = 6         # avoid micro-fragments in splitting

    # Output constraints
    tldr_max_chars: int = 240           # a tweet-ish, concise TL;DR cap


class GenericTranscriptSummarizer:
    """
    Public entry point for producing StructuredSummary from raw transcripts.

    Pipeline:
    1) Normalize + split transcript.
    2) Score sentences (hybrid TF-IDF-like + TextRank-like).
    3) Select top-N diverse sentences for sections via MMR.
    4) Build structured sections with simple heuristics.
    """

    def __init__(self, config: Optional[SummarizerConfig] = None, stopwords: set = STOPWORDS):
        self.cfg = config or SummarizerConfig()
        self.stopwords = stopwords

    # -------------------------- Public API --------------------------

    def summarize_text(self, transcript: str) -> StructuredSummary:
        # 1) Normalize & split
        clean = normalize_text(transcript)
        sentences = split_sentences(clean, min_words=self.cfg.min_sentence_words)

        if len(sentences) == 0:
            return StructuredSummary(
                tldr="",
                core_points=[],
                themes=[],
                cause_effect=[],
                takeaways=[],
                key_phrases=[],
                open_questions=[]
            )

        if len(sentences) <= 3:
            # Very short content → minimal structure
            tldr = textwrap.shorten(" ".join(sentences), width=self.cfg.tldr_max_chars, placeholder=" …")
            return StructuredSummary(
                tldr=tldr,
                core_points=[tldr],
                themes=[],
                cause_effect=[],
                takeaways=[],
                key_phrases=rake_keyphrases(clean, stopwords=self.stopwords, top_k=min(5, self.cfg.key_phrases)),
                open_questions=[]
            )

        # 2) Sentence scoring (hybrid)
        s1 = tfidf_like_scores(sentences, stopwords=self.stopwords)
        s2 = textrank_like_scores(sentences, stopwords=self.stopwords)
        # Hybrid score: geometric mean (emphasize agreement)
        hybrid = [math.sqrt(max(s1[i], 1e-9) * max(s2[i], 1e-9)) for i in range(len(sentences))]

        # Cap the candidate pool for speed & determinism
        pool_size = min(self.cfg.max_summary_sentences, len(sentences))
        # Take top pool_size sentences by hybrid score
        top_indices = sorted(range(len(sentences)), key=lambda i: hybrid[i], reverse=True)[:pool_size]
        pool_sentences = [sentences[i] for i in top_indices]
        pool_scores = [hybrid[i] for i in top_indices]

        # 3) Select diverse subsets for different sections
        #    We'll use different λ (diversity) to vary flavor slightly per section.
        tldr_idx_rel = mmr_select(pool_sentences, pool_scores, k=max(1, self.cfg.tldr_sentences), diversity=0.85, stopwords=self.stopwords)
        core_idx     = mmr_select(pool_sentences, pool_scores, k=self.cfg.core_points, diversity=self.cfg.mmr_lambda, stopwords=self.stopwords)
        themes_idx   = mmr_select(pool_sentences, pool_scores, k=self.cfg.themes, diversity=self.cfg.mmr_lambda * 0.9 + 0.05, stopwords=self.stopwords)
        ce_idx       = mmr_select(pool_sentences, pool_scores, k=self.cfg.cause_effect_pairs, diversity=0.6, stopwords=self.stopwords)
        tips_idx     = mmr_select(pool_sentences, pool_scores, k=self.cfg.takeaways, diversity=0.75, stopwords=self.stopwords)

        # 4) Build sections
        tldr = self._build_tldr([pool_sentences[i] for i in tldr_idx_rel])
        core_points = self._build_bullets([pool_sentences[i] for i in core_idx], max_len_words=28)
        themes = self._build_bullets([pool_sentences[i] for i in themes_idx], max_len_words=36)
        cause_effect = self._build_cause_effect([pool_sentences[i] for i in ce_idx])
        takeaways = self._build_takeaways([pool_sentences[i] for i in tips_idx])
        key_phrases = rake_keyphrases(clean, stopwords=self.stopwords, top_k=self.cfg.key_phrases)
        open_questions = self._build_open_questions(core_points, themes, takeaways)

        return StructuredSummary(
            tldr=tldr,
            core_points=core_points,
            themes=themes,
            cause_effect=cause_effect,
            takeaways=takeaways,
            key_phrases=key_phrases,
            open_questions=open_questions
        )

    # ------------------------ Section Builders ------------------------

    def _build_tldr(self, sentences: List[str]) -> str:
        """
        Compress 1–2 highly-relevant sentences into a TL;DR line:
        - Trim parentheticals and leading conjunctions.
        - Cut after first strong clause boundary.
        - Enforce character cap.
        """
        if not sentences:
            return ""
        joined = " ".join(sentences)
        t = self._declutter(joined)
        # Cut after a strong boundary to avoid run-ons
        t = re.split(r'[;:—-]', t)[0]
        t = WHITESPACE_RE.sub(' ', t).strip()
        return textwrap.shorten(t, width=self.cfg.tldr_max_chars, placeholder=" …")

    def _build_bullets(self, sentences: List[str], max_len_words: int) -> List[str]:
        """
        Convert selected sentences into concise bullets:
        - Remove asides, soften hedges, clamp length.
        """
        bullets: List[str] = []
        for s in sentences:
            s2 = self._declutter(s)
            words = s2.split()
            if len(words) > max_len_words:
                s2 = " ".join(words[:max_len_words]) + " …"
            bullets.append(s2)
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for b in bullets:
            low = b.lower()
            if low in seen:
                continue
            seen.add(low)
            uniq.append(b)
        return uniq

    def _build_cause_effect(self, sentences: List[str]) -> List[Tuple[str, str]]:
        """
        Extract cause → effect pairs from sentences containing cues like:
        if/then, so/therefore/thus/hence, because.

        Heuristic parsing keeps it generic and domain-independent.
        """
        pairs: List[Tuple[str, str]] = []
        for s in sentences:
            s_clean = self._declutter(s)
            # Try "if ... then ..." pattern
            m = re.search(r'\bif\s+(.+?)(?:,|\s)\s*(?:then\s+)?(.+)', s_clean, flags=re.I)
            if m:
                cause = m.group(1).strip(" ,.-")
                effect = m.group(2).strip(" ,.-")
                if cause and effect and cause != effect:
                    pairs.append((cause, effect))
                    continue
            # Try "... therefore/so/thus/hence ..."
            m2 = re.split(r'\b(therefore|thus|so|hence)\b', s_clean, maxsplit=1, flags=re.I)
            if len(m2) >= 3:
                cause = m2[0].strip(" ,.-")
                effect = m2[-1].strip(" ,.-")
                if cause and effect and cause != effect:
                    pairs.append((cause, effect))
        # Deduplicate pairs
        uniq: List[Tuple[str, str]] = []
        seen = set()
        for c, e in pairs:
            key = (c.lower(), e.lower())
            if key in seen:
                continue
            seen.add(key)
            uniq.append((c, e))
        return uniq[: self.cfg.cause_effect_pairs]

    def _build_takeaways(self, sentences: List[str]) -> List[str]:
        """
        Build actionable bullets by:
        - Detecting imperative-like starts, or
        - Converting advisory language into imperatives.
        Includes a generic fallback pack if none found.
        """
        out: List[str] = []
        for s in sentences:
            line = s.strip().rstrip(".")
            low = line.lower()
            if re.match(r'^(identify|define|measure|decide|plan|set|create|collect|analyze|approach|practice|start|stop|reduce|increase)\b', low):
                out.append(self._title_case_first_word(line))
            elif any(v in low for v in ("should", "try to", "aim to", "consider", "recommend")):
                out.append(self._to_imperative(line))
            elif any(v in low for v in ("do ", "take ", "make ", "use ", "focus ")):
                out.append(self._title_case_first_word(line))
            if len(out) >= self.cfg.takeaways:
                break

        if not out:
            out = [
                "Identify the main goal and restate it in one sentence.",
                "Extract 5–10 key phrases and use them to tag the content.",
                "List 3 actions you can do in 24 hours that move the topic forward.",
                "Define simple success metrics and write them next to each action.",
                "Schedule the first action on your calendar today."
            ]
        return out[: self.cfg.takeaways]

    def _build_open_questions(self, core_points: List[str], themes: List[str], takeaways: List[str]) -> List[str]:
        """
        Turn content into reflective prompts (generic across topics).
        """
        qs: List[str] = []
        src = (core_points + themes)[:3]
        for s in src:
            qs.append(f"What is one concrete next step related to: “{textwrap.shorten(s, 80)}”?")
        if takeaways:
            qs.append(f"Which takeaway will you complete first, and what will ‘done’ look like?")
        # Ensure cap
        return qs[: self.cfg.open_questions]

    # ---------------------------- Helpers ----------------------------

    @staticmethod
    def _declutter(s: str) -> str:
        """Remove leading conjunctions, parentheticals, and compress whitespace."""
        s = re.sub(r'^\s*(and|but|so|well|okay|ok)[, ]+', '', s, flags=re.I)
        s = re.sub(r'\([^)]*\)', '', s)
        s = WHITESPACE_RE.sub(' ', s).strip()
        return s

    @staticmethod
    def _to_imperative(s: str) -> str:
        """Convert advisory phrasing into a crisp imperative (best-effort)."""
        s = re.sub(r'^\s*(you\s+should|you\s+can|we\s+should|try\s+to|aim\s+to|consider|it\s+is\s+recommended\s+to)\s+', '', s, flags=re.I)
        s = s[0:1].upper() + s[1:] if s else s
        return s or "Do one small, concrete action."

    @staticmethod
    def _title_case_first_word(s: str) -> str:
        if not s:
            return s
        return s[0:1].upper() + s[1:]


# =============================================================================
# ------------------------------  PUBLIC API  ---------------------------------
# =============================================================================

def summarize_text(transcript: str, config: Optional[SummarizerConfig] = None) -> StructuredSummary:
    """
    One-shot convenience function:
        summary = summarize_text(transcript)
    """
    return GenericTranscriptSummarizer(config=config).summarize_text(transcript)


# =============================================================================
# ---------------------------------  CLI  -------------------------------------
# =============================================================================

def _cli():
    """
    Minimal CLI so you can test locally:

    Examples:
      python generic_transcript_summarizer.py input.txt
      python generic_transcript_summarizer.py - --format both --out summary

    Use '-' for stdin.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Generic, domain-agnostic transcript summarizer (offline).")
    parser.add_argument("input", help="Path to UTF-8 .txt file, or '-' to read from stdin.")
    parser.add_argument("--format", choices=("md","json","both"), default="md",
                        help="Output format.")
    parser.add_argument("--out", help="Output path (without extension) to write files instead of stdout.")
    parser.add_argument("--tldr", type=int, default=None, help="TL;DR sentences (1-2 recommended).")
    parser.add_argument("--core", type=int, default=None, help="Number of Core Points.")
    parser.add_argument("--themes", type=int, default=None, help="Number of Themes bullets.")
    args = parser.parse_args()

    # Read input
    if args.input == "-":
        raw = "".join(iter(input, ""))
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            raw = f.read()

    cfg = SummarizerConfig()
    if args.tldr is not None:
        cfg.tldr_sentences = max(1, min(2, args.tldr))
    if args.core is not None:
        cfg.core_points = max(3, min(12, args.core))
    if args.themes is not None:
        cfg.themes = max(3, min(12, args.themes))

    summary = summarize_text(raw, config=cfg)
    md = summary.to_markdown()
    js = summary.to_json()

    if not args.out:
        if args.format in ("md","both"):
            print(md)
            if args.format == "both":
                print("\n" + "-"*80 + "\n")
        if args.format in ("json","both"):
            print(js)
    else:
        base = args.out
        wrote = []
        if args.format in ("md","both"):
            p = base + ".md"
            with open(p, "w", encoding="utf-8") as f:
                f.write(md)
            wrote.append(p)
        if args.format in ("json","both"):
            p = base + ".json"
            with open(p, "w", encoding="utf-8") as f:
                f.write(js)
            wrote.append(p)
        print("Wrote:", ", ".join(wrote))


if __name__ == "__main__":
    _cli()
