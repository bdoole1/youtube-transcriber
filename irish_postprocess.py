import re
from dataclasses import dataclass

@dataclass
class IrishPostprocessConfig:
    fix_broadcast_headers: bool = True
    fix_common_phrases: bool = True
    add_punctuation: bool = True

# --- 1) Known stock phrases / headers (high confidence) ---
# Keep these conservative: only fix patterns that are extremely likely.
# HEADER_FIXES = [
#     # "Cuid A" variations including glued forms
#     (r"\bcuid\s*([aAbB])\b", lambda m: f"Cuid {m.group(1).upper()}."),

#     # Special-case: "cuidafógra" / "cuid afógra" at start
#     (r"^\s*cuid\s*a\s*f[oó]gra\b", "Cuid A. Fógra"),
#     (r"^\s*cuid\s*a[fF][oó]gra\b", "Cuid A. Fógra"),
# ]
HEADER_FIXES = [
    (r"^\s*cuid\s*a\b", "Cuid A."),
    (r"^\s*cuida\b", "Cuid A."),
    (r"^\s*cuid\s*b\b", "Cuid B."),
    (r"^\s*cidb\b", "Cuid B."),

    # Handle glued "cuid afógra" / "cuidafógra"
    (r"^\s*cuid\s*a\s*f[oó]gra\b", "Cuid A. Fógra"),
    (r"^\s*cuid\s*a[fF][oó]gra\b", "Cuid A. Fógra"),
]

PHRASE_FIXES = [
    # ASR near-miss: "a héinseo" => "a haon. Seo"
    (r"\ba\s+h[ée]in\s*seo\b", "a haon. Seo"),
    (r"\ba\s+h[ée]inseo\b", "a haon. Seo"),
    # Your confirmed phrase
    (r"\bf[oó]gra\s+a\s+hae?in\s*seo\b", "Fógra a haon. Seo"),
    (r"\bf[oó]gra\s+a\s+hae?in\b", "Fógra a haon."),
    # RTÉ / RnaG boilerplate
    (r"\brt[eé]\s*raidio\s+na\s+gae?iltachta\b", "RTÉ Raidió na Gaeltachta"),
    (r"\braidio\s+na\s+gae?iltachta\b", "Raidió na Gaeltachta"),
    # "fógara" (plural error) -> "fógra"
    (r"\bf[oó]gara\b", "fógra"),

    # mangled "ó RTÉ"
    (r"\b[óo]\s+[áa]rt\s*[ée]\b", "ó RTÉ"),
    (r"\b[áa]rt\s*[ée]\b", "RTÉ"),

    # "gaeltacht" missing final 'a'
    (r"\bgae?iltacht\b", "Gaeltachta"),
    (r"\braidio\s+na\s+gae?iltachta\b", "Raidió na Gaeltachta"),

    # Ensure "Fógra a haon. Seo" capitalization if it appears
    (r"\bf[oó]gra\s+a\s+haon\.\s*seo\b", "Fógra a haon. Seo"),

]

# --- 2) Word-boundary repairs (very small + safe) ---
BOUNDARY_FIXES = [
    # "afógra" -> "a. fógra" when it follows "cuid"
    (r"(Cuid\s+[A-Z])\s*\.?\s*a[fF][oó]gra\b", r"\1. Fógra"),
    # "seo fógra" -> "Seo fógra"
    (r"\bseo\s+f[oó]gra\b", "Seo fógra"),
]

# --- 3) Optional punctuation heuristic ---
def _punctuate(text: str) -> str:
    # Ensure a space after periods when missing: "haon.Seo" -> "haon. Seo"
    text = re.sub(r"\.(\S)", r". \1", text)
    # Add line breaks after common broadcast sentence starters for readability
    text = re.sub(r"\.\s*(Seo|Tá|Is|Cuireann)\b", r".\n\1", text)
    return text

def postprocess_irish_transcript(text: str, cfg: IrishPostprocessConfig | None = None) -> str:
    cfg = cfg or IrishPostprocessConfig()
    out = text.strip()

    # Normalise whitespace early
    out = re.sub(r"\s+", " ", out)

    # Apply fixes (case-insensitive matching)
    def apply_fixes(rules, s):
        for pat, rep in rules:
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
        return s

    if cfg.fix_broadcast_headers:
        out = apply_fixes(HEADER_FIXES, out)

    if cfg.fix_common_phrases:
        out = apply_fixes(PHRASE_FIXES, out)

    out = apply_fixes(BOUNDARY_FIXES, out)

    if cfg.add_punctuation:
        out = _punctuate(out)

    return out.strip()
