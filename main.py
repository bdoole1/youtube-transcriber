# required install:
# pip install git+https://github.com/openai/whisper.git    # large download
# pip install torch  # use the CUDA build if you want GPU acceleration
# pip install yt-dlp # robust YouTube downloader
# pip install librosa transformers tqdm
# sudo apt install ffmpeg

import time
import os
import re
import argparse
import math
import torch
import yt_dlp
import whisper
import librosa  # audio loading
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from generic_transcript_summarizer import (
    summarize_text as structured_summarize_text,
    SummarizerConfig as StructuredSummarizerConfig,
)

# optional: tqdm progress bar, but keep a fallback so script still runs
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

# -----------------------------
# YouTube download
# -----------------------------

def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube and convert to mp3 using yt-dlp, with a tqdm progress bar."""
    from tqdm import tqdm

    pbar = {"bar": None}

    def progress_hook(d):
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)

            if total:
                # create bar on first update
                if pbar["bar"] is None:
                    pbar["bar"] = tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc="Downloading audio",
                    )
                pbar["bar"].n = downloaded
                pbar["bar"].refresh()

        elif status == "finished":
            # download finished, close bar
            if pbar["bar"] is not None:
                pbar["bar"].n = pbar["bar"].total or pbar["bar"].n
                pbar["bar"].close()

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'keepvideo': False,
        "progress_hooks": [progress_hook],
        'noprogress': True,
    }

    print(f"[INFO] Downloading audio from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # print(f"[INFO] Downloading audio from: {url}")
        info = ydl.extract_info(url, download=True)
        downloaded = ydl.prepare_filename(info)  # e.g. "...webm" or "...m4a" before postprocess
        audio_filename = os.path.splitext(downloaded)[0] + ".mp3"

    if not os.path.exists(audio_filename):
        raise FileNotFoundError(f"[ERROR] File '{audio_filename}' was not created.")

    print(f"[INFO] Audio file saved as: {audio_filename}")
    return audio_filename

# -----------------------------
# Irish-specific ASR (wav2vec2)
# -----------------------------

IRISH_ASR_MODEL_ID = "Aditya3107/wav2vec2-large-xls-r-1b-ga-ie"

_irish_asr_model = None
_irish_asr_processor = None


# -----------------------------
# Irish-specific ASR (wav2vec2)
# -----------------------------

IRISH_ASR_MODEL_ID = "Aditya3107/wav2vec2-large-xls-r-1b-ga-ie"

_irish_asr_model = None
_irish_asr_processor = None


def load_irish_asr(device: str):
    """
    Lazy-load the Irish wav2vec2 model + processor on the given device.
    """
    global _irish_asr_model, _irish_asr_processor
    if _irish_asr_model is None or _irish_asr_processor is None:
        print(f"[INFO] Loading Irish ASR model '{IRISH_ASR_MODEL_ID}' on {device}")
        processor = Wav2Vec2Processor.from_pretrained(IRISH_ASR_MODEL_ID)
        model = Wav2Vec2ForCTC.from_pretrained(IRISH_ASR_MODEL_ID)
        model.to(device)
        model.eval()
        _irish_asr_model = model
        _irish_asr_processor = processor
    return _irish_asr_model, _irish_asr_processor


def transcribe_irish_wav2vec2(audio_path: str, device: str = "cpu") -> str:
    """
    High-accuracy Irish transcription using wav2vec2, with chunking so it fits in GPU RAM.

    - Uses smaller chunks on GPU to avoid OOM.
    - Uses a small overlap between chunks to reduce word-cutting at boundaries.
    - Drops a few words at the start of each chunk (except the first) to avoid repeats.
    """
    import numpy as np

    model, processor = load_irish_asr(device)

    # 1) Load audio as mono 16kHz
    target_sr = 16_000
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # 2) Chunk setup
    if device == "cuda":
        chunk_duration_s = 8   # shorter on GPU
    else:
        chunk_duration_s = 15  # a bit longer on CPU

    chunk_size = int(chunk_duration_s * target_sr)

    overlap_s = 0.5           # half-second overlap
    overlap = int(overlap_s * target_sr)

    # step = how far we move between chunks
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    min_chunk_samples = 4000  # ~0.25s; skip ultra-short tails

    n_samples = len(audio)
    total_chunks = math.ceil(n_samples / step)
    print(f"[INFO] Irish ASR: {n_samples} samples at {target_sr} Hz (~{n_samples/target_sr:.1f}s, ~{total_chunks} chunks)")

    texts: list[str] = []

    for idx, start in enumerate(
        tqdm(range(0, n_samples, step), total=total_chunks, desc="Irish ASR chunks")
    ):
        end = min(start + chunk_size, n_samples)
        chunk = audio[start:end]

        # Skip ultra-short trailing fragments that break conv1d
        if len(chunk) < min_chunk_samples:
            continue

        chunk = np.asarray(chunk, dtype="float32")

        inputs = processor(
            chunk,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

        with torch.no_grad():
            if device == "cuda":
                # NEW: use the recommended amp API to avoid the FutureWarning
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_values, attention_mask=attention_mask).logits
            else:
                logits = model(input_values, attention_mask=attention_mask).logits

        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0].strip()

        # Simple de-duplication for overlap:
        # drop a few leading words on all chunks except the first
        words = text.split()
        if idx > 0 and len(words) > 4:
            words = words[3:]
        cleaned = " ".join(words)
        texts.append(cleaned)

        if device == "cuda":
            torch.cuda.empty_cache()

    return " ".join(texts)



def transcribe_audio(
    audio_path: str,
    use_gpu: bool = True,
    model_size: str = "base",
    language: str | None = None,
) -> str:
    """Transcribe audio using Whisper or Irish wav2vec2 depending on language."""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] use_gpu={use_gpu}, device={device}")

    lang = (language or "").strip().lower()

    # If the user explicitly asked for Irish ("ga" / "ga-ie"), use wav2vec2.
    if lang.startswith("ga"):
        print("[INFO] Using Irish ASR model (wav2vec2) instead of Whisper")
        return transcribe_irish_wav2vec2(audio_path, device=device)

    # Otherwise fall back to Whisper (auto language or forced language)
    print(f"[INFO] Loading Whisper model '{model_size}' on device: {device}")
    model = whisper.load_model(model_size, device=device)

    kwargs = {"task": "transcribe"}
    if language is not None and lang != "auto":
        kwargs["language"] = language  # e.g. "en", "es", etc.

    print("[INFO] Transcribing with Whisper...")
    result = model.transcribe(audio_path, **kwargs)
    return result["text"]


# -----------------------------
# Summarization
# -----------------------------

_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being below between both
but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours  ourselves out over own same shan't
she she'd she'll she's should shouldn't so some such than that that's their theirs them themselves then there there's these they they'd
they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself
yourselves
""".split())

def _sentences(text: str):
    """Very basic sentence splitter."""
    parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]

def _extractive_summary(text: str, target_words: int = 200) -> str:
    """Lightweight frequency-based extractive summarizer (no extra deps)."""
    sents = _sentences(text)
    if not sents:
        return ""
    freq = {}
    for w in re.findall(r"[A-Za-z0-9']+", text.lower()):
        if w in _STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    scored = []
    for i, s in enumerate(sents):
        words = re.findall(r"[A-Za-z0-9']+", s.lower())
        score = sum(freq.get(w, 0) for w in words) / (len(words) + 1e-6)
        scored.append((score, i, s))
    scored.sort(reverse=True, key=lambda x: x[0])

    chosen, word_count = [], 0
    for _, _, s in scored:
        wc = len(s.split())
        chosen.append(s)
        word_count += wc
        if word_count >= target_words:
            break
    # preserve original order
    chosen_sorted = sorted(((i, s) for (_, i, s) in scored if s in chosen), key=lambda x: x[0])
    return " ".join(s for _, s in chosen_sorted)

def _maybe_abstractive_summary(text: str, target_words: int, use_gpu: bool):
    """Try transformers abstractive summarization if available; else return None."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception:
        return None

    model_name = "facebook/bart-large-cnn"
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading summarization model '{model_name}' on {device} (may download weights)")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        print(f"[WARN] Could not load transformers model: {e}")
        return None

    if device == "cuda":
        model = model.to("cuda")

    def chunks(chars: str, chunk_size: int = 3500):
        for i in range(0, len(chars), chunk_size):
            yield chars[i:i+chunk_size]

    summaries = []
    for chunk in chunks(text):
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )
        summaries.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    combined = " ".join(summaries)
    # Second-pass compression if very long
    if len(combined.split()) > target_words * 2:
        inputs = tokenizer(combined, return_tensors="pt", truncation=True, max_length=1024)
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )
        combined = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return combined

def summarize_text(text: str, use_gpu: bool, mode: str = "auto", target_words: int = 200) -> str:
    """
    Summarize transcript text.
    mode:
      - 'auto' (try abstractive w/ transformers if available; else extractive)
      - 'abstractive' (force transformers; fallback to extractive if not available)
      - 'extractive' (fast, no extra deps)
    """
    if mode not in {"auto", "abstractive", "extractive"}:
        mode = "auto"

    if mode in {"auto", "abstractive"}:
        abs_sum = _maybe_abstractive_summary(text, target_words, use_gpu)
        if abs_sum:
            return abs_sum

    # Fallback extractive
    return _extractive_summary(text, target_words=target_words)

# -----------------------------
# IO helpers
# -----------------------------

def save_text(text: str, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Saved: {filename}")

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="YouTube Whisper Transcript Tool")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--cpu", action="store_true", help="Force transcription to run on CPU")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep downloaded audio file")
    parser.add_argument("--output", type=str, default="transcript.txt", help="Transcript output file")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large)")
    # Summarization
    parser.add_argument("--summarize", action="store_true", help="Also generate a summary from the transcript")
    parser.add_argument("--summary-output", type=str, default="summary.txt", help="Summary output file")
    parser.add_argument("--summary-mode", type=str, choices=["auto", "abstractive", "extractive"], default="auto",
                        help="Summary mode: auto (try transformers), abstractive, or extractive")
    parser.add_argument("--summary-words", type=int, default=200, help="Target words for the summary (approximate)")
    # Structured summary (generic, Markdown + JSON)
    parser.add_argument("--structured", action="store_true", help="Generate a structured Markdown+JSON summary (generic_transcript_summarizer).")
    parser.add_argument("--structured-md", type=str, default="summary.md", help="Output path for structured Markdown summary (default: summary.md)")
    parser.add_argument("--structured-json", type=str, default="summary.json", help="Output path for structured JSON summary (default: summary.json)")
    parser.add_argument("--tldr-sentences", type=int, default=1, help="Structured summary: number of TL;DR sentences (1–2 typical).")
    parser.add_argument("--core-points", type=int, default=5, help="Structured summary: number of 'Core Points' bullets.")
    parser.add_argument("--themes", type=int, default=6, help="Structured summary: number of 'Themes' bullets.")
    parser.add_argument("--cause-effect", type=int, default=5, help="Structured summary: number of cause→effect pairs.")
    parser.add_argument("--takeaways", type=int, default=5, help="Structured summary: number of actionable takeaways.")
    parser.add_argument("--key-phrases", type=int, default=10, help="Structured summary: number of key phrases.")
    parser.add_argument("--open-questions", type=int, default=3, help="Structured summary: number of open questions.")
    parser.add_argument("--language", type=str, default="auto", help="Language code/name for Whisper (e.g. 'en', 'Spanish', or 'auto' for detection)")

    return parser.parse_args()

def main():
    start = time.time()  # ⏱️ START TIMER
    args = parse_args()

    # Step 1: Download audio
    t0 = time.time()
    audio_file = download_youtube_audio(args.url)
    t1 = time.time()
    print(f"[PERF] Download time: {t1 - t0:.1f} sec")

    # Step 2: Transcribe
    lang = None if args.language.lower() == "auto" else args.language
    t2 = time.time()
    transcript = transcribe_audio(
        audio_file,
        use_gpu=not args.cpu,
        model_size=args.model,
        language=lang,
    )
    t3 = time.time()
    print(f"[PERF] Transcription time: {t3 - t2:.1f} sec")

    # Step 3: Save transcript
    save_text(transcript, args.output)

    # Step 4: Optional summarization
    if args.summarize:
        print("[INFO] Generating summary...")
        t4 = time.time()
        summary = summarize_text(
            transcript,
            use_gpu=not args.cpu,
            mode=args.summary_mode,
            target_words=args.summary_words
        )
        save_text(summary, args.summary_output)
        t5 = time.time()
        print(f"[PERF] Summary time: {t5 - t4:.1f} sec")

    # Step 4b: Optional structured (generic) summary → Markdown + JSON
    if args.structured:
        print("[INFO] Generating structured (generic) summary...")
        cfg = StructuredSummarizerConfig(
            tldr_sentences=max(1, min(2, args.tldr_sentences)),
            core_points=max(3, args.core_points),
            themes=max(3, args.themes),
            cause_effect_pairs=max(0, args.cause_effect),
            takeaways=max(0, args.takeaways),
            key_phrases=max(0, args.key_phrases),
            open_questions=max(0, args.open_questions),
        )
        structured = structured_summarize_text(transcript, config=cfg)
        # Write Markdown
        save_text(structured.to_markdown(), args.structured_md)
        # Write JSON
        save_text(structured.to_json(), args.structured_json)

    # Step 5: Cleanup
    if not args.no_cleanup:
        try:
            os.remove(audio_file)
            print(f"[INFO] Deleted temporary file: {audio_file}")
        except OSError as e:
            print(f"[WARN] Could not delete {audio_file}: {e}")

    end = time.time()  # ⏱️ END TIMER
    duration = end - start

    # 🟢 Print nice human readable timing
    mins = duration // 60
    secs = duration % 60

    print(f"\n[PERF] Total runtime: {mins:.0f} min {secs:.1f} sec")

if __name__ == "__main__":
    main()
