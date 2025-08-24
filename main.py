# required install:
# pip install git+https://github.com/openai/whisper.git    # large download
# pip install torch  # use the CUDA build if you want GPU acceleration
# pip install yt-dlp # robust YouTube downloader
# sudo apt install ffmpeg
# Optional (for abstractive summaries on GPU/CPU):
#   pip install transformers sentencepiece

import os
import re
import argparse
import torch
import yt_dlp
import whisper

# -----------------------------
# YouTube download
# -----------------------------

def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube and convert to mp3 using yt-dlp."""
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
        'noprogress': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"[INFO] Downloading audio from: {url}")
        info = ydl.extract_info(url, download=True)
        downloaded = ydl.prepare_filename(info)  # e.g. "...webm" or "...m4a" before postprocess
        audio_filename = os.path.splitext(downloaded)[0] + ".mp3"

    if not os.path.exists(audio_filename):
        raise FileNotFoundError(f"[ERROR] File '{audio_filename}' was not created.")

    print(f"[INFO] Audio file saved as: {audio_filename}")
    return audio_filename

# -----------------------------
# Transcription
# -----------------------------

def transcribe_audio(audio_path: str, use_gpu: bool = True, model_size: str = "base") -> str:
    """Transcribe audio using OpenAI Whisper."""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading Whisper model '{model_size}' on device: {device}")
    model = whisper.load_model(model_size, device=device)

    print("[INFO] Transcribing...")
    result = model.transcribe(audio_path)
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
    return parser.parse_args()

def main():
    args = parse_args()

    # Step 1: Download audio
    audio_file = download_youtube_audio(args.url)

    # Step 2: Transcribe
    transcript = transcribe_audio(audio_file, use_gpu=not args.cpu, model_size=args.model)

    # Step 3: Save transcript
    save_text(transcript, args.output)

    # Step 4: Optional summarization
    if args.summarize:
        print("[INFO] Generating summary...")
        summary = summarize_text(
            transcript,
            use_gpu=not args.cpu,
            mode=args.summary_mode,
            target_words=args.summary_words
        )
        save_text(summary, args.summary_output)

    # Step 5: Cleanup
    if not args.no_cleanup:
        try:
            os.remove(audio_file)
            print(f"[INFO] Deleted temporary file: {audio_file}")
        except OSError as e:
            print(f"[WARN] Could not delete {audio_file}: {e}")

if __name__ == "__main__":
    main()
