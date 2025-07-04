# required install:
# pip install git+https://github.com/openai/whisper.git    # large download
# pip install torch # for whispers backend
# pip install yt-dlp # needed to replace pytube
# sudo apt install ffmpeg

# print(torch.cuda.is_available())
# print(torch.version.cuda)             # Shows the version PyTorch was built with
# print(torch.backends.cudnn.version()) # Shows cuDNN version if available
# print(torch.cuda.device_count())      # Should be > 0 if working
# print(torch.cuda.get_device_name(0))  # Name of your GPU

import os
import argparse
import torch
import yt_dlp
import whisper


def download_youtube_audio(url: str) -> str:
    """ Download audio from YouTube and convert to mp3 using yt-dlp. """
    ydl_opts = {
        'ffmpeg_location': '/usr/bin',
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s', # dynamic naming
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True, # False to disable output to debug
        'keepvideo': False, # force overwrite and log errors
        'noprogress': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"[INFO] Downloading audio from: {url}")
        # extract dynamically
        info = ydl.extract_info(url, download=True)
        downloaded_filename = ydl.prepare_filename(info)
        audio_filename = os.path.splitext(downloaded_filename)[0] + ".mp3"

    # error handling
    print(f"Audio file saved as: {audio_filename}")
    if not os.path.exists(audio_filename):
        raise FileNotFoundError(f"[ERROR] File '{audio_filename}' was not created.")

    print(f"[INFO] Audio file saved as: {audio_filename}")

    return audio_filename


def transcribe_audio(audio_path: str, use_gpu: bool = True) -> str:
    """ Transcribe audio using OpenAI Whisper. """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading Whisper model on device: {device}")

    model = whisper.load_model("base", device=device) # or tiny, small, medium, large
    print("[INFO] Transcribing...")
    result = model.transcribe(audio_path)
    return result["text"]


def save_transcript(text: str, filename: str = "transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Transcript saved to {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="YouTube Whisper Transcript Tool")
    parser.add_argument("url", help="YouTube video url")
    parser.add_argument("--cpu", action="store_true", help="Force transcriptions to run on CPU")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep download audio file")
    parser.add_argument("--output", type=str, default="transcript.txt", help="Output transcript file")
    return parser.parse_args()


def main():
    args = parse_args()

    audio_file = download_youtube_audio(args.url)
    transcript = transcribe_audio(audio_file, use_gpu=not args.cpu)
    save_transcript(transcript, args.output)

    if not args.no_cleanup:
        os.remove(audio_file)
        print(f"[INFO] Deleted temporary file: {audio_file}")


if __name__ == "__main__":
    main()
