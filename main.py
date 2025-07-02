# pip install pytube
# pip install git+https://github.com/openai/whisper.git    # large download
# pip install torch # for whispers backend

import torch
print(torch.cuda.is_available())

import whisper
from pytube import YouTube
import os

def download_youtube_audio(url, output_path="audio.mp4"):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    print(f"downloading audio: {yt.title}")
    audio_stream.download(filename=output_path)
    return output_path


def transcribe_audio(audio_path):
    print("loading whisper model...")
    model = whisper.load_model("base") # or tiny, small, medium, large
    print("Transcribing...")
    result = model.transcribe(audio_path)
    return result["text"]


def main():
    url = input("Enter youtube url: ")
    audio_file = "audio.mp4"
    
    # step 1 download audio
    download_youtube_audio(url, audio_file)

    # step 2 transcribe
    transcript = transcribe_audio(audio_file)

    # step 3 output
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)
    print("transcript saved to transcript.txt")

    # optional cleanup
    os.remove(audio_file)


if __name__ == "__main__":
    main()
