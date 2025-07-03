# pip install pytube
# pip install git+https://github.com/openai/whisper.git    # large download
# pip install torch # for whispers backend

# pip install yt-dlp # need to replace pytube

import torch
# print(torch.cuda.is_available())
# print(torch.version.cuda)             # Shows the version PyTorch was built with
# print(torch.backends.cudnn.version()) # Shows cuDNN version if available
# print(torch.cuda.device_count())      # Should be > 0 if working
# print(torch.cuda.get_device_name(0))  # Name of your GPU

import yt_dlp
import whisper
# from pytube import YouTube
import os

def download_youtube_audio(url, output_path="audio.mp3"):
    ydl_opts = {
        'ffmpeg_location': '/usr/bin',
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"downloading audio from: {url}")
        ydl.download([url])

    return output_path


def transcribe_audio(audio_path):
    print("loading whisper model...")
    model = whisper.load_model("base") # or tiny, small, medium, large
    print("Transcribing...")
    result = model.transcribe(audio_path)
    return result["text"]


def main():
    url = input("Enter youtube url: ")
    audio_file = "audio.mp3"
    
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
