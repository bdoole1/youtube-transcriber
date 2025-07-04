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


def download_youtube_audio(url):

    # show what file was created
    print("Files in current directory:", os.listdir("."))
    ydl_opts = {
        # 'ffmpeg_location': '/usr/bin',
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True, # False to disable output to debug
        'keepvideo': False, # force overwrite and log errors
        'noprogress': False,
        'outtmpl': '%(title)s.%(ext)s', # dynamic naming
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"downloading audio from: {url}")
        # extract dynamically
        info = ydl.extract_info(url, download=True)
        output_filename = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
        # old
        # ydl.download([url])

    # show what file was created
    print("Files in current directory:", os.listdir("."))

    # error handling#
    print(f"Audio file saved as: {output_filename}")
    if not os.path.exists(output_filename):
        raise FileNotFoundError(f"{output_filename} was not created. yt-dlp may have failed.")

    print(f"Downloading audio from: {url}")

    return output_filename


def transcribe_audio(audio_path):
    print("loading whisper model...")
    model = whisper.load_model("base") # or tiny, small, medium, large
    print("Transcribing...")
    result = model.transcribe(audio_path)
    return result["text"]


def main():
    url = input("Enter youtube url: ")

    # step 1 download audio
    audio_file = download_youtube_audio(url)

    # show what file was created
    print("Files in current directory:", os.listdir("."))

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
