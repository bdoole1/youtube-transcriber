# YouTube Transcript Generator (with Whisper + yt-dlp + GPU Support)

This tool downloads audio from a YouTube video and transcribes it to text using [OpenAI Whisper](https://github.com/openai/whisper). It leverages your **NVIDIA GPU** (or CPU) for fast transcription and is designed for easy usage and reliable automation.

---

## 🔧 Features

- 🎙️ Downloads best-quality audio from YouTube
- 🧠 Uses OpenAI's Whisper for accurate transcription
- 🚀 GPU acceleration support via PyTorch + CUDA
- ✅ CLI interface for easy integration in scripts
- 🧼 Optional cleanup of downloaded audio
- 📦 Minimal dependencies, self-contained script

---

## ⚙️ Requirements

Ensure the following are installed on your system:

- Python 3.8+
- pip (Python package manager)
- FFmpeg (`sudo apt install ffmpeg`)
- NVIDIA GPU with CUDA (optional, for acceleration)

---

## 📦 Installation

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install git+https://github.com/openai/whisper.git
pip install torch  # or torch with CUDA if using GPU
pip install yt-dlp
sudo apt install ffmpeg

If you're on an NVIDIA GPU system, make sure the appropriate PyTorch with CUDA is installed:
Install guide

## 🚀 Usage
python3 main.py "https://www.youtube.com/xyz"
Optional arguments:
Flag	Description
--cpu	Forces transcription to run on CPU
--no-cleanup	Keeps the downloaded .mp3 audio file
--output FILE	Specifies a custom transcript output path

## Examples
# GPU transcription with cleanup
python3 main.py "https://youtube.com/xyz"

# Force CPU and keep the audio
python3 main.py "https://youtube.com/xyz" --cpu --no-cleanup

# Output to custom file
python3 main.py "https://youtube.com/xyz" --output my_transcript.txt

## 🧠 How It Works
Uses yt-dlp to download the best audio stream and convert it to .mp3 using FFmpeg.

Loads Whisper in either cuda or cpu mode.

Performs automatic speech recognition (ASR) on the audio file.

Writes the transcript to transcript.txt (or user-defined location).

Optionally deletes the audio file for cleanliness.

## 🪤 Common Pitfalls
Make sure ffmpeg is installed and on your PATH (ffmpeg -version).

If using GPU, ensure that torch with CUDA is installed and torch.cuda.is_available() returns True.

Some YouTube titles may contain special characters; we now handle this using yt-dlp's filename builder.

If the audio file isn't created, check if yt-dlp failed silently due to copyright or network issues.

## 🧭 Roadmap / Future Plans
 Add support for batch processing of multiple videos

 Automatic language detection and language override flag

 Add subtitle (.srt/.vtt) export option

 Add GUI frontend for desktop use

 Docker container for cross-platform deployment
