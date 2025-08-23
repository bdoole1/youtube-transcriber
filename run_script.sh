#!/bin/bash
# Wrapper to run the YouTube Transcript Generator
# Usage: ./run_script.sh <youtube_url> [additional_flags]

if [ $# -lt 1 ]; then
  echo "Usage: $0 <youtube_url> [options]"
  echo "Example: $0 https://youtube.com/watch?v=abcd1234 --summarize"
  exit 1
fi

YOUTUBE_URL=$1
shift  # remove first arg so any extra flags get passed through

# Activate virtual environment if you use one
source ~/myenv/bin/activate
# source venv/bin/activate

python3 main.py "$YOUTUBE_URL" "$@"

