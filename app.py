import streamlit as st
from pytube import YouTube
import requests
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import openai
import tempfile

# Set OpenAI API key from Streamlit secrets or environment variable
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")


def download_video(video_url, filename="video.mp4"):
    if "youtube.com" in video_url or "youtu.be" in video_url:
        yt = YouTube(video_url)
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by('resolution').desc().first()
        stream.download(filename=filename)
    else:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename


def extract_audio(video_path, audio_path="audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="pcm_s16le")
    return audio_path


def transcribe_audio(audio_path):
    # Using OpenAI Whisper API for transcription
    audio_file = open(audio_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    audio_file.close()
    return transcript["text"]


def analyze_accent(transcript_text):
    # Prompt engineering to detect accent and confidence
    prompt = f"""
You are a linguistic expert. Based on the following English transcript of a spoken passage, identify the accent of the speaker. Classify it as one of these: British, American, Australian, Canadian, Indian, or Other.
Also provide a confidence score (0 to 100%) that the speaker is a native English speaker with that accent.

Transcript:
\"\"\"{transcript_text}\"\"\"

Respond in JSON format:
{{
  "accent": "<accent classification>",
  "confidence": "<confidence score>",
  "summary": "<short explanation>"
}}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in English accents."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=150,
    )

    try:
        import json
        answer = response.choices[0].message.content.strip()
        data = json.loads(answer)
        return data
    except Exception as e:
        # Fallback if JSON parse fails
        return {
            "accent": "Unknown",
            "confidence": "0",
            "summary": "Could not parse model output."
        }


def main():
    st.title("Accent Detection Tool for English Speakers")
    st.write("Upload a public video URL (YouTube, Loom, or direct MP4 link) and get the speaker's English accent classification.")

    video_url = st.text_input("Enter video URL:")

    if st.button("Analyze Accent") and video_url:
        with st.spinner("Downloading video..."):
            try:
                video_path = download_video(video_url, "temp_video.mp4")
            except Exception as e:
                st.error(f"Failed to download video: {e}")
                return

        with st.spinner("Extracting audio..."):
            try:
                audio_path = extract_audio(video_path, "temp_audio.wav")
            except Exception as e:
                st.error(f"Failed to extract audio: {e}")
                return

        with st.spinner("Transcribing audio..."):
            try:
                transcript = transcribe_audio(audio_path)
            except Exception as e:
                st.error(f"Failed to transcribe audio: {e}")
                return

        st.subheader("Transcript:")
        st.write(transcript)

        with st.spinner("Analyzing accent..."):
            result = analyze_accent(transcript)

        st.subheader("Accent Classification Result")
        st.write(f"**Accent:** {result.get('accent', 'Unknown')}")
        st.write(f"**Confidence:** {result.get('confidence', '0')}%")
        st.write(f"**Summary:** {result.get('summary', '')}")

        # Clean up files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    main()
