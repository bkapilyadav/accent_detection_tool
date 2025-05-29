import streamlit as st
from pytube import YouTube
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import tempfile
import os
import requests
import openai

# Set your OpenAI API key via Streamlit Cloud secrets or environment variable
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

def download_video(url):
    if "youtube.com" in url or "youtu.be" in url:
        yt = YouTube(url)
        stream = yt.streams.filter(only_video=False, file_extension='mp4').first()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        stream.download(filename=temp_file.name)
        return temp_file.name
    else:
        # Assume direct mp4 url
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(temp_file.name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return temp_file.name
        else:
            st.error("Failed to download video. Please check the URL.")
            return None

def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    return audio_path

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

def analyze_accent(text):
    # Prompt engineering for accent classification + confidence score
    prompt = f"""
You are an expert in detecting English accents. Given the following transcript of a spoken English audio, classify the speaker's accent into one of these categories: British, American, Australian, Canadian, Indian, Irish, Scottish, or Other.

Also provide a confidence score between 0 to 100% indicating how confident you are that the accent is English and correctly classified.

Transcript:
\"\"\"
{text}
\"\"\"

Provide your answer in this JSON format only:

{{
  "accent": "<accent_name>",
  "confidence": <confidence_score_between_0_and_100>,
  "summary": "<brief explanation>"
}}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    import json
    try:
        content = response['choices'][0]['message']['content']
        result = json.loads(content)
        return result
    except Exception as e:
        return {"accent": "Unknown", "confidence": 0, "summary": f"Parsing error: {str(e)}"}

def main():
    st.title("REM Waste - English Accent Detection Tool 🎤")

    video_url = st.text_input("Enter public video URL (YouTube, Loom, or direct MP4):")

    if st.button("Analyze Accent") and video_url:
        with st.spinner("Downloading video..."):
            video_path = download_video(video_url)
        if video_path is None:
            st.error("Could not download video. Try a different URL.")
            return
        
        with st.spinner("Extracting audio..."):
            audio_path = extract_audio_from_video(video_path)
        
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(audio_path)
        
        st.subheader("Transcript")
        st.write(transcript)

        with st.spinner("Analyzing accent..."):
            result = analyze_accent(transcript)

        st.subheader("Accent Analysis Result")
        st.write(f"**Accent:** {result.get('accent', 'Unknown')}")
        st.write(f"**Confidence:** {result.get('confidence', 0)}%")
        st.write(f"**Summary:** {result.get('summary', 'No summary available')}")

        # Clean up temp files
        try:
            os.remove(video_path)
            os.remove(audio_path)
        except:
            pass

if __name__ == "__main__":
    main()
