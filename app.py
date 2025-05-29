import streamlit as st
from pytube import YouTube
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import tempfile
import os
import openai

# Set OpenAI API key from Streamlit secrets or environment variable
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

st.title("🎤 English Accent Detection Tool")

st.markdown(
    """
    Enter a public video URL (YouTube, Loom, or direct MP4 link).
    The app will extract audio, transcribe it, and detect the English accent.
    """
)

video_url = st.text_input("Enter video URL")

def download_youtube_video(url, path):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by('resolution').desc().first()
    stream.download(output_path=path, filename="video.mp4")
    return os.path.join(path, "video.mp4")

def download_direct_mp4(url, path):
    import requests
    r = requests.get(url)
    file_path = os.path.join(path, "video.mp4")
    with open(file_path, 'wb') as f:
        f.write(r.content)
    return file_path

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".wav")
    video.audio.write_audiofile(audio_path, logger=None)
    video.close()
    return audio_path

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

def classify_accent(text):
    prompt = f"""
You are an expert linguist specialized in identifying English accents.

Based on the following transcript of a speaker, classify their English accent into one of these categories:
- British
- American
- Australian
- Canadian
- Irish
- Scottish
- Indian
- Other English Accent

Also provide a confidence score between 0 and 100% that the speaker is a native or fluent English speaker with that accent.

Transcript:
\"\"\"
{text}
\"\"\"

Return the response in JSON format:
{{
  "accent": "Accent name",
  "confidence": "Confidence score as percentage",
  "summary": "Brief explanation"
}}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an expert English accent classifier."},
                  {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    # Parse JSON from response safely
    import json
    try:
        answer_text = response['choices'][0]['message']['content'].strip()
        result = json.loads(answer_text)
    except Exception:
        # fallback if JSON parse fails, return raw text
        result = {"accent": "Unknown", "confidence": "0%", "summary": answer_text}
    return result

if st.button("Analyze Accent"):
    if not video_url:
        st.error("Please enter a video URL.")
    else:
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    if "youtube.com" in video_url or "youtu.be" in video_url:
                        video_path = download_youtube_video(video_url, tmp_dir)
                    elif video_url.endswith(".mp4"):
                        video_path = download_direct_mp4(video_url, tmp_dir)
                    else:
                        st.error("Unsupported video URL. Use YouTube or direct MP4 link.")
                        st.stop()

                    audio_path = extract_audio(video_path)
                    transcript = transcribe_audio(audio_path)

                    st.subheader("Transcript")
                    st.write(transcript)

                    classification = classify_accent(transcript)

                    st.subheader("Accent Classification")
                    st.write(f"**Accent:** {classification.get('accent','Unknown')}")
                    st.write(f"**Confidence:** {classification.get('confidence','0%')}")
                    st.write(f"**Summary:** {classification.get('summary','No summary available')}")
                except Exception as e:
                    st.error(f"Error processing video: {e}")
