import streamlit as st
from pytube import YouTube
import tempfile
from moviepy.editor import VideoFileClip
import os
from pydub import AudioSegment
import openai

# --- Helper functions ---

def download_video_youtube(url):
    yt = YouTube(url)
    stream = yt.streams.filter(only_video=False, file_extension='mp4').order_by('resolution').desc().first()
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    stream.download(output_path=os.path.dirname(temp_video_file.name), filename=os.path.basename(temp_video_file.name))
    return temp_video_file.name

def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    video.audio.write_audiofile(temp_audio_file.name, codec='pcm_s16le')
    video.close()
    return temp_audio_file.name

def transcribe_audio_openai(audio_file):
    audio_file_obj = open(audio_file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file_obj)
    audio_file_obj.close()
    return transcript["text"]

def analyze_accent(text):
    # Basic heuristic rules based on keywords / phrases
    accents = {
        "British": ["lorry", "biscuit", "petrol", "lift", "boot", "flat"],
        "American": ["truck", "cookie", "gasoline", "elevator", "trunk", "apartment"],
        "Australian": ["arvo", "servo", "brekkie", "mozzie"],
        "Indian": ["prepone", "timepass", "cousin-brother"],
    }

    scores = {}
    text_lower = text.lower()
    for accent, keywords in accents.items():
        count = sum(text_lower.count(word) for word in keywords)
        scores[accent] = count
    
    # Get highest score accent
    best_accent = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = (scores[best_accent] / total * 100) if total > 0 else 50  # 50% if unsure

    summary = f"Detected accent: {best_accent} with confidence {confidence:.1f}% based on keyword frequency."

    return best_accent, confidence, summary

# --- Streamlit UI ---

st.title("English Accent Detection Tool 🎤")

st.markdown("""
Upload a **YouTube URL** or a **direct MP4 video URL**.
The app will extract the audio, transcribe it, and classify the English accent.
""")

video_url = st.text_input("Enter YouTube or direct MP4 video URL")

if st.button("Analyze Accent"):
    if not video_url:
        st.error("Please enter a valid video URL.")
    else:
        with st.spinner("Downloading video..."):
            try:
                if "youtube.com" in video_url or "youtu.be" in video_url:
                    video_path = download_video_youtube(video_url)
                else:
                    st.error("Currently only YouTube URLs are supported. Direct MP4 support coming soon.")
                    st.stop()
            except Exception as e:
                st.error(f"Failed to download video: {e}")
                st.stop()

        with st.spinner("Extracting audio..."):
            try:
                audio_path = extract_audio_from_video(video_path)
            except Exception as e:
                st.error(f"Failed to extract audio: {e}")
                st.stop()

        with st.spinner("Transcribing audio with OpenAI Whisper..."):
            try:
                openai.api_key = st.secrets["OPENAI_API_KEY"]  # Put your OpenAI key in Streamlit secrets
                transcription = transcribe_audio_openai(audio_path)
                st.markdown("### Transcription:")
                st.write(transcription)
            except Exception as e:
                st.error(f"Failed to transcribe audio: {e}")
                st.stop()

        with st.spinner("Analyzing accent..."):
            accent, confidence, summary = analyze_accent(transcription)
            st.markdown("### Accent Classification Results:")
            st.write(f"**Accent:** {accent}")
            st.write(f"**Confidence Score:** {confidence:.1f}%")
            st.write(f"**Summary:** {summary}")

        # Cleanup temp files
        try:
            os.remove(video_path)
            os.remove(audio_path)
        except Exception:
            pass
