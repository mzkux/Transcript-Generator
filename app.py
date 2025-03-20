import streamlit as st
import ffmpeg
import whisper
import tempfile
import os
import io

def extract_audio(video_bytes):
    """Extract audio from uploaded video bytes using ffmpeg"""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video.flush()
        
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio:
            try:
                (
                    ffmpeg
                    .input(temp_video.name)
                    .output(temp_audio.name, acodec='pcm_s16le', ac=1, ar='16000')
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
                return temp_audio.read()
            except ffmpeg.Error as e:
                st.error(f"FFmpeg Error: {e.stderr.decode()}")
                return None

def transcribe_audio(audio_bytes, model_name):
    """Transcribe audio bytes using Whisper without saving"""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()
        
        model = whisper.load_model(model_name)
        result = whisper.transcribe(model, temp_audio.name)
        return result["text"]

st.title("Video to Text Transcription")
st.write("Upload a video file and transcribe it to text using Whisper AI.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mkv"])
model_option = st.selectbox("Choose Whisper Model", ["tiny", "base", "small", "medium", "large"])

if uploaded_file:
    if st.button("Transcribe"):
        with st.spinner("Extracting audio..."):
            audio_bytes = extract_audio(uploaded_file.read())

        if audio_bytes:
            with st.spinner("Transcribing..."):
                transcript = transcribe_audio(audio_bytes, model_option)
            
            st.success("Transcription Completed!")
            st.write("### Transcription Output:")
            st.text_area("Transcript", transcript, height=300)

            transcript_io = io.BytesIO(transcript.encode("utf-8"))
            st.download_button("Download Transcript", transcript_io, file_name=f"{os.path.splitext(uploaded_file.name)[0]}.txt")
