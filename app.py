from flask import Flask, render_template, request
import whisper
import numpy as np
import torch
import soundfile as sf
import io

app = Flask(__name__)

"""def processAudio(audio_path,device = "cpu"):
    model = whisper.load_model("large", device=device)
    result = model.transcribe(audio_path)
    detected_language, transcript = result["language"],result["text"]
    translate = model.transcribe(audio_path, task="translate")
    return detected_language,transcript,translate"""

DEVICE ="cpu"
MODEL_SIZE = "large" # Or choose "base", "small", "medium", "tiny"
print(f"Loading Whisper model ({MODEL_SIZE}) onto {DEVICE}...")
# Ensure the model is loaded outside the request-handling function
try:
    model = whisper.load_model(MODEL_SIZE, device=DEVICE)
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    # Handle the error appropriately, maybe exit or use a fallback
    model = None

# --- Modified function to process the uploaded file object ---
def processAudioFromUpload(audio_file_object, loaded_model):
    if loaded_model is None:
        return None, "Whisper model is not loaded.", None

    try:
        audio_bytes = audio_file_object.read()
        audio_np, original_sr = sf.read(io.BytesIO(audio_bytes))

        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)
        audio_np = audio_np.astype(np.float32)
        print("Starting transcription...")
        result = loaded_model.transcribe(audio_np, fp16=(DEVICE == "cuda"))
        detected_language = result["language"]
        transcript = result["text"]
        print(f"Detected Language: {detected_language}")
        print(f"Transcript: {transcript}")
        translate_result = loaded_model.transcribe(audio_np, task="translate", fp16=(DEVICE == "cuda"))
        translation_text = translate_result["text"]
        print(f"Translation: {translation_text}")

        return detected_language, transcript, translation_text
    
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None, f"Error processing audio: {str(e)}", None

@app.route('/', methods=["GET","POST"])
def home():
    if request.method=="POST":
        detected_language, transcript, translation_text =processAudioFromUpload(request.files.get('audio'),model)
        if detected_language is None:
            return render_template('index.html', error=transcript)
        return render_template('index.html',a=detected_language,b=transcript,c=translation_text)
    return render_template("index.html")

@app.route('/process', methods=["POST"])
def process():
    return 0