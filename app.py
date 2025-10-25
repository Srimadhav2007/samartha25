import io
import os
import numpy as np
import csv
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
from flask import Flask, request, render_template

app = Flask(__name__)

DEVICE = "cpu"  # or "cuda" if you have GPU
MODEL_SIZE = "large"  # can be "tiny", "base", "small", "medium", "large"

# --- Load Models with DISTINCT names ---
print(f"Loading Whisper model ({MODEL_SIZE}) onto {DEVICE}...")
try:
    whisper_model_loaded = whisper.load_model(MODEL_SIZE, device=DEVICE) # Use a distinct name
    print("✅ Whisper model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading Whisper model: {e}")
    whisper_model_loaded = None

yamnet_model_loaded = None # Initialize
yamnet_class_names = []  # Initialize

try:
    # Load the YAMNet model from TensorFlow Hub.
    yamnet_model_loaded = hub.load('https://tfhub.dev/google/yamnet/1') # Use a distinct name
    print("YAMNet model loaded successfully.")

    # Function to load class names from the CSV file provided by the model.
    def class_names_from_csv(class_map_csv_text):
        """Returns list of class names corresponding to score vector."""
        class_names = []
        with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names

    # Get the class map path from the model and load the names.
    class_map_path = yamnet_model_loaded.class_map_path().numpy().decode('utf-8')
    yamnet_class_names = class_names_from_csv(class_map_path) # Use a distinct name
    print(f"Loaded {len(yamnet_class_names)} YAMNet class names.")

except Exception as e:
    print(f"Error loading YAMNet model or class names: {e}")
    yamnet_model_loaded = None
    yamnet_class_names = []
# --- End Model Loading ---


# --- split_audio_on_silence function (Unchanged) ---
def split_audio_on_silence(
    audio_file_object,
    max_chunk_duration_ms=5000,
    min_silence_len_ms=700,
    silence_thresh_dbfs=-40,
    keep_silence_ms=300
    ):

    if audio_file_object is None:
        print("Error: No audio file object provided.")
        return []
    final_chunks_io = []

    try:
        try:
            audio = AudioSegment.from_file(audio_file_object)
            print(f"Audio loaded successfully. Duration: {len(audio) / 1000.0:.2f}s")
        except Exception as load_err:
            try:
                print(f"Initial load failed ({load_err}), trying explicit mp3 format...")
                if hasattr(audio_file_object, 'seek'):
                    audio_file_object.seek(0)
                audio = AudioSegment.from_file(audio_file_object, format="mp3")
                print(f"Audio loaded successfully as MP3. Duration: {len(audio) / 1000.0:.2f}s")
            except Exception as mp3_err:
                print(f"Failed to load audio as MP3 as well: {mp3_err}")
                raise

        segments = split_on_silence(
            audio,
            min_silence_len=min_silence_len_ms,
            silence_thresh=silence_thresh_dbfs,
            keep_silence=keep_silence_ms
        )

        if not segments:
            print("No speech segments found based on silence detection.")
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            buffer.seek(0)
            final_chunks_io.append(buffer)
            return final_chunks_io

        print(f"Split into {len(segments)} initial segments based on silence.")

        combined_segments = []
        current_chunk = None

        for segment in segments:
            if current_chunk is None:
                current_chunk = segment
            elif len(current_chunk) + len(segment) <= max_chunk_duration_ms:
                current_chunk += segment
            else:
                combined_segments.append(current_chunk)
                current_chunk = segment

        if current_chunk is not None:
            combined_segments.append(current_chunk)

        print(f"Combined into {len(combined_segments)} chunks aiming for max duration.")
        for i, chunk in enumerate(combined_segments):
            buffer = io.BytesIO()
            chunk.export(buffer, format="wav")
            # --- CRITICAL FIX: Seek before appending ---
            buffer.seek(0)
            # --- End Fix ---
            final_chunks_io.append(buffer)
            print(f"Exported chunk {i+1}, duration: {len(chunk) / 1000.0:.2f}s")

        return final_chunks_io

    except Exception as e:
        print(f"Error processing audio object in split function: {e}")
        # import traceback # Uncomment for debugging
        # print(traceback.format_exc()) # Uncomment for debugging
        return []

# --- processAudioFromBuffer function (Unchanged internally, uses passed model) ---
def processAudioFromBuffer(audio_buffer, loaded_whisper_model): # Pass the specific model
    # (Function content remains the same as your last version)
    if loaded_whisper_model is None:
        return None, "❌ Whisper model is not loaded.", None

    if audio_buffer is None:
        return None, "❌ No audio buffer provided.", None

    try:
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
        if not audio_bytes:
            return None, "❌ No audio data found in buffer.", None

        audio_np, original_sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)

        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)

        audio_np = audio_np.astype(np.float32)

        print("🎧 Starting transcription...")
        # Use the passed model
        result = loaded_whisper_model.transcribe(audio_np, fp16=(DEVICE == "cuda"))

        detected_language = result.get("language", "unknown")
        transcript = result.get("text", "").strip()

        print(f"🌐 Detected Language: {detected_language}")
        print(f"🗣️ Transcript: {transcript}")

        translation_text = transcript
        if detected_language and detected_language != "en":
            print("🌍 Translating...")
            try:
                # Use the passed model
                translate_result = loaded_whisper_model.transcribe(audio_np, task="translate", fp16=(DEVICE == "cuda"))
                translation_text = translate_result.get("text", "").strip()
                print(f"✅ Translation: {translation_text}")
            except Exception as translate_e:
                print(f"⚠️ Translation failed: {translate_e}")
        else:
            print("✅ No translation needed (already English or unknown).")

        return detected_language, transcript, translation_text

    except Exception as e:
        error_message = f"⚠️ Error processing audio buffer: {e}"
        print(error_message)
        return None, error_message, None

# --- classify_audio_sounds_from_buffer (Uses global yamnet_model_loaded) ---
def classify_audio_sounds_from_buffer(audio_buffer, top_n=5):
    # Use the distinct global names
    if yamnet_model_loaded is None or not yamnet_class_names:
        print("Error: YAMNet model or class names not loaded. Cannot classify.")
        return None # Return None if models aren't loaded

    if audio_buffer is None:
        print("Error: No audio buffer provided.")
        return []

    try:
        desired_sample_rate = 16000
        audio_buffer.seek(0)

        waveform, sample_rate = librosa.load(audio_buffer, sr=desired_sample_rate, mono=True)
        print(f"Audio loaded from buffer: duration={waveform.shape[0]/sample_rate:.2f}s, sample_rate={sample_rate}Hz")

        # Use the distinct YAMNet model variable
        scores, _, _ = yamnet_model_loaded(waveform)

        scores_np = scores.numpy()
        mean_scores = np.mean(scores_np, axis=0)
        top_class_indices = np.argsort(mean_scores)[::-1][:top_n]

        results = []
        for i in top_class_indices:
            # Use the distinct YAMNet class names variable
            sound_name = yamnet_class_names[i]
            score = mean_scores[i]
            results.append((sound_name, score))

        return results

    except Exception as e:
        print(f"Error processing audio buffer in YAMNet: {e}")
        # import traceback # Uncomment for debugging
        # print(traceback.format_exc()) # Uncomment for debugging
        return []


# --- Updated Flask Route ---
@app.route('/', methods=["GET","POST"])
def home():
    if request.method=="POST":
        audio_file = request.files.get("audio")
        if not audio_file or audio_file.filename == '':
            # Handle case where no file is uploaded
            return render_template("index.html", error="No audio file uploaded.")

        # --- Split the audio ---
        audio_chunks = split_audio_on_silence(audio_file)

        # --- Fix Initialization ---
        detected_language = "N/A" # Default value
        full_transcript = ""      # Accumulate transcript here
        full_translation = ""     # Accumulate translation here
        all_yamnet_results = []   # Store results for all chunks

        if not audio_chunks:
             # Handle case where splitting failed or returned empty
             return render_template("index.html", error="Could not process or split audio file.")

        # --- Process Chunks ---
        for i, chunk_buffer in enumerate(audio_chunks):
            print(f"\n--- Processing Chunk {i+1}/{len(audio_chunks)} ---")

            # Reset buffer position before each use
            chunk_buffer.seek(0)
            yamnet_result = classify_audio_sounds_from_buffer(chunk_buffer)
            if yamnet_result: # Check if None or empty list was returned
                 print("YAMNet results:", yamnet_result)
                 all_yamnet_results.append(yamnet_result) # Store results if needed
            else:
                 print("YAMNet failed for this chunk.")


            # Reset buffer position again before passing to Whisper
            chunk_buffer.seek(0)
            # --- Pass the loaded Whisper model explicitly ---
            whisper_result = processAudioFromBuffer(chunk_buffer, whisper_model_loaded)

            # --- Add Error Handling ---
            if whisper_result and whisper_result[0] is not None:
                lang, transcript_part, translation_part = whisper_result
                detected_language = lang # Update language (usually the same for all chunks)
                full_transcript += transcript_part + " " # Add space between parts
                full_translation += translation_part + " " # Add space between parts
            else:
                # Handle Whisper error for this chunk (e.g., log it)
                error_msg = whisper_result[1] if whisper_result else "Unknown Whisper error"
                print(f"Whisper processing failed for chunk {i+1}: {error_msg}")
                # Optionally add an error marker to the transcript/translation
                # full_transcript += f"[Error processing chunk {i+1}] "
                # full_translation += f"[Error processing chunk {i+1}] "


        # Pass accumulated results to template
        return render_template('index.html',
                               a=detected_language,
                               b=full_transcript.strip(),
                               c=full_translation.strip())

    # GET request
    return render_template("index.html")

# Needed to run directly (e.g., python app.py)
# if __name__ == '__main__':
#     app.run(debug=True) # Add debug=True for development if needed