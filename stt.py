import whisper

# Force CPU
device = "cpu"
print(f"Using device: {device}")

# Load the large model
model = whisper.load_model("large", device=device)

# Path to your audio file
audio_path = "/media/srimadhav/New Volume/Ubuntu_Data/samartha25/english/cv-corpus-22.0-delta-2025-06-20/en/clips/common_voice_en_42696072.mp3"

# Transcribe (Whisper handles language detection internally)
result = model.transcribe(audio_path)
print("Detected language:", result["language"])
print("Transcript:", result["text"])

# Translate to English
result_translate = model.transcribe(audio_path, task="translate")
print("Translated transcript:", result_translate["text"])

print('\n---------------------------------------------------------------------------------------------------------------------\n')


audio_path = "/media/srimadhav/New Volume/Ubuntu_Data/samartha25/arabic/cv-corpus-21.0-delta-2025-03-14/ar/clips/common_voice_ar_41921285.mp3"

# Transcribe (Whisper handles language detection internally)
result = model.transcribe(audio_path)
print("Detected language:", result["language"])
print("Transcript:", result["text"])

# Translate to English
result_translate = model.transcribe(audio_path, task="translate")
print("Translated transcript:", result_translate["text"])

print('\n---------------------------------------------------------------------------------------------------------------------\n')


audio_path = "/media/srimadhav/New Volume/Ubuntu_Data/samartha25/chinese/cv-corpus-20.0-delta-2024-12-06/zh-CN/clips/common_voice_zh-CN_41297646.mp3"

# Transcribe (Whisper handles language detection internally)
result = model.transcribe(audio_path)
print("Detected language:", result["language"])
print("Transcript:", result["text"])

# Translate to English
result_translate = model.transcribe(audio_path, task="translate")
print("Translated transcript:", result_translate["text"])

print('\n---------------------------------------------------------------------------------------------------------------------\n')


audio_path = "/media/srimadhav/New Volume/Ubuntu_Data/samartha25/hebrew/cv-corpus-22.0-2025-06-20/he/clips/common_voice_he_38078769.mp3"

# Transcribe (Whisper handles language detection internally)
result = model.transcribe(audio_path)
print("Detected language:", result["language"])
print("Transcript:", result["text"])

# Translate to English
result_translate = model.transcribe(audio_path, task="translate")
print("Translated transcript:", result_translate["text"])

print('\n---------------------------------------------------------------------------------------------------------------------\n')


audio_path = "/media/srimadhav/New Volume/Ubuntu_Data/samartha25/hindi/cv-corpus-22.0-2025-06-20/hi/clips/common_voice_hi_23795238.mp3"

# Transcribe (Whisper handles language detection internally)
result = model.transcribe(audio_path)
print("Detected language:", result["language"])
print("Transcript:", result["text"])

# Translate to English
result_translate = model.transcribe(audio_path, task="translate")
print("Translated transcript:", result_translate["text"])

print('\n---------------------------------------------------------------------------------------------------------------------\n')


audio_path = "/media/srimadhav/New Volume/Ubuntu_Data/samartha25/indonesian/cv-corpus-20.0-delta-2024-12-06/id/clips/common_voice_id_41274035.mp3"

# Transcribe (Whisper handles language detection internally)
result = model.transcribe(audio_path)
print("Detected language:", result["language"])
print("Transcript:", result["text"])

# Translate to English
result_translate = model.transcribe(audio_path, task="translate")
print("Translated transcript:", result_translate["text"])

print('\n---------------------------------------------------------------------------------------------------------------------\n')


audio_path = "/media/srimadhav/New Volume/Ubuntu_Data/samartha25/telugu/cv-corpus-22.0-2025-06-20/te/clips/common_voice_te_38821715.mp3"

# Transcribe (Whisper handles language detection internally)
result = model.transcribe(audio_path)
print("Detected language:", result["language"])
print("Transcript:", result["text"])

# Translate to English
result_translate = model.transcribe(audio_path, task="translate")
print("Translated transcript:", result_translate["text"])

print('\n---------------------------------------------------------------------------------------------------------------------\n')

audio_path = "/media/srimadhav/New Volume/Ubuntu_Data/samartha25/thai/cv-corpus-22.0-delta-2025-06-20/th/clips/common_voice_th_42696553.mp3"

# Transcribe (Whisper handles language detection internally)
result = model.transcribe(audio_path)
print("Detected language:", result["language"])
print("Transcript:", result["text"])

# Translate to English
result_translate = model.transcribe(audio_path, task="translate")
print("Translated transcript:", result_translate["text"])

print('\n---------------------------------------------------------------------------------------------------------------------\n')
