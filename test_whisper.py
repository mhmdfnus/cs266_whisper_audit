import whisper

model = whisper.load_model("turbo")
result = model.transcribe("/Users/mac/Desktop/cs266-final-project/oneminute.m4a")
print(result["text"])
