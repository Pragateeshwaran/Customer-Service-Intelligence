from speechbrain.inference.diarization import Speech_Emotion_Diarization
import os

classifier = None

def get_emotion(audio):
    global classifier
    os.environ["HF_HUB_REQUEST_TIMEOUT"] = "60"
    if classifier is None:
        classifier = Speech_Emotion_Diarization.from_hparams(source="speechbrain/emotion-diarization-wavlm-large")
    return classifier.diarize_file(audio)   

# print(get_emotion("temp.wav"))