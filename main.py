from gpt import HF_LLM
from speech_brain_app import get_emotion
from transcription import Transcriptor
from mail import assess

class csi:
    def __init__(self, model_name, llm, tokenizer=None):
        self.gpt = HF_LLM(model_name, llm)
        self.transcriptor = Transcriptor()
        self.model = llm
        self.tokenizer = tokenizer

    def process(self, path):
        emotions = get_emotion(path)
        transcripts = self.transcriptor.transcribe(path)
        assess(transcripts, self.model, self.tokenizer)
        return self.gpt.generate_response(transcripts, emotions)

    def process_return_with_transcripts(self, audio):
        emotions = get_emotion(audio)
        transcripts = self.transcriptor.transcribe(audio)
        assess(transcripts, self.model, self.tokenizer)
        return self.gpt.generate_response(transcripts, emotions), transcripts