import librosa
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor

tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")
audio , sample_rate = librosa.load(r"assests\speech.mp3",sr=16000)
input_values = tokenizer(audio,return_tensors = 'pt',padding=True).input_values
logits = model(input_values).logits
pre_ids = torch.argmax(logits,dim=-1)
trans = tokenizer.decode(pre_ids[0])
print(trans)
