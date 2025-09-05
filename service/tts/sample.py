import torch
import torchaudio
import os
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(script_dir, "assets", "female_sample.mp3")
wav, sampling_rate = torchaudio.load(audio_path)
speaker = model.make_speaker_embedding(wav, sampling_rate)

torch.manual_seed(421)

cond_dict = make_cond_dict(text="안녕하세요. 여기는 플래티어 AI 연구소입니다. 저희는 사전 학습된 모델을 이용하여 다양한 어플리케이션을 만들고 있어요.", speaker=speaker, language="ko")
conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
