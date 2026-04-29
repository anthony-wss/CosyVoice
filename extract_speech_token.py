import sys
sys.path.append('third_party/Matcha-TTS')
from hyperpyyaml import load_hyperpyyaml
import os
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav
import whisper
import numpy as np
import torch

class CosyVoice3Tokenizer:
    def __init__(self):
        model_dir='pretrained_models/Fun-CosyVoice3-0.5B'
        hyper_yaml_path = '{}/cosyvoice3.yaml'.format(model_dir)
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v3.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        self.device = "cuda"
        del configs
    
    def extract_token(self, path):
        speech = load_wav(path, 16000)
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.frontend.speech_tokenizer_session.run(None,
                                                         {self.frontend.speech_tokenizer_session.get_inputs()[0].name:
                                                          feat.detach().cpu().numpy(),
                                                          self.frontend.speech_tokenizer_session.get_inputs()[1].name:
                                                          np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len


def main(args):
    tokenizer = CosyVoice3Tokenizer()
    speech_token, _ = tokenizer.extract_token("/home/anthony/CosyVoice/zero_shot_0.wav")
    speech_token = speech_token.tolist()

    import json
    with open("speech_tokens.json", "w") as f:
        json.dump({"speech_token": speech_token}, f)

if __name__ == "__main__":
    main({})