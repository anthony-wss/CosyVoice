import sys
sys.path.append('third_party/Matcha-TTS')
from hyperpyyaml import load_hyperpyyaml
import os
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav
import whisper
import numpy as np
import torch
import argparse
import torchaudio
from datasets import load_from_disk

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
    
    def extract_token(self, path=None, wave=None, sr=None):
        if path:
            speech = load_wav(path, 16000)
        elif wave is not None:
            # We assume wave shape is [N] here
            assert sr, "you must provide the sampling rate"
            speech = torch.from_numpy(wave)
            if sr != 16000:
                assert sr >= 16000, f'wav sample rate {sr} must be greater than 16000'
                speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
            speech = speech.view([1, -1])
        else:
            raise ValueError("You must provide either path or wave data")

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


class CosyVoiceTokenizer:
    def __init__(self):
        model_dir='pretrained_models/CosyVoice-300M'
        hyper_yaml_path = '{}/cosyvoice.yaml'.format(model_dir)
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        self.device = "cuda"
        del configs
    
    def extract_token(self, path=None, wave=None, sr=None):
        if path:
            speech = load_wav(path, 16000)
        elif wave is not None:
            # We assume wave shape is [N] here
            assert sr, "you must provide the sampling rate"
            speech = torch.from_numpy(wave)
            if sr != 16000:
                assert sr >= 16000, f'wav sample rate {sr} must be greater than 16000'
                speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
            speech = speech.view([1, -1])
        else:
            raise ValueError("You must provide either path or wave data")

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
    ver = args.cv_version
    if ver == "1":
        tokenizer = CosyVoiceTokenizer()
    elif ver == "2":
        raise NotImplementedError("We do not implement CosyVoice2 yet")
    elif ver == "3":
        tokenizer = CosyVoice3Tokenizer()
    else:
        raise ValueError(f"cv_version should be 1, 2, or 3")

    ds = load_from_disk(args.hf_dataset)

    # Debug
    # ds = ds.select(range(5))

    def extract_CV_token(sample):
        if f"model_res_token_cv{ver}" in sample:
            return sample
        model_res_audio = sample["model_res_audio"]
        try:
            speech_token, _ = tokenizer.extract_token(wave=model_res_audio["array"], sr=model_res_audio["sampling_rate"])
            speech_token = speech_token.tolist()
        except Exception as e:
            print(f"Error while extracting CosyVoice {ver} token: {e}")
            speech_token = []
        sample[f"model_res_token_cv{ver}"] = speech_token
        return sample

    ds = ds.map(extract_CV_token)
    # ds.save_to_disk(args.hf_dataset)
    ds.save_to_disk("_debug_hf_dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_dataset", help="path to the source huggingface dataset", type=str)
    parser.add_argument("--cv_version", help="CosyVoice version. Should be 1, 2, or 3", type=str)
    args = parser.parse_args()
    main(args)
