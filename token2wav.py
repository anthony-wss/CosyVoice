import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav
import json
import uuid
import torch
import soundfile as sf
import argparse

# AUDIO_PROMPT_PATH="/home/anthony/CosyVoice/zero_shot_0.wav"
# OUTPUT_PATH="/home/anthony/CosyVoice/output.wav"
AUDIO_PROMPT_PATH="/work/u3937558/seedvc/ref_speech.mp3"
OUTPUT_PATH="./output.wav"

def main(args):
    if args.cv_version == "1":
        cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M')
    elif args.cv_version == "2":
        raise NotImplementedError("We do not implement CosyVoice2 yet")
    elif args.cv_version == "3":
        cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    else:
        raise ValueError(f"cv_version should be 1, 2, or 3")

    with open("speech_tokens_cv1.json", "r") as f:
        speech_token = json.load(f)["speech_token"]
        speech_token = torch.tensor(speech_token)

    flow_prompt_speech_token, _ = cosyvoice.frontend._extract_speech_token(AUDIO_PROMPT_PATH)
    prompt_speech_feat, _ = cosyvoice.frontend._extract_speech_feat(AUDIO_PROMPT_PATH)
    flow_embedding = cosyvoice.frontend._extract_spk_embedding(AUDIO_PROMPT_PATH)
    this_uuid = str(uuid.uuid1())
    speed = 1.0
    
    if args.cv_version == "1":
        with cosyvoice.model.lock:
            cosyvoice.model.tts_speech_token_dict[this_uuid], cosyvoice.model.llm_end_dict[this_uuid] = [], False
            cosyvoice.model.hift_cache_dict[this_uuid] = None
            cosyvoice.model.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            cosyvoice.model.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        output_wav = cosyvoice.model.token2wav(
            token=speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            uuid=this_uuid,
            finalize=True,
            speed=speed
        )
    elif args.cv_version == "2":
        raise NotImplementedError("We do not implement CosyVoice2 yet")
    elif args.cv_version == "3":
        output_wav = cosyvoice.model.token2wav(
            token=speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            token_offset=0,
            uuid=this_uuid,
            stream=False,
            finalize=True,
            speed=speed
        )
    else:
        raise ValueError(f"cv_version should be 1, 2, or 3")
    
    sf.write(OUTPUT_PATH, output_wav.squeeze().cpu().numpy(), 24000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_version", help="CosyVoice version. Should be 1, 2, or 3", type=str)
    args = parser.parse_args()
    main(args)