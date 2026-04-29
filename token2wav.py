import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav
import json
import uuid
import torch
import soundfile as sf

AUDIO_PROMPT_PATH="/home/anthony/CosyVoice/zero_shot_0.wav"
OUTPUT_PATH="/home/anthony/CosyVoice/output.wav"

def main():
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

    with open("speech_tokens.json", "r") as f:
        speech_token = json.load(f)["speech_token"]
        speech_token = torch.tensor(speech_token)
    flow_prompt_speech_token, _ = cosyvoice.frontend._extract_speech_token(AUDIO_PROMPT_PATH)
    prompt_speech_feat, _ = cosyvoice.frontend._extract_speech_feat(AUDIO_PROMPT_PATH)
    flow_embedding = cosyvoice.frontend._extract_spk_embedding(AUDIO_PROMPT_PATH)
    this_uuid = str(uuid.uuid1())
    speed = 1.0
    
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
    sf.write(OUTPUT_PATH, output_wav.squeeze().cpu().numpy(), 24000)

if __name__ == "__main__":
    main()