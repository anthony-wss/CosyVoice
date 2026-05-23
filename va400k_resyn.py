import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('../WavTokenizer')
import argparse
from datasets import load_dataset
from extract_speech_token import CosyVoice3Tokenizer
from cosyvoice.cli.cosyvoice import AutoModel
import uuid
import torch

from encoder.utils import convert_audio
import torchaudio
from decoder.pretrained import WavTokenizer

AUDIO_PROMPT_PATH="/work/u3937558/seedvc/ref_speech.mp3"

def main(args):
    ds = load_dataset(args.source_dataset)
    ds = ds["train"]

    if args.n != -1:
        assert 1 <= args.n and args.n <= len(ds)
        ds = ds.select(range(args.n))
    
    cosyvoice1 = AutoModel(model_dir='pretrained_models/CosyVoice-300M')
    flow_prompt_speech_token, _ = cosyvoice1.frontend._extract_speech_token(AUDIO_PROMPT_PATH)
    prompt_speech_feat, _ = cosyvoice1.frontend._extract_speech_feat(AUDIO_PROMPT_PATH)
    flow_embedding = cosyvoice1.frontend._extract_spk_embedding(AUDIO_PROMPT_PATH)
    speed = 1.0

    cv3_tokenizer = CosyVoice3Tokenizer()

    wavtokenizer40 = WavTokenizer.from_pretrained0802(
        "../WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        "/home/u3937558/.cache/huggingface/hub/models--novateur--WavTokenizer-large-unify-40token/blobs/72182c1b6bd5ea7f84cf3ec78a0a3244cf42daa660b2e9bce23f5d74064d8205"
    )
    wavtokenizer40 = wavtokenizer40.to("cuda")

    wavtokenizer75 = WavTokenizer.from_pretrained0802(
        "../WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        "/home/u3937558/.cache/huggingface/hub/models--novateur--WavTokenizer-large-speech-75token/blobs/5dd430d5f0d96e2313babb1b896d0f990b65bfd143b2894fc3851cfc3cca9846"
    )
    wavtokenizer75 = wavtokenizer75.to("cuda")

    
    def change_token(sample):
        speech_token = sample["answer_cosyvoice_speech_token"]
        speech_token = torch.tensor([speech_token])
        this_uuid = str(uuid.uuid1())
        with cosyvoice1.model.lock:
            cosyvoice1.model.tts_speech_token_dict[this_uuid], cosyvoice1.model.llm_end_dict[this_uuid] = [], False
            cosyvoice1.model.hift_cache_dict[this_uuid] = None
            cosyvoice1.model.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            cosyvoice1.model.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        output_wav = cosyvoice1.model.token2wav(
            token=speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            uuid=this_uuid,
            finalize=True,
            speed=speed
        )
        
        cv3_tokens = cv3_tokenizer.extract_token(wave=output_wav.squeeze().cpu().numpy(), sr=24000)
        sample["model_res_token_cv3"] = cv3_tokens[0]
        del sample["answer_cosyvoice_speech_token"]

        bandwidth_id = torch.tensor([0])
        output_wav = output_wav.to("cuda")

        _, wavtok_40_tokens = wavtokenizer40.encode_infer(output_wav, bandwidth_id=bandwidth_id)
        _, wavtok_75_tokens = wavtokenizer75.encode_infer(output_wav, bandwidth_id=bandwidth_id)
        sample["model_res_token_wavtok40"] = wavtok_40_tokens
        sample["model_res_token_wavtok75"] = wavtok_75_tokens

        return sample
    
    ds = ds.map(change_token)
    ds.save_to_disk("_debug_va400k_cv3_subset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dataset", help="Path to the VA400k dataset", type=str)
    parser.add_argument("-n", help="Number of samples to process", type=int, default=-1)
    args = parser.parse_args()

    main(args)
