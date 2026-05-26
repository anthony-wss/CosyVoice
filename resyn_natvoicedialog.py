from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
from filelock import FileLock
import json
import os
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import soundfile as sf
import torchaudio.functional as F
import torch
import numpy as np

CHUNK_SIZE = 500
PROG_FILE = "./progress.json"
WORKER_ID = os.environ.get("WORKER_ID")
MODEL_REF_SPEECH = "/home/anthony/seedvc-research/ref_speech.mp3"
lock = FileLock(PROG_FILE + ".lock")

def get_next_chunk():
    chunk_to_process = None
    chunk_id = None
    with lock:
        with open(PROG_FILE, "r") as f:
            progress = json.load(f)
        for k in progress.keys():
            if progress[k].get("status", None) == "queued":
                progress[k]["status"] = "processing"
                progress[k]["worker"] = WORKER_ID
                chunk_id = k
                chunk_to_process = progress[k]
                break
        with open(PROG_FILE, "w") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    return chunk_id, chunk_to_process

def set_prog_done(chunk_id):
    with lock:
        with open(PROG_FILE, "r") as f:
            progress = json.load(f)
        progress[chunk_id]["status"] = "done"
        with open(PROG_FILE, "w") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

class CosyVoice3:
    def __init__(self):
        self.cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
        self.cosyvoice
        self.model_ref_24k, sr = sf.read(MODEL_REF_SPEECH)
        self.model_ref_24k = torch.from_numpy(self.model_ref_24k).float()
        if sr != 24000:
            self.model_ref_24k = F.resample(self.model_ref_24k, orig_freq=sr, new_freq=24000)
        self.model_ref_text = "Well, the Foundation as a whole should be kept still secret at this point in time."
    
    @torch.inference_mode()
    def zero_shot(self, ref_speech, ref_rate, ref_text, target_text):
        if isinstance(ref_speech, np.ndarray):
            ref_speech = torch.from_numpy(ref_speech).float()

        if ref_rate != 24000:
            ref_speech = F.resample(ref_speech, orig_freq=ref_rate, new_freq=24000)
        
        if ref_speech.ndim == 1:
            ref_speech = ref_speech.reshape([1, -1])

        generated_chunks = []
        for j in self.cosyvoice.inference_zero_shot(
            ref_text,
            f'You are a helpful assistant.<|endofprompt|>{target_text}',
            ref_speech, 
            stream=False
        ):
            generated_chunks.append(j['tts_speech'].cpu().numpy().flatten())
        
        if not generated_chunks:
            return {"array": np.array([]), "sampling_rate": 24000}
            
        final_audio = np.concatenate(generated_chunks)
        return {"array": final_audio, "sampling_rate": 24000}

cv3 = CosyVoice3()

def resyn_batch(batch):
    user_speeches = batch["user_inp_audio"]
    user_texts = batch["user_inp"]
    model_texts = batch["model_res"]
    
    user_speech_syn_list = []
    model_speech_syn_list = []
    
    for i in range(len(model_texts)):
        user_speech = user_speeches[i]["array"]
        user_speech_rate = user_speeches[i]["sampling_rate"]
        user_text = user_texts[i]["text"]
        model_text = model_texts[i]["text"]
        
        # Synthesize User Speech
        user_syn = cv3.zero_shot(user_speech, user_speech_rate, user_text, user_text)
        user_speech_syn_list.append(user_syn)
        
        # Synthesize Model Speech
        model_syn = cv3.zero_shot(cv3.model_ref_24k, 24000, cv3.model_ref_text, model_text)
        model_speech_syn_list.append(model_syn)
        
    batch["user_speech_syn"] = user_speech_syn_list
    batch["model_speech_syn"] = model_speech_syn_list
    return batch

if __name__ == "__main__":
    ds_audio = load_from_disk("/home/anthony/podcast_dialogue_dataset/hf_dataset")
    ds_text  = load_from_disk("/home/anthony/podcast_dialogue_dataset/hf_dataset_text_only")

    ds = concatenate_datasets([ds_audio, ds_text], axis=1)
    
    os.makedirs("./resyn_natvoicedialog_chunk_tmp/", exist_ok=True)

    chunk_id, chunk = get_next_chunk()
    while chunk is not None:
        ds_sub = ds.select(range(chunk["start"], chunk["end"]))
        ds_sub = ds_sub.map(resyn_batch, batched=True, batch_size=32)
        ds_sub.save_to_disk(f"./resyn_natvoicedialog_chunk_tmp/chunk_{chunk_id}")
        set_prog_done(chunk_id)
        chunk_id, chunk = get_next_chunk()
