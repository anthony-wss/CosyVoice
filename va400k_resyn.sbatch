#!/bin/bash
#SBATCH --job-name=extract_token
#SBATCH --partition=dev
#SBATCH --account=MST115022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00
#SBATCH --exclude=25a-hgpn121
#SBATCH --output=/work/u3937558/speech_tokenizers/CosyVoice/logs/resynth_%j.log

mkdir -p /work/u3937558/speech_tokenizers/CosyVoice/logs

bash -c 'export HF_HUB_CACHE=/work/u3937558/seedvc/seed-vc/checkpoints/hf_cache && \
  export HF_HUB_OFFLINE=1 && \
  export TRANSFORMERS_OFFLINE=1 && \
  unset SSL_CERT_FILE && \
  uv run torchrun --nproc_per_node=8 va400k_resyn.py \
    worstchan/VoiceAssistant-400K-SLAM-Omni \
    -n -1'
