#!/bin/bash
#SBATCH --job-name=extract_token
#SBATCH --partition=dev
#SBATCH --account=MST115022
#SBATCH --array=1-4%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=/work/u3937558/speech_tokenizers/CosyVoice/logs/resynth_%j.log

mkdir -p /work/u3937558/speech_tokenizers/CosyVoice/logs

bash -c 'export HF_HUB_CACHE=/work/u3937558/seedvc/seed-vc/checkpoints/hf_cache && \
  export HF_HUB_OFFLINE=1 && \
  export TRANSFORMERS_OFFLINE=1 && \
  unset SSL_CERT_FILE && \
  uv run /work/u3937558/speech_tokenizers/CosyVoice/extract_speech_token.py \
    /work/u3937558/seedvc/hf_converted_dataset_with_text/ \
    --cv_version 1'
