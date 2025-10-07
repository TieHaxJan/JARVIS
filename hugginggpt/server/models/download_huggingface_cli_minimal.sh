#!/bin/bash

# Ensure huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Install it with: pip install huggingface_hub"
    exit 1
fi

# -----------------------------
# MODEL DOWNLOAD SECTION
# -----------------------------
models=(
    "nlpconnect/vit-gpt2-image-captioning"
    "lllyasviel/sd-controlnet-canny"
    "runwayml/stable-diffusion-v1-5"
    "openai/whisper-base"
    "Intel/dpt-large"
    "microsoft/swin-tiny-patch4-window7-224"
    "facebook/detr-resnet-50-panoptic"
    "lllyasviel/ControlNet"
)

echo "==================== DOWNLOADING MODELS ===================="
for model in "${models[@]}"; do
    model_dir="./${model}"
    echo ">> Downloading model: ${model}"
    hf download "$model" --local-dir "$model_dir"
done
