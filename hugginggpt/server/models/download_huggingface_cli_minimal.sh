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
    "lllyasviel/sd-controlnet-hed"
    "lllyasviel/sd-controlnet-openpose"
    "lllyasviel/sd-controlnet-scribble"
    "runwayml/stable-diffusion-v1-5"
    "damo-vilab/text-to-video-ms-1.7b"
    "microsoft/speecht5_asr"
    "microsoft/speecht5_tts"
    "microsoft/speecht5_hifigan"
    "facebook/detr-resnet-101"
    "openai/whisper-base"
    "impira/layoutlm-document-qa"
    "dandelin/vilt-b32-finetuned-vqa"
    "lambdalabs/sd-image-variations-diffusers"
    "facebook/maskformer-swin-base-coco"
    "Intel/dpt-hybrid-midas"
    "microsoft/trocr-base-printed"
    "dslim/distilbert-NER"
)

echo "==================== DOWNLOADING MODELS ===================="
for model in "${models[@]}"; do
    model_dir="./${model}"
    echo ">> Downloading model: ${model}"
    hf download "$model" --local-dir "$model_dir"
done
