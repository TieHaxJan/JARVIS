#!/bin/bash

# Ensure huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Install it with: pip install huggingface_hub"
    exit 1
fi

# Ensure Python 'datasets' is installed
if ! python3 -c "import datasets" &> /dev/null; then
    echo "Python package 'datasets' not found. Install it with: pip install datasets"
    exit 1
fi

# -----------------------------
# MODEL DOWNLOAD SECTION
# -----------------------------
models=(
    "nlpconnect/vit-gpt2-image-captioning"
    "lllyasviel/ControlNet"
    "lllyasviel/sd-controlnet-canny"
    "lllyasviel/sd-controlnet-depth"
    "lllyasviel/sd-controlnet-hed"
    "lllyasviel/sd-controlnet-mlsd"
    "lllyasviel/sd-controlnet-openpose"
    "lllyasviel/sd-controlnet-scribble"
    "lllyasviel/sd-controlnet-seg"
    "runwayml/stable-diffusion-v1-5"
    "damo-vilab/text-to-video-ms-1.7b"
    "microsoft/speecht5_asr"
    "JorisCos/DCCRNet_Libri1Mix_enhsingle_16k"
    "espnet/kan-bayashi_ljspeech_vits"
    "facebook/detr-resnet-101"
    "microsoft/speecht5_hifigan"
    "microsoft/speecht5_vc"
    "openai/whisper-base"
    "Intel/dpt-large"
    "facebook/detr-resnet-50-panoptic"
    "facebook/detr-resnet-50"
    "google/owlvit-base-patch32"
    "impira/layoutlm-document-qa"
    "ydshieh/vit-gpt2-coco-en"
    "dandelin/vilt-b32-finetuned-vqa"
    "lambdalabs/sd-image-variations-diffusers"
    "facebook/maskformer-swin-base-coco"
    "Intel/dpt-hybrid-midas"
)

echo "==================== DOWNLOADING MODELS ===================="
for model in "${models[@]}"; do
    model_dir="./${model}"
    echo ">> Downloading model: ${model}"
    huggingface-cli download "$model" --local-dir "$model_dir" --local-dir-use-symlinks False
done

# -----------------------------
# DATASET DOWNLOAD SECTION
# -----------------------------
datasets=("Matthijs/cmu-arctic-xvectors")

echo "==================== DOWNLOADING DATASETS ===================="
for dataset in "${datasets[@]}"; do
    dataset_dir="./${dataset}"
    echo ">> Downloading dataset: ${dataset}"

    python3 - <<EOF
from datasets import load_dataset

dataset = load_dataset("${dataset}")
dataset.save_to_disk("${dataset_dir}")
print("✅ Saved: ${dataset_dir}")
EOF

done

echo "✅ All downloads complete."
