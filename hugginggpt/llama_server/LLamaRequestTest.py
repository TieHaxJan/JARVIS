import requests

prompt = """
#1 Task Planning Stage: You are a helpful AI assistant. Based on the user's request, you will recommend the best model from Hugging Face for the task, and explain why.

You must respond in JSON format like:
{
    "id": "<model-id>",
    "reason": "<why this model is best suited>"
}
Allowed task types: ["token-classification", "text2text-generation", "summarization", "translation", "question-answering", "conversational", "text-generation", "sentence-similarity", "tabular-classification", "object-detection", "image-classification", "image-to-image", "image-to-text", "text-to-image", "text-to-video", "visual-question-answering", "document-question-answering", "image-segmentation", "depth-estimation", "text-to-speech", "automatic-speech-recognition", "audio-to-audio", "audio-classification", "canny-control", "hed-control", "mlsd-control", "normal-control", "openpose-control", "canny-text-to-image", "depth-text-to-image", "hed-text-to-image", "mlsd-text-to-image", "normal-text-to-image", "openpose-text-to-image", "seg-text-to-image"]
Available models: ["nlpconnect/vit-gpt2-image-captioning", "runwayml/stable-diffusion-v1-5", "damo-vilab/text-to-video-ms-1.7b", "microsoft/speecht5_asr", "JorisCos/DCCRNet_Libri1Mix_enhsingle_16k", "espnet/kan-bayashi_ljspeech_vits", "facebook/detr-resnet-101", "microsoft/speecht5_hifigan", "microsoft/speecht5_vc", "openai/whisper-base", "Intel/dpt-large", "facebook/detr-resnet-50-panoptic", "facebook/detr-resnet-50", "google/owlvit-base-patch32", "impira/layoutlm-document-qa", "ydshieh/vit-gpt2-coco-en", "dandelin/vilt-b32-finetuned-vqa", "lambdalabs/sd-image-variations-diffusers", "facebook/maskformer-swin-base-coco", "Intel/dpt-hybrid-midas"]

Consider model popularity, task suitability, performance, and availability.
Respond with the JSON entry and ONLY the JSON entry!

User: I want to count zebras in an image.

Assistant:
"""

res = requests.post("http://localhost:8010/v1/completions", json={
    "model": "llama-3.1-8b-instruct",
    "prompt": prompt,
    "max_tokens": 512,
    "temperature": 0,
    "stop": ["User:"]
})
print(res.json())