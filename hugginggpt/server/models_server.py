import argparse
import logging
import random
import uuid
import numpy as np
from transformers import pipeline
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5ForSpeechToSpeech
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import ViltProcessor
from transformers import BeitImageProcessor
from datasets import load_dataset, load_from_disk
from PIL import Image
import flask
from flask import request, jsonify
import waitress
from flask_cors import CORS
import io
import requests
import shutil
from io import BytesIO
from torchvision import transforms
import torch
import torchaudio
from speechbrain.pretrained import WaveformEnhancement
import joblib
from huggingface_hub import hf_hub_url, hf_hub_download
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation, AutoFeatureExtractor
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector, CannyDetector, MidasDetector
from controlnet_aux.open_pose.body import Body
from controlnet_aux.mlsd.models.mbv2_mlsd_large import MobileV2_MLSD_Large
from controlnet_aux.hed import Network
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import warnings
import time
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf
from asteroid.models import BaseModel
import traceback
import os
import yaml

def load_image(url_or_path):
    if url_or_path.startswith("http"):
        response = requests.get(url_or_path)
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(url_or_path).convert("RGB")


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config.default.yaml")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

# host = config["local_inference_endpoint"]["host"]
port = config["local_inference_endpoint"]["port"]

local_deployment = config["local_deployment"]
device = config.get("device", "cuda:0") 

PROXY = None
if config["proxy"]:
    PROXY = {
        "https": config["proxy"],
    }

app = flask.Flask(__name__)
CORS(app)

start = time.time()

local_fold = "models"
# if args.config.endswith(".dev"):
#     local_fold = "models_dev"


def load_pipes(local_deployment):
    other_pipes = {}
    standard_pipes = {}
    controlnet_sd_pipes = {}
    if local_deployment in ["full"]:
        other_pipes = {
            "nlpconnect/vit-gpt2-image-captioning":{
                "model": VisionEncoderDecoderModel.from_pretrained(f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "feature_extractor": ViTImageProcessor.from_pretrained(f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "tokenizer": AutoTokenizer.from_pretrained(f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "device": device
            },
            # "Salesforce/blip-image-captioning-large": {
            #     "model": BlipForConditionalGeneration.from_pretrained(f"{local_fold}/Salesforce/blip-image-captioning-large"),
            #     "processor": BlipProcessor.from_pretrained(f"{local_fold}/Salesforce/blip-image-captioning-large"),
            #     "device": device
            # },
            "damo-vilab/text-to-video-ms-1.7b": {
                "model": DiffusionPipeline.from_pretrained(f"{local_fold}/damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"),
                "device": device
            },
            # "facebook/maskformer-swin-large-ade": {
            #     "model": MaskFormerForInstanceSegmentation.from_pretrained(f"{local_fold}/facebook/maskformer-swin-large-ade"),
            #     "feature_extractor" : AutoFeatureExtractor.from_pretrained("facebook/maskformer-swin-large-ade"),
            #     "device": device
            # },
            # "microsoft/trocr-base-printed": {
            #     "processor": TrOCRProcessor.from_pretrained(f"{local_fold}/microsoft/trocr-base-printed"),
            #     "model": VisionEncoderDecoderModel.from_pretrained(f"{local_fold}/microsoft/trocr-base-printed"),
            #     "device": device
            # },
            # "microsoft/trocr-base-handwritten": {
            #     "processor": TrOCRProcessor.from_pretrained(f"{local_fold}/microsoft/trocr-base-handwritten"),
            #     "model": VisionEncoderDecoderModel.from_pretrained(f"{local_fold}/microsoft/trocr-base-handwritten"),
            #     "device": device
            # },
            "JorisCos/DCCRNet_Libri1Mix_enhsingle_16k": {
                "model": BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k"),
                "device": device
            },
            "espnet/kan-bayashi_ljspeech_vits": {
                "model": Text2Speech.from_pretrained(f"espnet/kan-bayashi_ljspeech_vits"),
                "device": device
            },
            "lambdalabs/sd-image-variations-diffusers": {
                "model": DiffusionPipeline.from_pretrained(f"{local_fold}/lambdalabs/sd-image-variations-diffusers"), #torch_dtype=torch.float16
                "device": device
            },
            # "CompVis/stable-diffusion-v1-4": {
            #     "model": DiffusionPipeline.from_pretrained(f"{local_fold}/CompVis/stable-diffusion-v1-4"),
            #     "device": device
            # },
            # "stabilityai/stable-diffusion-2-1": {
            #     "model": DiffusionPipeline.from_pretrained(f"{local_fold}/stabilityai/stable-diffusion-2-1"),
            #     "device": device
            # },
            "runwayml/stable-diffusion-v1-5": {
                "model": DiffusionPipeline.from_pretrained(f"{local_fold}/runwayml/stable-diffusion-v1-5"),
                "device": device
            },
            # "microsoft/speecht5_tts":{
            #     "processor": SpeechT5Processor.from_pretrained(f"{local_fold}/microsoft/speecht5_tts"),
            #     "model": SpeechT5ForTextToSpeech.from_pretrained(f"{local_fold}/microsoft/speecht5_tts"),
            #     "vocoder":  SpeechT5HifiGan.from_pretrained(f"{local_fold}/microsoft/speecht5_hifigan"),
            #     "embeddings_dataset": load_dataset(f"{local_fold}/Matthijs/cmu-arctic-xvectors", split="validation"),
            #     "device": device
            # },
            # "speechbrain/mtl-mimic-voicebank": {
            #     "model": WaveformEnhancement.from_hparams(source="speechbrain/mtl-mimic-voicebank", savedir="models/mtl-mimic-voicebank"),
            #     "device": device
            # },
            #"microsoft/speecht5_vc":{
            #    "processor": SpeechT5Processor.from_pretrained(f"{local_fold}/microsoft/speecht5_vc"),
            #    "model": SpeechT5ForSpeechToSpeech.from_pretrained(f"{local_fold}/microsoft/speecht5_vc"),
            #    "vocoder": SpeechT5HifiGan.from_pretrained(f"{local_fold}/microsoft/speecht5_hifigan"),
            #    "embeddings_dataset": load_dataset(f"{local_fold}/Matthijs/cmu-arctic-xvectors", split="validation"),
            #    "device": device
            #},
            # "julien-c/wine-quality": {
            #     "model": joblib.load(cached_download(hf_hub_url("julien-c/wine-quality", "sklearn_model.joblib")))
            # },
            # "facebook/timesformer-base-finetuned-k400": {
            #     "processor": AutoImageProcessor.from_pretrained(f"{local_fold}/facebook/timesformer-base-finetuned-k400"),
            #     "model": TimesformerForVideoClassification.from_pretrained(f"{local_fold}/facebook/timesformer-base-finetuned-k400"),
            #     "device": device
            # },
            "facebook/maskformer-swin-base-coco": {
                "feature_extractor": MaskFormerFeatureExtractor.from_pretrained(f"{local_fold}/facebook/maskformer-swin-base-coco"),
                "model": MaskFormerForInstanceSegmentation.from_pretrained(f"{local_fold}/facebook/maskformer-swin-base-coco"),
                "device": device
            },
            "Intel/dpt-hybrid-midas": {
                "model": DPTForDepthEstimation.from_pretrained(f"{local_fold}/Intel/dpt-hybrid-midas", low_cpu_mem_usage=True),
                "feature_extractor": DPTFeatureExtractor.from_pretrained(f"{local_fold}/Intel/dpt-hybrid-midas"),
                "device": device
            }
        }

    if local_deployment in ["full", "standard"]:
        standard_pipes = {
            # "superb/wav2vec2-base-superb-ks": {
            #     "model": pipeline(task="audio-classification", model=f"{local_fold}/superb/wav2vec2-base-superb-ks"), 
            #     "device": device
            # },
            "openai/whisper-base": {
                "model": pipeline(task="automatic-speech-recognition", model=f"{local_fold}/openai/whisper-base"), 
                "device": device
            },
            "microsoft/speecht5_asr": {
                "model": pipeline(task="automatic-speech-recognition", model=f"{local_fold}/microsoft/speecht5_asr"), 
                "device": device
            },
            "Intel/dpt-large": {
                "model": pipeline(task="depth-estimation", model=f"{local_fold}/Intel/dpt-large"), 
                "device": device
            },
            # "microsoft/beit-base-patch16-224-pt22k-ft22k": {
            #     "model": pipeline(task="image-classification", model=f"{local_fold}/microsoft/beit-base-patch16-224-pt22k-ft22k"), 
            #     "device": device
            # },
            "facebook/detr-resnet-50-panoptic": {
                "model": pipeline(task="image-segmentation", model=f"{local_fold}/facebook/detr-resnet-50-panoptic"), 
                "device": device
            },
            "facebook/detr-resnet-101": {
                "model": pipeline(task="object-detection", model=f"{local_fold}/facebook/detr-resnet-101"), 
                "device": device
            },
            # "openai/clip-vit-large-patch14": {
            #     "model": pipeline(task="zero-shot-image-classification", model=f"{local_fold}/openai/clip-vit-large-patch14"), 
            #     "device": device
            # },
            "google/owlvit-base-patch32": {
                "model": pipeline(task="zero-shot-object-detection", model=f"{local_fold}/google/owlvit-base-patch32"), 
                "device": device
            },
            # "microsoft/DialoGPT-medium": {
            #     "model": pipeline(task="conversational", model=f"{local_fold}/microsoft/DialoGPT-medium"), 
            #     "device": device
            # },
            # "bert-base-uncased": {
            #     "model": pipeline(task="fill-mask", model=f"{local_fold}/bert-base-uncased"), 
            #     "device": device
            # },
            # "deepset/roberta-base-squad2": {
            #     "model": pipeline(task = "question-answering", model=f"{local_fold}/deepset/roberta-base-squad2"), 
            #     "device": device
            # },
            # "facebook/bart-large-cnn": {
            #     "model": pipeline(task="summarization", model=f"{local_fold}/facebook/bart-large-cnn"), 
            #     "device": device
            # },
            # "google/tapas-base-finetuned-wtq": {
            #     "model": pipeline(task="table-question-answering", model=f"{local_fold}/google/tapas-base-finetuned-wtq"), 
            #     "device": device
            # },
            # "distilbert-base-uncased-finetuned-sst-2-english": {
            #     "model": pipeline(task="text-classification", model=f"{local_fold}/distilbert-base-uncased-finetuned-sst-2-english"), 
            #     "device": device
            # },
            # "gpt2": {
            #     "model": pipeline(task="text-generation", model="gpt2"), 
            #     "device": device
            # },
            # "mrm8488/t5-base-finetuned-question-generation-ap": {
            #     "model": pipeline(task="text2text-generation", model=f"{local_fold}/mrm8488/t5-base-finetuned-question-generation-ap"), 
            #     "device": device
            # },
            # "Jean-Baptiste/camembert-ner": {
            #     "model": pipeline(task="token-classification", model=f"{local_fold}/Jean-Baptiste/camembert-ner", aggregation_strategy="simple"), 
            #     "device": device
            # },
            # "t5-base": {
            #     "model": pipeline(task="translation", model=f"{local_fold}/t5-base"), 
            #     "device": device
            # },
            "impira/layoutlm-document-qa": {
                "model": pipeline(task="document-question-answering", model=f"{local_fold}/impira/layoutlm-document-qa"), 
                "device": device
            },
            "ydshieh/vit-gpt2-coco-en": {
                "model": pipeline(task="image-to-text", model=f"{local_fold}/ydshieh/vit-gpt2-coco-en"), 
                "device": device
            },
            "dandelin/vilt-b32-finetuned-vqa": {
                "model": pipeline(task="visual-question-answering", model=f"{local_fold}/dandelin/vilt-b32-finetuned-vqa"), 
                "device": device
            }
        }

    if local_deployment in ["full", "standard", "minimal"]:
        controlnet = ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        controlnetpipe = StableDiffusionControlNetPipeline.from_pretrained(
            f"{local_fold}/runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        )

        def mlsd_control_network():
            model = MobileV2_MLSD_Large()
            model.load_state_dict(torch.load(f"{local_fold}/lllyasviel/ControlNet/annotator/ckpts/mlsd_large_512_fp32.pth"), strict=True)
            return MLSDdetector(model)


        hed_network = Network(f"{local_fold}/lllyasviel/ControlNet/annotator/ckpts/network-bsds500.pth")

        controlnet_sd_pipes = {
            "openpose-control": {
                "model": OpenposeDetector(Body(f"{local_fold}/lllyasviel/ControlNet/annotator/ckpts/body_pose_model.pth"))
            },
            "mlsd-control": {
                "model": mlsd_control_network()
            },
            "hed-control": {
                "model": HEDdetector(hed_network)
            },
            "scribble-control": {
                "model": HEDdetector(hed_network)
            },
            "midas-control": {
                "model": MidasDetector(model_path=f"{local_fold}/lllyasviel/ControlNet/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt")
            },
            "canny-control": {
                "model": CannyDetector()
            },
            "lllyasviel/sd-controlnet-canny":{
                "control": controlnet, 
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-depth":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16),
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-hed":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16), 
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-mlsd":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16), 
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-openpose":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16), 
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-scribble":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16), 
                "model": controlnetpipe,
                "device": device
            },
            "lllyasviel/sd-controlnet-seg":{
                "control": ControlNetModel.from_pretrained(f"{local_fold}/lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16), 
                "model": controlnetpipe,
                "device": device
            }    
        }
        
    if local_deployment in ["thesis"]:
        # --- Vision, Audio, Text, ControlNet subset for your thesis ---
        standard_pipes = {
            # ASR
            "openai/whisper-base": {
                "model": pipeline(task="automatic-speech-recognition",
                                  model=f"{local_fold}/openai/whisper-base"),
                "device": device
            },
            "microsoft/speecht5_asr": {
                "model": pipeline(task="automatic-speech-recognition",
                                  model=f"{local_fold}/microsoft/speecht5_asr"),
                "device": device
            },
            # Object detection
            "facebook/detr-resnet-101": {
                "model": pipeline(task="object-detection",
                                  model=f"{local_fold}/facebook/detr-resnet-101", device=0),
                "device": device
            },
            # Document QA
            "impira/layoutlm-document-qa": {
                "model": pipeline(task="document-question-answering",
                                  model=f"{local_fold}/impira/layoutlm-document-qa"),
                "device": device
            },
            # Visual QA
            "dandelin/vilt-b32-finetuned-vqa": {
                "model": pipeline(
                    task="visual-question-answering",
                    model=f"{local_fold}/dandelin/vilt-b32-finetuned-vqa",
                    device=device
                ),
                "processor": ViltProcessor.from_pretrained(
                    f"{local_fold}/dandelin/vilt-b32-finetuned-vqa"
                ),
                "device": device
            },
            # Image segmentation
            "facebook/maskformer-swin-base-coco": {
                "feature_extractor": MaskFormerFeatureExtractor.from_pretrained(
                    f"{local_fold}/facebook/maskformer-swin-base-coco"),
                "model": MaskFormerForInstanceSegmentation.from_pretrained(
                    f"{local_fold}/facebook/maskformer-swin-base-coco"),
                "device": device
            },
            # Depth estimation
            "Intel/dpt-hybrid-midas": {
                "model": DPTForDepthEstimation.from_pretrained(
                    f"{local_fold}/Intel/dpt-hybrid-midas", low_cpu_mem_usage=True),
                "feature_extractor": DPTFeatureExtractor.from_pretrained(
                    f"{local_fold}/Intel/dpt-hybrid-midas"),
                "device": device
            },
            # Image-to-text
            "nlpconnect/vit-gpt2-image-captioning": {
                "model": VisionEncoderDecoderModel.from_pretrained(
                    f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "feature_extractor": ViTImageProcessor.from_pretrained(
                    f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "tokenizer": AutoTokenizer.from_pretrained(
                    f"{local_fold}/nlpconnect/vit-gpt2-image-captioning"),
                "device": device
            },
            # NER
            "dslim/distilbert-NER": {
                "model": pipeline(task="token-classification",
                                  model=f"{local_fold}/dslim/distilbert-NER",
                                  aggregation_strategy="simple"),
                "device": device
            },
            # Text-to-Speech
            "microsoft/speecht5_tts": {
                "processor": SpeechT5Processor.from_pretrained(
                    f"{local_fold}/microsoft/speecht5_tts"
                ),
                "model": SpeechT5ForTextToSpeech.from_pretrained(
                    f"{local_fold}/microsoft/speecht5_tts"
                ),
                "vocoder": SpeechT5HifiGan.from_pretrained(
                    f"{local_fold}/microsoft/speecht5_hifigan"
                ),
                "device": device
            },
            # image-classification
            "microsoft/beit-base-patch16-224-pt22k-ft22k": {
                "model": pipeline(
                    task="image-classification",
                    model=f"{local_fold}/microsoft/beit-base-patch16-224-pt22k-ft22k",
                    image_processor=BeitImageProcessor.from_pretrained(
                        f"{local_fold}/microsoft/beit-base-patch16-224-pt22k-ft22k"
                    ),
                    device=device
                ),
                "device": device
            },
        }

        other_pipes = {
            # Text-to-video
            "damo-vilab/text-to-video-ms-1.7b": {
                "model": DiffusionPipeline.from_pretrained(
                    f"{local_fold}/damo-vilab/text-to-video-ms-1.7b",
                    torch_dtype=torch.float16,
                    variant="fp16"),
                "device": device
            },
            # Image variations
            "lambdalabs/sd-image-variations-diffusers": {
                "model": DiffusionPipeline.from_pretrained(
                    f"{local_fold}/lambdalabs/sd-image-variations-diffusers"),
                "device": device
            },
            # Base Stable Diffusion
            "runwayml/stable-diffusion-v1-5": {
                "model": DiffusionPipeline.from_pretrained(
                    f"{local_fold}/runwayml/stable-diffusion-v1-5"),
                "device": device
            }
        }

        # --- Optional ControlNet support (no annotators needed) ---
        controlnet_sd_pipes = {
            "lllyasviel/sd-controlnet-canny": {
                "control": ControlNetModel.from_pretrained(
                    f"{local_fold}/lllyasviel/sd-controlnet-canny",
                    ),
                "model": StableDiffusionControlNetPipeline.from_pretrained(
                    f"{local_fold}/runwayml/stable-diffusion-v1-5",
                    controlnet=ControlNetModel.from_pretrained(
                        f"{local_fold}/lllyasviel/sd-controlnet-canny",
                        ),
                    ),
                "device": device
            },
            "lllyasviel/sd-controlnet-hed": {
                "control": ControlNetModel.from_pretrained(
                    f"{local_fold}/lllyasviel/sd-controlnet-hed",
                    ),
                "model": StableDiffusionControlNetPipeline.from_pretrained(
                    f"{local_fold}/runwayml/stable-diffusion-v1-5",
                    controlnet=ControlNetModel.from_pretrained(
                        f"{local_fold}/lllyasviel/sd-controlnet-hed",
                        ),
                    ),
                "device": device
            },
            "lllyasviel/sd-controlnet-openpose": {
                "control": ControlNetModel.from_pretrained(
                    f"{local_fold}/lllyasviel/sd-controlnet-openpose",
                    ),
                "model": StableDiffusionControlNetPipeline.from_pretrained(
                    f"{local_fold}/runwayml/stable-diffusion-v1-5",
                    controlnet=ControlNetModel.from_pretrained(
                        f"{local_fold}/lllyasviel/sd-controlnet-openpose",
                        ),
                    ),
                "device": device
            },
            "lllyasviel/sd-controlnet-scribble": {
                "control": ControlNetModel.from_pretrained(
                    f"{local_fold}/lllyasviel/sd-controlnet-scribble",
                    ),
                "model": StableDiffusionControlNetPipeline.from_pretrained(
                    f"{local_fold}/runwayml/stable-diffusion-v1-5",
                    controlnet=ControlNetModel.from_pretrained(
                        f"{local_fold}/lllyasviel/sd-controlnet-scribble",
                        ),
                    ),
                "device": device
            }
        }
    pipes = {**standard_pipes, **other_pipes, **controlnet_sd_pipes}
    return pipes

pipes = load_pipes(local_deployment)

end = time.time()
during = end - start

@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({"error": {"message": str(e)}}), 500

print(f"[ ready ] {during}s")

@app.route('/running', methods=['GET'])
def running():
    return jsonify({"running": True})

@app.route('/status/<path:model_id>', methods=['GET'])
def status(model_id):
    disabled_models = ["microsoft/trocr-base-handwritten"]
    if model_id in pipes.keys() and model_id not in disabled_models:
        print(f"[ check {model_id} ] success")
        return jsonify({"loaded": True})
    else:
        #print(f"[ check {model_id} ] failed")
        return jsonify({"loaded": False})

@app.route('/models/<path:model_id>', methods=['POST'])
def models(model_id):
    if model_id not in pipes:
        return jsonify({"error": {"message": f"Model {model_id} not loaded"}}), 404
    
    def ensure_pipeline_on_device(pipe, device_str):
        """
        Ensures that the given pipeline (or model) and its submodules are moved
        to the correct torch.device, fixing mismatches between CPU and GPU tensors.
        """
        # Normalize device_str → string
        if isinstance(device_str, torch.device):
            device_str = str(device_str)
    
        dev = torch.device(device_str if "cuda" in device_str else "cpu")

        try:
            if hasattr(pipe, "to"):
                pipe.to(dev)
            if hasattr(pipe, "model") and hasattr(pipe.model, "to"):
                pipe.model.to(dev)
            if hasattr(pipe, "unet"):
                pipe.unet.to(dev)
            if hasattr(pipe, "vae"):
                pipe.vae.to(dev)
            if hasattr(pipe, "text_encoder"):
                pipe.text_encoder.to(dev)
            if hasattr(pipe, "controlnet"):
                pipe.controlnet.to(dev)
            if hasattr(pipe, "encoder"):
                pipe.encoder.to(dev)
            if hasattr(pipe, "decoder"):
                pipe.decoder.to(dev)
            if hasattr(pipe, "tokenizer") and hasattr(pipe.tokenizer, "device_map"):
                pipe.tokenizer.device_map = device_str
        except Exception as e:
            print(f"[warn] Failed to move pipeline to {device_str}: {e}")

        return pipe



    while pipes.get(model_id, {}).get("using", False):
        print(f"[ inference {model_id} ] waiting")
        time.sleep(0.1)
    pipes[model_id]["using"] = True
    print(f"[ inference {model_id} ] start")

    start = time.time()
    result = None

    pipe = pipes[model_id]["model"]

    # ---- device placement (fixes: can't set attribute 'device') ----
    if "device" in pipes[model_id]:
        ensure_pipeline_on_device(pipe, pipes[model_id]["device"])

    try:
        # -------------------- TEXT → VIDEO --------------------
        if model_id == "damo-vilab/text-to-video-ms-1.7b":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            prompt = request.get_json()["text"]
            video_frames = pipe(prompt, num_inference_steps=50, num_frames=40).frames
            video_path = export_to_video(video_frames) 
            file_name = str(uuid.uuid4())[:4] + ".mp4"
            public_dir = "public/videos"
            os.makedirs(public_dir, exist_ok=True)
            output_path = os.path.join(public_dir, file_name)
            shutil.move(video_path, output_path)
            result = {"path": f"/videos/{file_name}"}

        # -------------------- CONTROLNET (TXT+CTRL → IMG) --------------------
        if model_id.startswith("lllyasviel/sd-controlnet-"):
            # --- ensure the base diffusion pipeline is on GPU ---
            device_str = pipes[model_id]["device"]
            ensure_pipeline_on_device(pipe, device_str)

            # --- ensure controlnet is on same device & dtype ---
            ctrl = pipes[model_id]["control"]
            ctrl.to(torch.device(device_str))
            if hasattr(pipe, "dtype") and ctrl.dtype != pipe.dtype:
                ctrl = ctrl.to(dtype=pipe.dtype)

            # attach controlnet
            pipe.controlnet = ctrl
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            # load control image and run inference
            control_image = load_image(request.get_json()["img_url"])
            out_image: Image = pipe(
                request.get_json()["text"],
                num_inference_steps=20,
                image=control_image
            ).images[0]

            file_name = str(uuid.uuid4())[:4]
            out_image.save(f"public/images/{file_name}.png")
            result = {"path": f"/images/{file_name}.png"}

        # -------------------- CONTROL MAP GENERATORS --------------------
        if model_id.endswith("-control"):
            image = load_image(request.get_json()["img_url"])
            if "scribble" in model_id:
                control = pipe(image, scribble=True)
            elif "canny" in model_id:
                control = pipe(image, low_threshold=100, high_threshold=200)
            else:
                control = pipe(image)
            file_name = str(uuid.uuid4())[:4]
            control.save(f"public/images/{file_name}.png")
            result = {"path": f"/images/{file_name}.png"}

        # -------------------- IMG → IMG (Image Variations) --------------------
        if model_id == "lambdalabs/sd-image-variations-diffusers":
            im = load_image(request.get_json()["img_url"])
            file_name = str(uuid.uuid4())[:4]
            tform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                     [0.26862954, 0.26130258, 0.27577711]),
            ])
            inp = tform(im).to(pipes[model_id]["device"]).unsqueeze(0)
            out = pipe(inp, guidance_scale=3)
            out["images"][0].save(f"public/images/{file_name}.jpg")
            result = {"path": f"/images/{file_name}.jpg"}

        # -------------------- IMG → TXT (Captioning) --------------------
        if model_id == "ydshieh/vit-gpt2-coco-en":
            img_url = request.get_json()["img_url"]
            generated_text = pipe(img_url)[0]['generated_text']
            result = {"generated text": generated_text}

        if model_id == "nlpconnect/vit-gpt2-image-captioning":
            image = load_image(request.get_json()["img_url"]).convert("RGB")
            pixel_values = pipes[model_id]["feature_extractor"](images=image, return_tensors="pt").pixel_values.to(pipes[model_id]["device"])
            generated_ids = pipe.generate(pixel_values, max_length=200, num_beams=1)
            generated_text = pipes[model_id]["tokenizer"].batch_decode(generated_ids, skip_special_tokens=True)[0]
            result = {"generated text": generated_text}

        # -------------------- TXT → IMG (Stable Diffusion) --------------------
        if model_id == "runwayml/stable-diffusion-v1-5":
            file_name = str(uuid.uuid4())[:4]
            text = request.get_json()["text"]
            out = pipe(prompt=text, num_inference_steps=30)
            out["images"][0].save(f"public/images/{file_name}.jpg")
            result = {"path": f"/images/{file_name}.jpg"}

        # -------------------- DETECTION --------------------
        if model_id == "google/owlvit-base-patch32":
            img_url = request.get_json()["img_url"]
            open_types = ["cat", "couch", "person", "car", "dog", "horse", "sheep", "cow", "elephant", "bear",
                          "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                          "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                          "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                          "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                          "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                          "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                          "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
                          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird"]
            result = pipe(img_url, candidate_labels=open_types)
            
        if model_id == "facebook/detr-resnet-101":
            img_url = request.get_json()["img_url"]
            result = pipe(img_url)

        # -------------------- VQA (ViLT) --------------------
        if model_id == "dandelin/vilt-b32-finetuned-vqa":
            pipe = pipes[model_id]["model"]
            processor = pipes[model_id]["processor"]
            device = torch.device(pipes[model_id]["device"])

            question = request.get_json()["text"]
            img_url = request.get_json()["img_url"]

            # Load image
            image = load_image(img_url).convert("RGB")

            # Preprocess using the processor (tokenizer + feature extractor)
            encoding = processor(
                text=question,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )

            # Move to correct device
            encoding = {k: v.to(device) for k, v in encoding.items()}

            # Ensure model on same device
            ensure_pipeline_on_device(pipe, device)

            # Run inference
            with torch.no_grad():
                outputs = pipe.model(**encoding)

            logits = outputs.logits
            answer_id = logits.argmax(-1).item()
            answer = pipe.model.config.id2label[answer_id]

            result = {"answer": answer}


        # -------------------- Document QA --------------------
        if model_id == "impira/layoutlm-document-qa":
            question = request.get_json()["text"]
            img_url = request.get_json()["img_url"]
            result = pipe(image=img_url, question=question)

        # -------------------- DEPTH --------------------
        if model_id == "Intel/dpt-hybrid-midas":
            image = load_image(request.get_json()["img_url"])
            inputs = pipes[model_id]["feature_extractor"](images=image, return_tensors="pt")

            device = torch.device(pipes[model_id]["device"])
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            with torch.no_grad():
                ensure_pipeline_on_device(pipe, pipes[model_id]["device"])
                outputs = pipe(**inputs)
                predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            out_img = Image.fromarray(formatted)
            name = str(uuid.uuid4())[:4]
            out_img.save(f"public/images/{name}.jpg")
            result = {"path": f"/images/{name}.jpg"}

        # -------------------- TTS (SpeechT5, no dataset) --------------------
        if model_id == "microsoft/speecht5_tts":
            text = request.get_json()["text"]
            proc = pipes[model_id]["processor"]
            vocoder = pipes[model_id]["vocoder"]
            ensure_pipeline_on_device(vocoder, pipes[model_id]["device"])

            inputs = proc(text=text, return_tensors="pt")
            # fixed random speaker embedding for determinism (no dataset)
            torch.manual_seed(42)
            speaker_embeddings = torch.randn(1, 512, device=torch.device(pipes[model_id]["device"]))

            input_ids = inputs["input_ids"].to(pipes[model_id]["device"])
            speech = pipe.generate_speech(input_ids, speaker_embeddings, vocoder=vocoder)
            name = str(uuid.uuid4())[:4]
            sf.write(f"public/audios/{name}.wav", speech.cpu().numpy(), samplerate=16000)
            result = {"path": f"/audios/{name}.wav"}

        # -------------------- ASR --------------------
        if model_id == "openai/whisper-base" or model_id == "microsoft/speecht5_asr":
            import torchaudio
            audio_url = request.get_json()["audio_url"]
        
            pipe = pipes[model_id]["model"]
            device_str = pipes[model_id]["device"]
            dev = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
            ensure_pipeline_on_device(pipe, str(dev))
        
            # --- Load audio
            waveform, sr = torchaudio.load(audio_url)
            waveform = waveform.mean(dim=0).unsqueeze(0)  # mono
            waveform = waveform.to(torch.float32)
        
            # --- Feature extraction
            feature_extractor = pipe.feature_extractor
            input_features = feature_extractor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=sr).input_features
            input_features = input_features.to(dev)
        
            # --- Generate text directly (no pipeline call)
            with torch.no_grad():
                predicted_ids = pipe.model.generate(input_features)
                transcription = pipe.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
            result = {"text": transcription}

        # -------------------- SEGMENTATION --------------------
        if model_id == "facebook/detr-resnet-50-panoptic":
            segments = pipe(request.get_json()["img_url"])
            image = load_image(request.get_json()["img_url"]).convert('RGBA')
            colors = [(random.randint(100, 255), random.randint(100, 255),
                       random.randint(100, 255), 50) for _ in range(len(segments))]
            for i, segment in enumerate(segments):
                mask = segment["mask"].convert('L')
                layer = Image.new('RGBA', mask.size, colors[i])
                image.paste(layer, (0, 0), mask)
            name = str(uuid.uuid4())[:4]
            image.save(f"public/images/{name}.jpg")
            result = {"path": f"/images/{name}.jpg"}

        if model_id == "facebook/maskformer-swin-base-coco":
            image = load_image(request.get_json()["img_url"])

            fe = pipes[model_id]["feature_extractor"]
            model = pipes[model_id]["model"]

            # Ensure model is on correct device
            device = torch.device(pipes[model_id]["device"])
            model = model.to(device)

            # Preprocess
            inputs = fe(images=image, return_tensors="pt")

            # >>> FIX: Move all tensors to model device <<<
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)   # NOT "pipe"

            # panoptic post-process (CPU output)
            post = fe.post_process_panoptic_segmentation(
                outputs,
                target_sizes=[image.size[::-1]]
            )[0]

            predicted_panoptic_map = post["segmentation"].cpu().numpy()
            predicted_panoptic_map = Image.fromarray(predicted_panoptic_map.astype(np.uint8))

            name = str(uuid.uuid4())[:4]
            predicted_panoptic_map.save(f"public/images/{name}.jpg")

            result = {"path": f"/images/{name}.jpg"}

        # -------------------- NER --------------------
        if model_id == "dslim/distilbert-NER":
            text = request.get_json()["text"]
            pipe = pipes[model_id]["model"]
            device = pipes[model_id]["device"]

            # --- ensure model and tokenizer on correct device ---
            ensure_pipeline_on_device(pipe, device)

            # --- manually push inputs to same device (pipeline doesn't always) ---
            encoded = pipe.tokenizer(text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = pipe.model(**encoded)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()

            tokens = pipe.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
            labels = [pipe.model.config.id2label[p] for p in predictions]
            ents = [{"token": t, "label": l} for t, l in zip(tokens, labels)]

            result = {"entities": ents}
            
        # ------------- Image Classification -------------
        
        if model_id == "microsoft/beit-base-patch16-224-pt22k-ft22k":
            img_url = request.get_json()["img_url"]

            pipe = pipes[model_id]["model"]
            device_str = pipes[model_id]["device"]
            ensure_pipeline_on_device(pipe, device_str)

            image = load_image(img_url).convert("RGB")

            processor = BeitImageProcessor.from_pretrained(
                f"{local_fold}/microsoft/beit-base-patch16-224-pt22k-ft22k"
            )

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device_str) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = pipe.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            top_k = request.get_json().get("top_k", 5)
            values, indices = probs.topk(top_k)

            id2label = pipe.model.config.id2label
            preds = [
                {"label": id2label[idx.item()], "score": val.item()}
                for val, idx in zip(values[0], indices[0])
            ]

            result = {"predictions": preds}

    except Exception as e:
        print(e)
        traceback.print_exc()
        result = {"error": {"message": "Error when running the model inference."}}

    if "device" in pipes[model_id]:
        try:
            # Only move back to CPU if pipeline dtype supports it
            if getattr(pipe, "dtype", torch.float32) != torch.float16:
                ensure_pipeline_on_device(pipe, "cpu")
            else:
                # Keep float16 models on GPU; just clear unused cache
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[warn] Skipped CPU move for {model_id}: {e}")

    pipes[model_id]["using"] = False

    if result is None:
        result = {"error": {"message": "model not found"}}

    end = time.time()
    during = end - start
    print(f"[ complete {model_id} ] {during}s")
    print(f"[ result {model_id} ] {result}")

    return jsonify(result)


if __name__ == '__main__':
    # temp folders
    if not os.path.exists("public/audios"):
        os.makedirs("public/audios")
    if not os.path.exists("public/images"):
        os.makedirs("public/images")
    if not os.path.exists("public/videos"):
        os.makedirs("public/videos")
             
    waitress.serve(app, host="0.0.0.0", port=port)