import base64
import copy
from io import BytesIO
import io
import os
import random
import time
import traceback
import uuid
import requests
import re
import json
import logging
import argparse
import yaml
from PIL import Image, ImageDraw
from pydub import AudioSegment
import threading
from queue import Queue
import flask
from flask import request, jsonify
import waitress
from flask_cors import CORS, cross_origin
from get_token_ids import get_token_ids_for_task_parsing, get_token_ids_for_choose_model, count_tokens, get_max_context_length
from huggingface_hub.inference_api import InferenceApi
from huggingface_hub.inference_api import ALL_TASKS
from huggingface_hub import InferenceClient
from huggingface_hub import HfApi
from thefuzz import fuzz

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config.default.yaml")
parser.add_argument("--mode", type=str, default="cli")
args = parser.parse_args()

if __name__ != "__main__":
    args.config = "configs/config.gradio.yaml"
    args.mode = "gradio"

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

os.makedirs("logs", exist_ok=True)
os.makedirs("public/images", exist_ok=True)
os.makedirs("public/audios", exist_ok=True)
os.makedirs("public/videos", exist_ok=True)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not config["debug"]:
    handler.setLevel(logging.CRITICAL)
logger.addHandler(handler)

log_file = config["log_file"]
if log_file:
    filehandler = logging.FileHandler(log_file)
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

LLM = config["model"]
use_completion = config["use_completion"]

def load_image(url_or_path):
    if url_or_path.startswith("http"):
        response = requests.get(url_or_path)
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(url_or_path).convert("RGB")


# consistent: wrong msra model name 
LLM_encoding = LLM
if config["dev"] and LLM == "gpt-3.5-turbo":
    LLM_encoding = "text-davinci-003"
task_parsing_highlight_ids = get_token_ids_for_task_parsing(LLM_encoding)
choose_model_highlight_ids = get_token_ids_for_choose_model(LLM_encoding)

# ENDPOINT	MODEL NAME	
# /v1/chat/completions	gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301	
# /v1/completions	text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001, davinci, curie, babbage, ada

if use_completion:
    api_name = "completions"
else:
    api_name = "chat/completions"

API_TYPE = None
# priority: local > azure > openai
if "dev" in config and config["dev"]:
    API_TYPE = "local"
elif "azure" in config:
    API_TYPE = "azure"
elif "openai" in config:
    API_TYPE = "openai"
elif "huggingface" in config:
    API_TYPE = "huggingface"
else:
    logger.warning(f"No endpoint specified in {args.config}. The endpoint will be set dynamically according to the client.")

if args.mode in ["test", "cli"]:
    assert API_TYPE, "Only server mode supports dynamic endpoint."

API_KEY = None
API_ENDPOINT = None
if API_TYPE == "local":
    API_ENDPOINT = f"{config['local']['endpoint']}/v1/{api_name}"
elif API_TYPE == "azure":
    API_ENDPOINT = f"{config['azure']['base_url']}/openai/deployments/{config['azure']['deployment_name']}/{api_name}?api-version={config['azure']['api_version']}"
    API_KEY = config["azure"]["api_key"]
elif API_TYPE == "openai":
    API_ENDPOINT = f"https://api.openai.com/v1/{api_name}"
    #API_ENDPOINT = f"http://0.0.0.0:8010/v1/{api_name}"
    if config["openai"]["api_key"].startswith("sk-"):  # Check for valid OpenAI key in config file
        API_KEY = config["openai"]["api_key"]
    elif "OPENAI_API_KEY" in os.environ and os.getenv("OPENAI_API_KEY").startswith("sk-"):  # Check for environment variable OPENAI_API_KEY
        API_KEY = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(f"Incorrect OpenAI key. Please check your {args.config} file.")
elif API_TYPE == "huggingface":
    API_KEY = config["huggingface"]["token"]
    client = InferenceClient(
        provider="auto",
        api_key=API_KEY,
    )

PROXY = None
if config["proxy"]:
    PROXY = {
        "https": config["proxy"],
    }

inference_mode = config["inference_mode"]

# check the local_inference_endpoint
Model_Server = None
if inference_mode!="huggingface":
    Model_Server = "http://" + config["local_inference_endpoint"]["host"] + ":" + str(config["local_inference_endpoint"]["port"])
    message = f"The server of local inference endpoints is not running, please start it first. (or using `inference_mode: huggingface` in {args.config} for a feature-limited experience)"
    try:
        r = requests.get(Model_Server + "/running")
        if r.status_code != 200:
            raise ValueError(message)
    except:
        raise ValueError(message)


parse_task_demos_or_presteps = open(config["demos_or_presteps"]["parse_task"], "r").read()
choose_model_demos_or_presteps = open(config["demos_or_presteps"]["choose_model"], "r").read()
response_results_demos_or_presteps = open(config["demos_or_presteps"]["response_results"], "r").read()

parse_task_prompt = config["prompt"]["parse_task"]
choose_model_prompt = config["prompt"]["choose_model"]
response_results_prompt = config["prompt"]["response_results"]

parse_task_tprompt = config["tprompt"]["parse_task"]
choose_model_tprompt = config["tprompt"]["choose_model"]
response_results_tprompt = config["tprompt"]["response_results"]

MODELS = [json.loads(line) for line in open("data/p0_models.jsonl", "r").readlines()]
MODELS_MAP = {}
for model in MODELS:
    tag = model["task"]
    if tag not in MODELS_MAP:
        MODELS_MAP[tag] = []
    MODELS_MAP[tag].append(model)
METADATAS = {}
for model in MODELS:
    METADATAS[model["id"]] = model
for task, models in MODELS_MAP.items():
    logger.debug(f"{task}: {len(models)} models")

HUGGINGFACE_HEADERS = {}
if config["huggingface"]["token"] and config["huggingface"]["token"].startswith("hf_"):  # Check for valid huggingface token in config file
    HUGGINGFACE_HEADERS = {
        "Authorization": f"Bearer {config['huggingface']['token']}",
    }
elif "HUGGINGFACE_ACCESS_TOKEN" in os.environ and os.getenv("HUGGINGFACE_ACCESS_TOKEN").startswith("hf_"):  # Check for environment variable HUGGINGFACE_ACCESS_TOKEN
    HUGGINGFACE_HEADERS = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACE_ACCESS_TOKEN')}",
    }
else:
    raise ValueError(f"Incorrect HuggingFace token. Please check your {args.config} file.")

def convert_chat_to_completion(data):
    messages = data.pop('messages', [])
    tprompt = ""
    if messages[0]['role'] == "system":
        tprompt = messages[0]['content']
        messages = messages[1:]
    final_prompt = ""
    for message in messages:
        if message['role'] == "user":
            final_prompt += ("<im_start>"+ "user" + "\n" + message['content'] + "<im_end>\n")
        elif message['role'] == "assistant":
            final_prompt += ("<im_start>"+ "assistant" + "\n" + message['content'] + "<im_end>\n")
        else:
            final_prompt += ("<im_start>"+ "system" + "\n" + message['content'] + "<im_end>\n")
    final_prompt = tprompt + final_prompt
    final_prompt = final_prompt + "<im_start>assistant"
    data["prompt"] = final_prompt
    data['stop'] = data.get('stop', ["<im_end>"])
    data['max_tokens'] = data.get('max_tokens', max(get_max_context_length(LLM) - count_tokens(LLM_encoding, final_prompt), 1))
    return data

def send_request(data):
    api_key = data.pop("api_key")
    api_type = data.pop("api_type")
    api_endpoint = data.pop("api_endpoint")
    if use_completion and "prompt" not in data:
        data = convert_chat_to_completion(data)
    if api_type == "openai":
        HEADER = {
            "Authorization": f"Bearer {api_key}"
        }
    elif api_type == "azure":
        HEADER = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
    else:
        HEADER = None
    #print("[ Request ]:", data.get("prompt"))
    if api_type == "huggingface":
        response = client.chat_completion(
            model=data["model"],
            messages=data["messages"],
            temperature=data.get("temperature", 0),
            logit_bias=data.get("logit_bias")
        )
        #print(f"[ Response ]: {response}")
        content = response["choices"][0]["message"]["content"].strip()
        cleaned = re.sub(r"<END>\s*$", "", content).rstrip()
        return cleaned
    response = requests.post(api_endpoint, json=data, headers=HEADER, proxies=PROXY)
    #print(f"[ Response ]: {response.json()}")
    if "error" in response.json():
        return response.json()
    logger.debug(response.text.strip())
    if use_completion:
        return response.json()["choices"][0]["text"].strip()
    else:
        return response.json()["choices"][0]["message"]["content"].strip()

def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key +"}}", value.replace('"', "'").replace('\n', ""))
    return text

def find_json(s):
    s = s.replace("\'", "\"")
    start = s.find("{")
    end = s.rfind("}")
    res = s[start:end+1]
    res = res.replace("\n", "")
    return res

def field_extract(s, field):
    try:
        field_rep = re.compile(f'{field}.*?:.*?"(.*?)"', re.IGNORECASE)
        extracted = field_rep.search(s).group(1).replace("\"", "\'")
    except:
        field_rep = re.compile(f'{field}:\ *"(.*?)"', re.IGNORECASE)
        extracted = field_rep.search(s).group(1).replace("\"", "\'")
    return extracted

def get_id_reason(choose_str):
    reason = field_extract(choose_str, "reason")
    id = field_extract(choose_str, "id")
    choose = {"id": id, "reason": reason}
    return id.strip(), reason.strip(), choose

def record_case(success, **args):
    if success:
        f = open("logs/log_success.jsonl", "a")
    else:
        f = open("logs/log_fail.jsonl", "a")
    log = args
    f.write(json.dumps(log) + "\n")
    f.close()

def image_to_bytes(img_url):
    img_byte = io.BytesIO()
    type = img_url.split(".")[-1]
    load_image(img_url).save(img_byte, format="png")
    img_data = img_byte.getvalue()
    return img_data

def resource_has_dep(command):
    args = command["args"]
    for _, v in args.items():
        if "<GENERATED>" in v:
            return True
    return False

def fix_dep(tasks):
    for task in tasks:
        args = task["args"]
        task["dep"] = []
        for k, v in args.items():
            if "<GENERATED>" in v:
                dep_task_id = int(v.split("-")[1])
                if dep_task_id not in task["dep"]:
                    task["dep"].append(dep_task_id)
        if len(task["dep"]) == 0:
            task["dep"] = [-1]
    return tasks

def unfold(tasks):
    flag_unfold_task = False
    try:
        for task in tasks:
            for key, value in task["args"].items():
                if "<GENERATED>" in value:
                    generated_items = value.split(",")
                    if len(generated_items) > 1:
                        flag_unfold_task = True
                        for item in generated_items:
                            new_task = copy.deepcopy(task)
                            dep_task_id = int(item.split("-")[1])
                            new_task["dep"] = [dep_task_id]
                            new_task["args"][key] = item
                            tasks.append(new_task)
                        tasks.remove(task)
    except Exception as e:
        print(e)
        traceback.print_exc()
        logger.debug("unfold task failed.")

    if flag_unfold_task:
        logger.debug(f"unfold tasks: {tasks}")
        
    return tasks

def chitchat(messages, api_key, api_type, api_endpoint):
    data = {
        "model": LLM,
        "messages": messages,
        "api_key": api_key,
        "api_type": api_type,
        "api_endpoint": api_endpoint
    }
    return send_request(data)

def build_llama_prompt(task_type, tprompt, demo_text, user_prompt):
    return f"""{tprompt.strip()} 
    {demo_text.strip()} 
    User: {user_prompt.strip()} 
    Assistant:"""

def parse_task(context, input, api_key, api_type, api_endpoint):
    demos_or_presteps = parse_task_demos_or_presteps
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": parse_task_tprompt})

    # cut chat logs
    start = 0
    while start <= len(context):
        history = context[start:]
        prompt = replace_slot(parse_task_prompt, {
            "input": input,
            "context": history
        })
        if not use_completion:
            messages = copy.deepcopy(json.loads(demos_or_presteps))
            messages.insert(0, {"role": "system", "content": parse_task_tprompt})
            messages.append({"role": "user", "content": prompt})
            history_text = "<im_end>\nuser<im_start>".join([m["content"] for m in messages])
            num = count_tokens(LLM_encoding, history_text)
            if get_max_context_length(LLM) - num > 800:
                break
        else:
            # Completion mode doesn't need token check, just build flat prompt
            break
        start += 2
    
    if use_completion:
        full_prompt = build_llama_prompt(
            "parse_task",
            parse_task_tprompt,
            demos_or_presteps,
            prompt
        )
        logger.debug(f"Full_prompt: {full_prompt}")
        data = {
            "model": LLM,
            "prompt": full_prompt,
            "temperature": 0,
            "max_tokens": 1024,
            "stop": ["User:"],
            "api_key": api_key,
            "api_type": api_type,
            "api_endpoint": api_endpoint
        }
    else:
        logger.debug(f"Messages: {messages}")
        # Chat-style message for OpenAI API
        data = {
            "model": LLM,
            "messages": messages,
            "temperature": 0,
            "logit_bias": {item: config["logit_bias"]["parse_task"] for item in task_parsing_highlight_ids},
            "api_key": api_key,
            "api_type": api_type,
            "api_endpoint": api_endpoint
        }

    return send_request(data)

def choose_model(input, task, metas, api_key, api_type, api_endpoint):
    prompt = replace_slot(choose_model_prompt, {
        "input": input,
        "task": task,
        "metas": metas,
    })
    demos_or_presteps = replace_slot(choose_model_demos_or_presteps, {
        "input": input,
        "task": task,
        "metas": metas
    })
    if use_completion:
        full_prompt = build_llama_prompt(
            "choose_model",
            choose_model_tprompt,
            demos_or_presteps,
            prompt
        )
        data = {
            "model": LLM,
            "prompt": full_prompt,
            "temperature": 0,
            "max_tokens": 1024,
            "stop": ["User:"],
            "api_key": api_key,
            "api_type": api_type,
            "api_endpoint": api_endpoint
        }
    else:
        messages = json.loads(demos_or_presteps)
        messages.insert(0, {"role": "system", "content": choose_model_tprompt})
        messages.append({"role": "user", "content": prompt})
        data = {
            "model": LLM,
            "messages": messages,
            "temperature": 0,
            "logit_bias": {item: config["logit_bias"]["choose_model"] for item in choose_model_highlight_ids},
            "api_key": api_key,
            "api_type": api_type,
            "api_endpoint": api_endpoint
        }

    return send_request(data)


def response_results(input, results, api_key, api_type, api_endpoint):
    results = [v for k, v in sorted(results.items(), key=lambda item: item[0])]
    prompt = replace_slot(response_results_prompt, {
        "input": input,
    })
    demos_or_presteps = replace_slot(response_results_demos_or_presteps, {
        "input": input,
        "processes": json.dumps(results, ensure_ascii=False)
    })
    if use_completion:
        full_prompt = build_llama_prompt(
            "response_results",
            response_results_tprompt,
            demos_or_presteps,
            prompt
        )
        data = {
            "model": LLM,
            "prompt": full_prompt,
            "temperature": 0,
            "max_tokens": 1024,
            "stop": ["<END>"],
            "api_key": api_key,
            "api_type": api_type,
            "api_endpoint": api_endpoint
        }
    else:
        messages = json.loads(demos_or_presteps)
        messages.insert(0, {"role": "system", "content": response_results_tprompt})
        messages.append({"role": "user", "content": prompt})
        data = {
            "model": LLM,
            "messages": messages,
            "temperature": 0,
            "api_key": api_key,
            "api_type": api_type,
            "api_endpoint": api_endpoint
        }

    return send_request(data)

def huggingface_model_inference(model_id, data, task):

    inferenceClient = InferenceClient(
        provider="auto",
        api_key=API_KEY,
        model=model_id
    )
    
    # NLP tasks
    if task == "question-answering":
        result = inferenceClient.question_answering(question=data["text"], context=(data["context"] if "context" in data else ""))
    if task == "sentence-similarity":
        result = inferenceClient.sentence_similarity(sentence=data["text1"], other_sentences=data["text2"])
    if task == "text-classification":
        result = inferenceClient.text_classification(text=data["text"])
    if task == "token-classification":
        result = inferenceClient.token_classification(text=data["text"])
    if task == "summarization":
        result = inferenceClient.summarization(text=data["text"])
    if task == "translation":
        result = inferenceClient.translation(text=data["text"])
    if task in ["text2text-generation", "conversational", "text-generation"]:
        result = inferenceClient.text_generation(prompt=data["text"])
    
    # CV tasks
    if task == "visual-question-answering":
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        result = inferenceClient.visual_question_answering(image=img_data, question=data["text"])
    
    if task == "document-question-answering":
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        result = inferenceClient.document_question_answering(image=img_data, question=data["text"])

    if task == "image-to-image":
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        img = inferenceClient.image_to_image(image=img_data, prompt=data["text"])
        name = str(uuid.uuid4())[:4]
        img.save(f"public/images/{name}.png")
        result = {}
        result["generated image"] = f"/images/{name}.png"
    
    if task == "text-to-image":
        img = inferenceClient.text_to_image(prompt=data["text"])
        name = str(uuid.uuid4())[:4]
        img.save(f"public/images/{name}.png")
        result = {}
        result["generated image"] = f"/images/{name}.png"

    if task == "image-segmentation":
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        image = Image.open(BytesIO(img_data))
        predicted = inferenceClient.image_segmentation(image=img_data)
        colors = []
        for i in range(len(predicted)):
            colors.append((random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 155))
        for i, pred in enumerate(predicted):
            label = pred["label"]
            mask = pred.pop("mask").encode("utf-8")
            mask = base64.b64decode(mask)
            mask = Image.open(BytesIO(mask), mode='r')
            mask = mask.convert('L')

            layer = Image.new('RGBA', mask.size, colors[i])
            image.paste(layer, (0, 0), mask)
        name = str(uuid.uuid4())[:4]
        image.save(f"public/images/{name}.jpg")
        result = {}
        result["generated image"] = f"/images/{name}.jpg"
        result["predicted"] = predicted

    if task == "object-detection":
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        predicted = inferenceClient.object_detection(image=img_data)
        image = Image.open(BytesIO(img_data))
        draw = ImageDraw.Draw(image)
        labels = list(item['label'] for item in predicted)
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (random.randint(0, 255), random.randint(0, 100), random.randint(0, 255))
        for label in predicted:
            box = label["box"]
            draw.rectangle(((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])), outline=color_map[label["label"]], width=2)
            draw.text((box["xmin"]+5, box["ymin"]-15), label["label"], fill=color_map[label["label"]])
        name = str(uuid.uuid4())[:4]
        image.save(f"public/images/{name}.jpg")
        result = {}
        result["generated image"] = f"/images/{name}.jpg"
        result["predicted"] = predicted

    if task in ["image-classification"]:
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        result = inferenceClient.image_classification(image=img_data)
 
    if task == "image-to-text":
        img_url = data["image"]
        img_data = image_to_bytes(img_url)
        result = inferenceClient.image_to_text(image=img_data)
    
    # AUDIO tasks
    if task == "text-to-speech":
        response = inferenceClient.image_to_text(text=data["text"])
        # response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, json={"inputs": text})
        name = str(uuid.uuid4())[:4]
        with open(f"public/audios/{name}.flac", "wb") as f:
            f.write(response.content)
        result = {"generated audio": f"/audios/{name}.flac"}
    if task == "automatic-speech-recognition":
        audio_url = data["audio"]
        audio_data = requests.get(audio_url, timeout=10).content
        result = inferenceClient.automatic_speech_recognition(audio=audio_data)
    if task == "audio-classification":
        audio_url = data["audio"]
        audio_data = requests.get(audio_url, timeout=10).content
        result = inferenceClient.audio_classification(audio=audio_data)
    if task == "audio-to-audio":
        audio_url = data["audio"]
        response = inferenceClient.audio_classification(audio=audio_data)
        result = response.json()
        content = None
        type = None
        for k, v in result[0].items():
            if k == "blob":
                content = base64.b64decode(v.encode("utf-8"))
            if k == "content-type":
                type = "audio/flac".split("/")[-1]
        audio = AudioSegment.from_file(BytesIO(content))
        name = str(uuid.uuid4())[:4]
        audio.export(f"public/audios/{name}.{type}", format=type)
        result = {"generated audio": f"/audios/{name}.{type}"}
    return result

def local_model_inference(model_id, data, task):
    task_url = f"{Model_Server}/models/{model_id}"
    
    # contronlet
    if model_id.startswith("lllyasviel/sd-controlnet-"):
        img_url = data["image"]
        text = data["text"]
        response = requests.post(task_url, json={"img_url": img_url, "text": text})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if model_id.endswith("-control"):
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
        
    if task == "text-to-video":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated video"] = results.pop("path")
        return results

    # NLP tasks
    if task == "question-answering" or task == "sentence-similarity":
        response = requests.post(task_url, json=data)
        return response.json()
    if task in ["text-classification",  "token-classification", "text2text-generation", "summarization", "translation", "conversational", "text-generation"]:
        response = requests.post(task_url, json=data)
        return response.json()

    # CV tasks
    if task == "depth-estimation":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if task == "image-segmentation":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        results["generated image"] = results.pop("path")
        return results
    if task == "image-to-image":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if task == "text-to-image":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if task == "object-detection":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        predicted = response.json()
        if "error" in predicted:
            return predicted
        image = load_image(img_url)
        draw = ImageDraw.Draw(image)
        labels = list(item['label'] for item in predicted)
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (random.randint(0, 255), random.randint(0, 100), random.randint(0, 255))
        for label in predicted:
            box = label["box"]
            draw.rectangle(((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])), outline=color_map[label["label"]], width=2)
            draw.text((box["xmin"]+5, box["ymin"]-15), label["label"], fill=color_map[label["label"]])
        name = str(uuid.uuid4())[:4]
        image.save(f"public/images/{name}.jpg")
        results = {}
        results["generated image"] = f"/images/{name}.jpg"
        results["predicted"] = predicted
        return results
    if task in ["image-classification", "image-to-text", "document-question-answering", "visual-question-answering"]:
        img_url = data["image"]
        text = None
        if "text" in data:
            text = data["text"]
        response = requests.post(task_url, json={"img_url": img_url, "text": text})
        results = response.json()
        return results
    # AUDIO tasks
    if task == "text-to-speech":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated audio"] = results.pop("path")
        return results
    if task in ["automatic-speech-recognition", "audio-to-audio", "audio-classification"]:
        audio_url = data["audio"]
        response = requests.post(task_url, json={"audio_url": audio_url})
        return response.json()


def model_inference(model_id, data, hosted_on, task):
    if hosted_on == "unknown":
        localStatusUrl = f"{Model_Server}/status/{model_id}"
        r = requests.get(localStatusUrl)
        logger.debug("Local Server Status: " + str(r.json()))
        if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
            hosted_on = "local"
        else:
            huggingfaceStatusUrl = f"https://api-inference.huggingface.co/status/{model_id}"
            r = requests.get(huggingfaceStatusUrl, headers=HUGGINGFACE_HEADERS, proxies=PROXY)
            logger.debug("Huggingface Status: " + str(r.json()))
            if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
                hosted_on = "huggingface"
    try:
        if hosted_on == "local":
            inference_result = local_model_inference(model_id, data, task)
        elif hosted_on == "huggingface":
            inference_result = huggingface_model_inference(model_id, data, task)
    except Exception as e:
        print(e)
        traceback.print_exc()
        inference_result = {"error":{"message": str(e)}}
    return inference_result

def is_model_warm(model_id: str) -> bool:
    api = HfApi()
    try:
        info = api.model_info(model_id)
        return info.inference == "warm"
    except Exception as e:
        logger.error(f"Error checking model status for {model_id}: {e}")
        return False

def get_model_status(model_id, url, headers, queue = None):
    if "huggingface" in url:
        try:
            api = HfApi(token=headers.get("Authorization", "").replace("Bearer ", ""))
            info = api.model_info(model_id)
            status = info.inference == "warm"
        except Exception as e:
            status = False
        endpoint_type = "huggingface"
    else:
        # Local check
        try:
            r = requests.get(url)
            r.raise_for_status()
            status = r.json().get("loaded", False)
        except:
            status = False
        endpoint_type = "local"

    if queue:
        queue.put((model_id, status, endpoint_type if status else None))
    return status

def get_avaliable_models(candidates, topk=5):
    all_available_models = {"local": [], "huggingface": []}
    threads = []
    result_queue = Queue()

    for candidate in candidates:
        model_id = candidate["id"]

        if inference_mode != "local":
            huggingfaceStatusUrl = f"https://api-inference.huggingface.co/status/{model_id}"
            thread = threading.Thread(target=get_model_status, args=(model_id, huggingfaceStatusUrl, HUGGINGFACE_HEADERS, result_queue))
            threads.append(thread)
            thread.start()
        
        if inference_mode != "huggingface" and config["local_deployment"] != "minimal":
            localStatusUrl = f"{Model_Server}/status/{model_id}"
            thread = threading.Thread(target=get_model_status, args=(model_id, localStatusUrl, {}, result_queue))
            threads.append(thread)
            thread.start()
        
    result_count = len(threads)
    while result_count:
        model_id, status, endpoint_type = result_queue.get()
        logger.debug(f"Model: {model_id} | Status: {status} | Source: {endpoint_type}")
        if status and model_id not in all_available_models[endpoint_type]:
            all_available_models[endpoint_type].append(model_id)
        if len(all_available_models["local"] + all_available_models["huggingface"]) >= topk:
            break
        result_count -= 1

    for thread in threads:
        thread.join()

    return all_available_models

def collect_result(command, choose, inference_result):
    result = {"task": command}
    result["inference result"] = inference_result
    result["choose model result"] = choose
    logger.debug(f"inference result: {inference_result}")
    return result


def run_task(input, command, results, api_key, api_type, api_endpoint):
    id = command["id"]
    args = command["args"]
    task = command["task"]
    deps = command["dep"]
    if deps[0] != -1:
        dep_tasks = [results[dep] for dep in deps]
    else:
        dep_tasks = []
    
    logger.debug(f"Run task: {id} - {task}")
    logger.debug("Deps: " + json.dumps(dep_tasks))

    if deps[0] != -1:
        if "image" in args and "<GENERATED>-" in args["image"]:
            resource_id = int(args["image"].split("-")[1])
            if "generated image" in results[resource_id]["inference result"]:
                args["image"] = results[resource_id]["inference result"]["generated image"]
        if "audio" in args and "<GENERATED>-" in args["audio"]:
            resource_id = int(args["audio"].split("-")[1])
            if "generated audio" in results[resource_id]["inference result"]:
                args["audio"] = results[resource_id]["inference result"]["generated audio"]
        if "text" in args and "<GENERATED>-" in args["text"]:
            resource_id = int(args["text"].split("-")[1])
            if "generated text" in results[resource_id]["inference result"]:
                args["text"] = results[resource_id]["inference result"]["generated text"]

    text = image = audio = None
    for dep_task in dep_tasks:
        if "generated text" in dep_task["inference result"]:
            text = dep_task["inference result"]["generated text"]
            logger.debug("Detect the generated text of dependency task (from results):" + text)
        elif "text" in dep_task["task"]["args"]:
            text = dep_task["task"]["args"]["text"]
            logger.debug("Detect the text of dependency task (from args): " + text)
        if "generated image" in dep_task["inference result"]:
            image = dep_task["inference result"]["generated image"]
            logger.debug("Detect the generated image of dependency task (from results): " + image)
        elif "image" in dep_task["task"]["args"]:
            image = dep_task["task"]["args"]["image"]
            logger.debug("Detect the image of dependency task (from args): " + image)
        if "generated audio" in dep_task["inference result"]:
            audio = dep_task["inference result"]["generated audio"]
            logger.debug("Detect the generated audio of dependency task (from results): " + audio)
        elif "audio" in dep_task["task"]["args"]:
            audio = dep_task["task"]["args"]["audio"]
            logger.debug("Detect the audio of dependency task (from args): " + audio)

    if "image" in args and "<GENERATED>" in args["image"]:
        if image:
            args["image"] = image
    if "audio" in args and "<GENERATED>" in args["audio"]:
        if audio:
            args["audio"] = audio
    if "text" in args and "<GENERATED>" in args["text"]:
        if text:
            args["text"] = text

    for resource in ["image", "audio"]:
        if resource in args and not args[resource].startswith("public/") and len(args[resource]) > 0 and not args[resource].startswith("http"):
            args[resource] = f"public/{args[resource]}"
    
    if "-text-to-image" in command['task'] and "text" not in args:
        logger.debug("control-text-to-image task, but text is empty, so we use control-generation instead.")
        control = task.split("-")[0]
        
        if control == "seg":
            task = "image-segmentation"
            command['task'] = task
        elif control == "depth":
            task = "depth-estimation"
            command['task'] = task
        else:
            task = f"{control}-control"

    command["args"] = args
    logger.debug(f"parsed task: {command}")

    if task.endswith("-text-to-image") or task.endswith("-control"):
        if inference_mode != "huggingface":
            if task.endswith("-text-to-image"):
                control = task.split("-")[0]
                best_model_id = f"lllyasviel/sd-controlnet-{control}"
            else:
                best_model_id = task
            hosted_on = "local"
            reason = "ControlNet is the best model for this task."
            choose = {"id": best_model_id, "reason": reason}
            logger.debug(f"chosen model: {choose}")
        else:
            logger.warning(f"Task {command['task']} is not available. ControlNet need to be deployed locally.")
            record_case(success=False, **{"input": input, "task": command, "reason": f"Task {command['task']} is not available. ControlNet need to be deployed locally.", "op":"message"})
            inference_result = {"error": f"service related to ControlNet is not available."}
            results[id] = collect_result(command, "", inference_result)
            return False
    elif task in ["summarization", "translation", "conversational", "text-generation", "text2text-generation"]: # ChatGPT Can do
        best_model_id = "ChatGPT"
        reason = "ChatGPT performs well on some NLP tasks as well."
        choose = {"id": best_model_id, "reason": reason}
        messages = [{
            "role": "user",
            "content": f"[ {input} ] contains a task in JSON format {command}. Now you are a {command['task']} system, the arguments are {command['args']}. Just help me do {command['task']} and give me the result. The result must be in text form without any urls."
        }]
        response = chitchat(messages, api_key, api_type, api_endpoint)
        results[id] = collect_result(command, choose, {"response": response})
        return True
    else:
        if task not in MODELS_MAP:
            logger.warning(f"no available models on {task} task.")
            record_case(success=False, **{"input": input, "task": command, "reason": f"task not support: {command['task']}", "op":"message"})
            inference_result = {"error": f"{command['task']} not found in available tasks."}
            results[id] = collect_result(command, "", inference_result)
            return False

        # Look at all candidates
        candidates = MODELS_MAP[task][:31]
        all_avaliable_models = get_avaliable_models(candidates, config["num_candidate_models"])
        all_avaliable_model_ids = all_avaliable_models["local"] + all_avaliable_models["huggingface"]
        logger.debug(f"avaliable models on {command['task']}: {all_avaliable_models}")

        if len(all_avaliable_model_ids) == 0:
            logger.warning(f"no available models on {command['task']}")
            record_case(success=False, **{"input": input, "task": command, "reason": f"no available models: {command['task']}", "op":"message"})
            inference_result = {"error": f"no available models on {command['task']} task."}
            results[id] = collect_result(command, "", inference_result)
            return False
            
        if len(all_avaliable_model_ids) == 1:
            best_model_id = all_avaliable_model_ids[0]
            hosted_on = "local" if best_model_id in all_avaliable_models["local"] else "huggingface"
            reason = "Only one model available."
            choose = {"id": best_model_id, "reason": reason}
            logger.debug(f"chosen model: {choose}")
        else:
            cand_models_info = [
                {
                    "id": model["id"],
                    "inference endpoint": all_avaliable_models.get(
                        "local" if model["id"] in all_avaliable_models["local"] else "huggingface"
                    ),
                    "likes": model.get("likes"),
                    "description": model.get("description", "")[:config["max_description_length"]],
                    # "language": model.get("meta").get("language") if model.get("meta") else None,
                    "tags": model.get("meta").get("tags") if model.get("meta") else None,
                }
                for model in candidates
                if model["id"] in all_avaliable_model_ids
            ]

            choose_str = choose_model(input, command, cand_models_info, api_key, api_type, api_endpoint)
            logger.debug(f"chosen model: {choose_str}")
            try:
                choose = json.loads(choose_str)
                reason = choose["reason"]
                best_model_id = choose["id"]
                hosted_on = "local" if best_model_id in all_avaliable_models["local"] else "huggingface"
            except Exception as e:
                logger.warning(f"the response [ {choose_str} ] is not a valid JSON, try to find the model id and reason in the response.")
                choose_str = find_json(choose_str)
                best_model_id, reason, choose  = get_id_reason(choose_str)
                hosted_on = "local" if best_model_id in all_avaliable_models["local"] else "huggingface"
    inference_result = model_inference(best_model_id, args, hosted_on, command['task'])

    if "error" in inference_result:
        logger.warning(f"Inference error: {inference_result['error']}")
        record_case(success=False, **{"input": input, "task": command, "reason": f"inference error: {inference_result['error']}", "op":"message"})
        results[id] = collect_result(command, choose, inference_result)
        return False
    
    results[id] = collect_result(command, choose, inference_result)
    return True

def chat_huggingface(messages, api_key, api_type, api_endpoint, return_planning = False, return_results = False):
    start = time.time()
    context = messages[:-1]
    input = messages[-1]["content"]
    logger.info("*"*80)
    logger.info(f"input: {input}")

    task_str = parse_task(context, input, api_key, api_type, api_endpoint)

    if "error" in task_str:
        record_case(success=False, **{"input": input, "task": task_str, "reason": f"task parsing error: {task_str['error']['message']}", "op":"report message"})
        return {"message": task_str["error"]["message"]}

    task_str = task_str.strip()
    logger.info(task_str)

    try:
        tasks = json.loads(task_str)
    except Exception as e:
        logger.debug(e)
        response = chitchat(messages, api_key, api_type, api_endpoint)
        record_case(success=False, **{"input": input, "task": task_str, "reason": "task parsing fail", "op":"chitchat"})
        return {"message": response}
    
    if task_str == "[]":  # using LLM response for empty task
        record_case(success=False, **{"input": input, "task": [], "reason": "task parsing fail: empty", "op": "chitchat"})
        response = chitchat(messages, api_key, api_type, api_endpoint)
        return {"message": response}

    if len(tasks) == 1 and tasks[0]["task"] in ["summarization", "translation", "conversational", "text-generation", "text2text-generation"]:
        record_case(success=True, **{"input": input, "task": tasks, "reason": "chitchat tasks", "op": "chitchat"})
        response = chitchat(messages, api_key, api_type, api_endpoint)
        return {"message": response}

    tasks = unfold(tasks)
    tasks = fix_dep(tasks)
    logger.debug(tasks)
    
    if return_planning:
        return tasks

    results = {}
    threads = []
    tasks = tasks[:]
    d = dict()
    retry = 0
    while True:
        num_thread = len(threads)
        for task in tasks:
            # logger.debug(f"d.keys(): {d.keys()}, dep: {dep}")
            for dep_id in task["dep"]:
                if dep_id >= task["id"]:
                    task["dep"] = [-1]
                    break
            dep = task["dep"]
            if dep[0] == -1 or len(list(set(dep).intersection(d.keys()))) == len(dep):
                tasks.remove(task)
                thread = threading.Thread(target=run_task, args=(input, task, d, api_key, api_type, api_endpoint))
                thread.start()
                threads.append(thread)
        if num_thread == len(threads):
            time.sleep(0.5)
            retry += 1
        if retry > 160:
            logger.debug("User has waited too long, Loop break.")
            break
        if len(tasks) == 0:
            break
    for thread in threads:
        thread.join()
    
    results = d.copy()

    logger.debug(results)
    if return_results:
        return results
    
    response = response_results(input, results, api_key, api_type, api_endpoint).strip()

    end = time.time()
    during = end - start

    answer = {"message": response}
    record_case(success=True, **{"input": input, "task": task_str, "results": results, "response": response, "during": during, "op":"response"})
    logger.info(f"response: {response}")
    return answer

def test():
    # single round examples
    inputs = [
        "Given a collection of image A: /examples/a.jpg, B: /examples/b.jpg, C: /examples/c.jpg, please tell me how many zebras in these picture?"
        "Can you give me a picture of a small bird flying in the sky with trees and clouds. Generate a high definition image if possible.",
        "Please answer all the named entities in the sentence: Iron Man is a superhero appearing in American comic books published by Marvel Comics. The character was co-created by writer and editor Stan Lee, developed by scripter Larry Lieber, and designed by artists Don Heck and Jack Kirby.",
        "please dub for me: 'Iron Man is a superhero appearing in American comic books published by Marvel Comics. The character was co-created by writer and editor Stan Lee, developed by scripter Larry Lieber, and designed by artists Don Heck and Jack Kirby.'"
        "Given an image: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg, please answer the question: What is on top of the building?",
        "Please generate a canny image based on /examples/f.jpg"
        ]
        
    for input in inputs:
        messages = [{"role": "user", "content": input}]
        chat_huggingface(messages, API_KEY, API_TYPE, API_ENDPOINT, return_planning = False, return_results = False)
    
    # multi rounds example
    messages = [
        {"role": "user", "content": "Please generate a canny image based on /examples/f.jpg"},
        {"role": "assistant", "content": """Sure. I understand your request. Based on the inference results of the models, I have generated a canny image for you. The workflow I used is as follows: First, I used the image-to-text model (nlpconnect/vit-gpt2-image-captioning) to convert the image /examples/f.jpg to text. The generated text is "a herd of giraffes and zebras grazing in a field". Second, I used the canny-control model (canny-control) to generate a canny image from the text. Unfortunately, the model failed to generate the canny image. Finally, I used the canny-text-to-image model (lllyasviel/sd-controlnet-canny) to generate a canny image from the text. The generated image is located at /images/f16d.png. I hope this answers your request. Is there anything else I can help you with?"""},
        {"role": "user", "content": """then based on the above canny image and a prompt "a photo of a zoo", generate a new image."""},
    ]
    chat_huggingface(messages, API_KEY, API_TYPE, API_ENDPOINT, return_planning = False, return_results = False)

exit_variants = ["exit", "exut", "exot", "exir", "exiy", "exjt", "exiit", "quit", "bye"]

def cli():
    messages = []
    print("Welcome to Jarvis! A collaborative system that consists of an LLM as the controller and numerous expert models as collaborative executors. Jarvis can plan tasks, schedule Hugging Face models, generate friendly responses based on your requests, and help you with many things. Please enter your request (`exit` to exit).")
    while True:
        message = input("[ User ]: ")
        normalized = message.strip().lower()
        if any(fuzz.ratio(normalized, variant) >= 80 for variant in exit_variants):
            print("[ Jarvis ]: Goodnight.")
            break
        messages.append({"role": "user", "content": message})
        answer = chat_huggingface(messages, API_KEY, API_TYPE, API_ENDPOINT, return_planning=False, return_results=False)
        print("[ Jarvis ]: ", answer["message"])
        messages.append({"role": "assistant", "content": answer["message"]})


def server():
    http_listen = config["http_listen"]
    host = http_listen["host"]
    port = http_listen["port"]

    app = flask.Flask(__name__, static_folder="public", static_url_path="/")
    app.config['DEBUG'] = False
    CORS(app)
    
    @cross_origin()
    @app.route('/tasks', methods=['POST'])
    def tasks():
        data = request.get_json()
        messages = data["messages"]
        api_key = data.get("api_key", API_KEY)
        api_endpoint = data.get("api_endpoint", API_ENDPOINT)
        api_type = data.get("api_type", API_TYPE)
        if api_key is None or api_type is None or api_endpoint is None:
            return jsonify({"error": "Please provide api_key, api_type and api_endpoint"}) 
        response = chat_huggingface(messages, api_key, api_type, api_endpoint, return_planning=True)
        return jsonify(response)

    @cross_origin()
    @app.route('/results', methods=['POST'])
    def results():
        data = request.get_json()
        messages = data["messages"]
        api_key = data.get("api_key", API_KEY)
        api_endpoint = data.get("api_endpoint", API_ENDPOINT)
        api_type = data.get("api_type", API_TYPE)
        if api_key is None or api_type is None or api_endpoint is None:
            return jsonify({"error": "Please provide api_key, api_type and api_endpoint"}) 
        response = chat_huggingface(messages, api_key, api_type, api_endpoint, return_results=True)
        return jsonify(response)

    @cross_origin()
    @app.route('/hugginggpt', methods=['POST'])
    def chat():
        data = request.get_json()
        messages = data["messages"]
        api_key = data.get("api_key", API_KEY)
        api_endpoint = data.get("api_endpoint", API_ENDPOINT)
        api_type = data.get("api_type", API_TYPE)
        if api_key is None or api_type is None or api_endpoint is None:
            return jsonify({"error": "Please provide api_key, api_type and api_endpoint"}) 
        response = chat_huggingface(messages, api_key, api_type, api_endpoint)
        return jsonify(response)
    print("server running...")
    waitress.serve(app, host=host, port=port)

if __name__ == "__main__":
    if args.mode == "test":
        test()
    elif args.mode == "server":
        server()
    elif args.mode == "cli":
        cli()
