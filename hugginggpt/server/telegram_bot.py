import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any
import argparse
import asyncio
import librosa
import soundfile as sf
from urllib.parse import urlparse, unquote

import requests
import yaml
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Args / config
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Flask server endpoint
HUGGINGGPT_SERVER_URL = "http://127.0.0.1:8004/hugginggpt"
HF_TOKEN = config['huggingface']['token']
MODEL_NAME = config['model']

# Static/public directory used by Flask:
# app = flask.Flask(__name__, static_folder="public", static_url_path="/")
PUBLIC_DIR = Path(config.get("public_dir", "public"))
TELEGRAM_UPLOAD_DIR = PUBLIC_DIR / "uploads" / "telegram"
TELEGRAM_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Optional base URL for absolute media links if ever needed
HUGGINGGPT_BASE_URL = config.get("hugginggpt_base_url", "").rstrip("/")

# Regex patterns for media paths returned in backend text
IMAGE_PATTERN = re.compile(
    r'(?:http[s]?://\S+\.(?:jpg|jpeg|png|gif|webp|tiff)|file://\S+\.(?:jpg|jpeg|png|gif|webp|tiff)|/\S+\.(?:jpg|jpeg|png|gif|webp|tiff))',
    re.IGNORECASE,
)

AUDIO_PATTERN = re.compile(
    r'(?:http[s]?://\S+|file://\S+|/\S+\.(?:wav|flac|mp3|ogg|m4a))',
    re.IGNORECASE,
)

VIDEO_PATTERN = re.compile(
    r'(?:http[s]?://\S+|file://\S+|/\S+\.(?:mp4|mov|webm|mkv))',
    re.IGNORECASE,
)


# ------------------------------------------------------------
# Backend call
# ------------------------------------------------------------
def call_hugginggpt(messages: list[dict[str, str]]) -> Any:
    payload = {
                "messages": messages,
                "api_type": "huggingface",
                "api_key": HF_TOKEN,
                "api_endpoint": "",
                "model": MODEL_NAME,
            }
    response = requests.post(HUGGINGGPT_SERVER_URL, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def extract_text_from_response(data: Any) -> str:
    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        for key in ["message", "response", "text", "result"]:
            if key in data and isinstance(data[key], str):
                return data[key]

        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                msg = choice.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg["content"])

        return str(data)

    return str(data)


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def split_text(text: str, max_len: int = 4000) -> list[str]:
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""

    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > max_len:
            if current:
                chunks.append(current)
            current = line
        else:
            current += line

    if current:
        chunks.append(current)

    return chunks

def resample_audio(input_path: str, output_path: str, target_sr: int = 16000):
    audio, sr = librosa.load(input_path, sr=None)  # keep original SR
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    sf.write(output_path, audio_resampled, target_sr)

def normalize_media_ref(ref: str) -> str:
    ref = ref.strip()

    if ref.startswith("file://"):
        parsed = urlparse(ref)
        ref = parsed.path

    ref = unquote(ref)

    while "//" in ref and not ref.startswith("http"):
        ref = ref.replace("//", "/")

    return ref


def should_send_media_ref(ref: str) -> bool:
    ref = normalize_media_ref(ref)

    if ref.startswith("/uploads/telegram/"):
        return False

    return (
        ref.startswith("/images/")
        or ref.startswith("/audios/")
        or ref.startswith("/videos/")
        or ref.startswith("http://")
        or ref.startswith("https://")
    )


def unique_normalized_refs(items: list[str]) -> list[str]:
    seen = set()
    out = []

    for item in items:
        norm = normalize_media_ref(item)
        if not should_send_media_ref(norm):
            continue
        if norm not in seen:
            seen.add(norm)
            out.append(norm)

    return out

def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def resolve_local_media_path(media_ref: str) -> Path | None:
    media_ref = normalize_media_ref(media_ref)

    if media_ref.startswith("http://") or media_ref.startswith("https://"):
        return None

    if media_ref.startswith("/"):
        return PUBLIC_DIR / media_ref.lstrip("/")

    return None


def build_user_message_from_photo(caption: str | None, relative_path: str) -> str:
    """
    Keep backend unchanged:
    embed the image path directly in the normal text content.
    """
    cleaned_caption = (caption or "").strip()

    if cleaned_caption:
        return f"{cleaned_caption} {relative_path}"

    return f"Please describe this image {relative_path}"


async def send_reply_with_media(update: Update, reply_text: str) -> None:
    image_refs = unique_normalized_refs(IMAGE_PATTERN.findall(reply_text))
    audio_refs = unique_normalized_refs(AUDIO_PATTERN.findall(reply_text))
    video_refs = unique_normalized_refs(VIDEO_PATTERN.findall(reply_text))

    caption_limit = 1024  # Telegram caption limit
    caption_text = reply_text[:caption_limit]
    for ref in image_refs:
        norm = normalize_media_ref(ref)
        local_path = resolve_local_media_path(ref)
        logger.info("Image ref raw=%s norm=%s local=%s exists=%s", ref, norm, local_path, local_path.exists() if local_path else None)

    # If there is at least one image, send the first one with caption
    if image_refs:
        first_image = image_refs[0]

        try:
            if first_image.startswith("http://") or first_image.startswith("https://"):
                await update.message.reply_photo(photo=first_image, caption=caption_text)
            else:
                local_path = resolve_local_media_path(first_image)
                if local_path and local_path.exists():
                    with open(local_path, "rb") as f:
                        await update.message.reply_photo(photo=f, caption=caption_text)
                else:
                    # fallback to text if file missing
                    for chunk in split_text(reply_text):
                        await update.message.reply_text(chunk)
        except Exception:
            logger.exception("Failed to send captioned image: %s", first_image)
            for chunk in split_text(reply_text):
                await update.message.reply_text(chunk)

        # send additional images without repeating caption
        for ref in image_refs[1:]:
            try:
                if ref.startswith("http://") or ref.startswith("https://"):
                    await update.message.reply_photo(photo=ref)
                else:
                    local_path = resolve_local_media_path(ref)
                    if local_path and local_path.exists():
                        with open(local_path, "rb") as f:
                            await update.message.reply_photo(photo=f)
            except Exception:
                logger.exception("Failed to send additional image: %s", ref)

    else:
        # no image -> send text normally
        for chunk in split_text(reply_text):
            await update.message.reply_text(chunk)

    # audio
    for ref in audio_refs:
        try:
            if ref.startswith("http://") or ref.startswith("https://"):
                await update.message.reply_document(document=ref)
            else:
                local_path = resolve_local_media_path(ref)
                if local_path and local_path.exists():
                    with open(local_path, "rb") as f:
                        await update.message.reply_document(document=f)
        except Exception:
            logger.exception("Failed to send audio back to Telegram: %s", ref)

    # video
    for ref in video_refs:
        try:
            if ref.startswith("http://") or ref.startswith("https://"):
                await update.message.reply_video(video=ref, caption="" if image_refs else None)
            else:
                local_path = resolve_local_media_path(ref)
                if local_path and local_path.exists():
                    with open(local_path, "rb") as f:
                        await update.message.reply_video(video=f)
        except Exception:
            logger.exception("Failed to send video back to Telegram: %s", ref)


async def run_backend_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str) -> None:
    messages = context.chat_data.get("messages", [])

    if not messages:
        messages = [
            {"role": "system", "content": "You are HuggingGPT, an AI assistant."}
        ]

    messages.append({"role": "user", "content": user_text})

    waiting_msg = None

    try:
        waiting_gif = "public/waiting/mr_bean_waiting.gif"

        waiting_msg = await update.message.reply_animation(
            animation=waiting_gif,
            caption="Thinking..."
        )

        data = await asyncio.to_thread(call_hugginggpt, messages)
        reply_text = extract_text_from_response(data)

        messages.append({"role": "assistant", "content": reply_text})
        context.chat_data["messages"] = messages

    except Exception as e:
        logger.exception("Telegram bot error")
        reply_text = f"Error while contacting HuggingGPT:\n{e}"

    finally:
        if waiting_msg is not None:
            try:
                await waiting_msg.delete()
            except Exception:
                logger.exception("Failed to delete waiting GIF")

    if reply_text.startswith("Error while contacting HuggingGPT:"):
        for chunk in split_text(reply_text):
            await update.message.reply_text(chunk)
        return

    # Check whether the reply contains at least one sendable output image.
    image_refs = unique_normalized_refs(IMAGE_PATTERN.findall(reply_text))
    has_output_image = len(image_refs) > 0

    if has_output_image:
        caption_text = reply_text[:1024]
        remaining_text = reply_text[1024:].strip()

        await send_reply_with_media(update, caption_text)

        if remaining_text:
            for chunk in split_text(remaining_text):
                await update.message.reply_text(chunk)
    else:
        await send_reply_with_media(update, reply_text)


# ------------------------------------------------------------
# Commands
# ------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! I am the Telegram interface for HuggingGPT.\n"
        "Send me a message, or send an image with an optional caption."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/start - start the bot\n"
        "/help - show help\n"
        "/reset - clear local chat history\n\n"
        "You can send:\n"
        "- text messages\n"
        "- voice messages, reply to these to instruct the model\n"
        "- a photo\n"
        "- a photo with a caption like:\n"
        "  Count the objects in this image"
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["messages"] = []
    await update.message.reply_text("Chat history cleared.")


# ------------------------------------------------------------
# Message handlers
# ------------------------------------------------------------
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    if not user_text:
        return

    # check if replying to a voice message
    reply = update.message.reply_to_message
    voice_store = context.chat_data.get("voice_messages", {})

    if reply and reply.message_id in voice_store:
        audio_path = voice_store[reply.message_id]

        # build prompt using audio file path
        user_text = f"{user_text} {audio_path}"

    await run_backend_and_reply(update, context, user_text)


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.UPLOAD_PHOTO,
        )

        # Largest available size
        photo = update.message.photo[-1]
        telegram_file = await context.bot.get_file(photo.file_id)

        original_suffix = Path(telegram_file.file_path or "").suffix.lower()
        if original_suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            original_suffix = ".jpg"

        filename = f"{uuid.uuid4().hex}{original_suffix}"
        local_path = TELEGRAM_UPLOAD_DIR / filename

        await telegram_file.download_to_drive(custom_path=str(local_path))

        # Path that backend can understand without any schema change
        relative_path = f"/uploads/telegram/{filename}"

        user_text = build_user_message_from_photo(update.message.caption, relative_path)

        logger.info("Saved Telegram photo to %s", local_path)
        logger.info("Forwarding as text message: %s", user_text)

        await run_backend_and_reply(update, context, user_text)

    except Exception as e:
        logger.exception("Telegram photo handling error")
        await update.message.reply_text(f"Error while handling the image:\n{e}")
        
async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.voice:
        return

    voice = update.message.voice
    telegram_file = await context.bot.get_file(voice.file_id)

    filename = f"{uuid.uuid4().hex}.ogg"
    local_path = TELEGRAM_UPLOAD_DIR / filename

    await telegram_file.download_to_drive(custom_path=str(local_path))

    resampled_path = TELEGRAM_UPLOAD_DIR / f"{uuid.uuid4().hex}_16k.wav"
    resample_audio(str(local_path), str(resampled_path))

    relative_path = f"/uploads/telegram/{resampled_path.name}"

    # store mapping: message_id → file path
    voice_store = context.chat_data.setdefault("voice_messages", {})
    voice_store[update.message.message_id] = relative_path

    await update.message.reply_text(
        "Voice message received. Reply to your voice message with instructions."
    )


def main() -> None:
    telegram_token = config.get("telegram_bot_token")
    if not telegram_token:
        raise RuntimeError("Missing telegram_bot_token in config")

    app = Application.builder().token(telegram_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))

    # Photo handler first, then text
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logger.info("Telegram bot started")
    app.run_polling()


if __name__ == "__main__":
    main()