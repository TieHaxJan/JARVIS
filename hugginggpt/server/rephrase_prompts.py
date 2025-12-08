import re
import json
from openai import OpenAI

client = OpenAI(api_key="")

INPUT_FILE = "test_prompts.json"
OUTPUT_FILE = "test_prompts_rephrased.json"

# Regex to capture file paths, resource IDs, and URLs
FILE_PATTERN = re.compile(
    r'(?:/[^ \n\t]+(?:\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|svg)))|'  # /images/a.jpg
    r'(?:https?://[^\s]+)|'                                       # URLs
    r'(?:[A-Za-z]:\\[^\s]+)|'                                     # Windows paths
    r'(?:\b[A-Za-z0-9_-]+\.[A-Za-z]{2,4}\b)'                      # filenames like image.png
)

def preserve_and_rephrase(prompt: str) -> str:
    """
    Rephrases text while preserving file paths and URLs.
    We replace file paths with placeholders, rephrase the text,
    then restore the original paths.
    """

    # 1. Extract links + file paths
    matches = FILE_PATTERN.findall(prompt)

    placeholder_map = {}
    temp_prompt = prompt

    for i, match in enumerate(matches):
        placeholder = f"__FILE_{i}__"
        placeholder_map[placeholder] = match
        temp_prompt = temp_prompt.replace(match, placeholder)

    # 2. Ask the LLM to rephrase ONLY the natural language, not placeholders
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": (
                 "Rephrase the user's prompt while preserving placeholders like __FILE_X__. "
                 "Do NOT add or remove placeholders. "
                 "Return ONLY the rewritten prompt."
             )},
            {"role": "user", "content": temp_prompt}
        ],
        max_tokens=200
    )

    rewritten = response.choices[0].message.content.strip()

    # 3. Put the file paths back
    for placeholder, value in placeholder_map.items():
        rewritten = rewritten.replace(placeholder, value)

    return rewritten


def main():
    # Load the JSON array
    with open(INPUT_FILE, "r") as f:
        items = json.load(f)

    # Process each item
    for obj in items:
        original = obj.get("prompt", "")
        print(f"Rephrasing: {original}")

        new_prompt = preserve_and_rephrase(original)
        obj["prompt"] = new_prompt  # Replace with new prompt

        print(f" → {new_prompt}\n")

    # Save new file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved rewritten prompts to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
