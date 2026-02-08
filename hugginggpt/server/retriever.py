import json
import os
import logging
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pathlib
import urllib.parse
import re

def mask_paths(s: str) -> str:
    if not isinstance(s, str):
        return s

    # (same regexes as in build_clean_context)
    _PATH_EXTS = (
        "png","jpg","jpeg","gif","webp","bmp","tiff","svg",
        "mp3","wav","flac","m4a","ogg",
        "mp4","mov","avi","mkv","webm",
        "pdf","txt","json","csv","yaml","yml","xml","html",
        "zip","tar","gz","7z","doc","docx","ppt","pptx","xls","xlsx"
    )

    # 1) URLs
    s = re.sub(r'https?://[^\s)>\]\'"]+', '[PATH]', s)
    # 2) Windows paths
    s = re.sub(r'[A-Za-z]:\\[^\s\'"]+', '[PATH]', s)
    # 3) Unix paths
    s = re.sub(r'(?<!\w)/[^\s\'"]+', '[PATH]', s)
    # 4) relative paths with extensions
    rel_ext_pattern = r'(?:(?:\./|\.\./)?(?:[\w\-\.]+/)+[\w\-.]+\.(?:' + "|".join(_PATH_EXTS) + '))'
    s = re.sub(rel_ext_pattern, '[PATH]', s, flags=re.IGNORECASE)
    # 5) JSON-style "/foo/bar.png"
    s = re.sub(r'\"(?:[\\/][\w\-.]+)+(?:\.[\w]+)\"', '"[PATH]"', s)

    return s

class ExplanationRetriever:
    def __init__(self, db_path="data/explanations.jsonl", model_name="Snowflake/snowflake-arctic-embed-m-v2.0", threshold=0.95):
        self.db_path = db_path
        self.embedder = SentenceTransformer(model_name, trust_remote_code=True)
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

        if not os.path.exists(self.db_path):
            open(self.db_path, "w").close()
            self.logger.debug(f"Created new explanations DB at {self.db_path}")

    def embed(self, text: str) -> np.ndarray:
        emb = self.embedder.encode(self.canonicalize(text), convert_to_numpy=True, batch_size=32)
        emb = emb.astype(np.float32, copy=False)
        emb /= (np.linalg.norm(emb) + 1e-12)
        return emb
    
    def canonicalize(self, text: str) -> str:
        """
        Canonicalize text for retrieval:
        - normalize casing
        - mask paths consistently
        - normalize whitespace
        - lightly normalize punctuation
        - preserve structure and technical tokens
        """
        if not isinstance(text, str):
            text = str(text)

        # 1) lowercase
        t = text.lower()

        # 2) mask paths (your existing logic)
        t = mask_paths(t)

        # 3) normalize whitespace (newlines → spaces)
        t = re.sub(r"\s+", " ", t).strip()

        # 4) normalize quotes/backticks (formatting noise)
        t = re.sub(r"[\"`']", "", t)

        # 5) collapse repeated punctuation (but keep symbols meaningful to code)
        #    e.g. "!!!" → "!"
        t = re.sub(r"([!?.,:;])\1+", r"\1", t)

        return t

    def load_db(self):
        with open(self.db_path, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        self.logger.debug(f"Loaded {len(entries)} entries from {self.db_path}")
        return entries

    def _mask_entry_strings(self, obj):
        """
        Recursively mask paths in all string fields of the object before saving.
        """
        if isinstance(obj, dict):
            return {k: self._mask_entry_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._mask_entry_strings(v) for v in obj]
        if isinstance(obj, str):
            return mask_paths(obj)
        return obj
    
    def _load_db_matrix(self):
        db = self.load_db()
        entries = []
        embs = []

        for entry in db:
            emb = entry.get("embedding", None)
            if emb is None:
                continue

            emb = np.asarray(emb, dtype=np.float32)

            embs.append(emb)
            entries.append(entry)

        if not embs:
            return None, None

        E = np.vstack(embs)  # (N, d)
        return entries, E


    def save_entry(self, entry: Dict[str, Any]):
        # Ensure *all* strings are path-masked before persisting
        sanitized = self._mask_entry_strings(entry)
        with open(self.db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
        preview = sanitized.get("explanation", "")[:60]
        self.logger.info(f"Saved new explanation to {self.db_path}: {preview}...")

    def find_similar(self, query_text: str):
        entries, E_db = self._load_db_matrix()
        if entries is None:
            self.logger.debug("DB empty → no similar entries")
            return None

        q = self.embed(query_text)            # (d,)
        q = q.reshape(1, -1)                 # (1, d) for sklearn

        sims = cosine_similarity(q, E_db)[0] # (N,)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        self.logger.debug(f"Best similarity = {best_sim:.3f} (threshold = {self.threshold})")

        if best_sim >= self.threshold:
            return {"entry": entries[best_idx], "cosine_similarity": best_sim}

        return None

    def retrieve_explanation(self, task_id: str, task_description: str, hugginggpt_output: str, base_explanation: str) -> Dict[str, Any]:
        """Check if a similar explanation exists, otherwise save base explanation"""
        task_text = mask_paths(f"{task_description}\n{hugginggpt_output}")
        similar = self.find_similar(task_text)

        if similar:
            return {
                "mode": "retrieved",
                "entry": similar["entry"],
                "cosine_similarity": similar["cosine_similarity"]
            }

        emb = self.embed(task_text).tolist()

        entry = {
            "task_id": task_id, 
            "task_description": mask_paths(task_description),
            "hugginggpt_output": mask_paths(hugginggpt_output),
            "explanation": base_explanation,
            "embedding": emb
        }
        self.save_entry(entry)
        return {"mode": "base", "entry": entry, "cosine_similarity": 0.0}
    
    def update_explanation(self, task_id: str, new_explanation: str):
        """
        Update the explanation of an existing entry by matching the exact task_id. Rewrites the JSONL file.
        """
        db = self.load_db()

        updated = False
        new_lines = []

        for entry in db:
            # Match criteria — EXACT task_id match
            if entry.get("task_id") == task_id:
                entry["explanation"] = new_explanation
                updated = True

            new_lines.append(entry)

        # If nothing matched, log a warning but do NOT fail
        if not updated:
            self.logger.warning(
                f"No existing entry found to update for task_id='{task_id}...'"
            )
        else:
            self.logger.info(f"Updated explanation for: {task_id}...")

        # Rewrite the DB file safely
        with open(self.db_path, "w", encoding="utf-8") as f:
            for entry in new_lines:
                # also mask before saving
                sanitized = self._mask_entry_strings(entry)
                f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
