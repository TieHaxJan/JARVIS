import json
import os
import logging
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ExplanationRetriever:
    def __init__(self, db_path="data/explanations.jsonl", model_name="all-MiniLM-L6-v2", threshold=0.8):
        self.db_path = db_path
        self.embedder = SentenceTransformer(model_name)
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

        # DB anlegen, falls nicht existiert
        if not os.path.exists(self.db_path):
            open(self.db_path, "w").close()
            self.logger.debug(f"Created new explanations DB at {self.db_path}")

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, convert_to_numpy=True)

    def load_db(self):
        with open(self.db_path, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        self.logger.debug(f"Loaded {len(entries)} entries from {self.db_path}")
        return entries

    def save_entry(self, entry: Dict[str, Any]):
        with open(self.db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.logger.info(f"Saved new explanation to {self.db_path}: {entry['explanation'][:60]}...")

    def find_similar(self, query_text: str) -> Optional[Dict[str, Any]]:
        db = self.load_db()
        if not db:
            self.logger.debug("DB empty → no similar entries")
            return None
        query_emb = self.embed(query_text).reshape(1, -1)

        sims = []
        for entry in db:
            if "embedding" not in entry:
                continue
            emb = np.array(entry["embedding"]).reshape(1, -1)
            sim = cosine_similarity(query_emb, emb)[0][0]
            sims.append((sim, entry))

        if not sims:
            self.logger.debug("No valid embeddings found in DB")
            return None

        best_sim, best_entry = max(sims, key=lambda x: x[0])
        self.logger.debug(f"Best similarity = {best_sim:.3f} (threshold = {self.threshold})")

        if best_sim >= self.threshold:
            self.logger.info(f"Retrieved similar explanation (sim={best_sim:.3f})")
            return best_entry
        return None

    def handle_task(self, task_id: str, task_description: str, hugginggpt_output: str, base_explainer_fn) -> Dict[str, Any]:
        """Check if a similar explanation exists, otherwise create one via base_explainer_fn."""
        task_text = f"{task_description}\n{hugginggpt_output}"
        similar = self.find_similar(task_text)

        if similar:
            return {"mode": "retrieved", "entry": similar}

        # sonst neue Base Explanation erzeugen
        explanation = base_explainer_fn(task_description, hugginggpt_output)
        emb = self.embed(task_text).tolist()

        entry = {
            "task_id": task_id,
            "task_description": task_description,
            "hugginggpt_output": hugginggpt_output,
            "explanation": explanation,
            "embedding": emb
        }
        self.save_entry(entry)
        return {"mode": "base", "entry": entry}