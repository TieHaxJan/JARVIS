import logging
import json
from typing import Dict

logger = logging.getLogger(__name__)

class ExplanationJudge:
    def __init__(self, chitchat_fn, api_key: str, api_type: str, api_endpoint: str, model: str):
        self.chitchat_fn = chitchat_fn
        self.api_key = api_key
        self.api_type = api_type
        self.api_endpoint = api_endpoint
        self.model = model

    def decide(self, task_description: str, hugginggpt_output: str,
               base_explanation: str, retrieved_explanation: str) -> Dict[str, str]:
        """
        Compare base vs. retrieved explanation via LLM judge.
        Returns {"winner": "base"|"retrieved", "reason": "..."}
        """

        prompt = f"""
        You are a judge. Your job is to compare two explanations of a system decision.

        Task: {task_description}
        HuggingGPT Output: {hugginggpt_output}

        Explanation A (Base):
        {base_explanation}

        Explanation B (Retrieved):
        {retrieved_explanation}

        Which explanation is clearer and more user-friendly for a non-technical person? Also pay special attention to if the explanation matches the task at hand, i.e. if a specific image is mentioned the explanation should mention the same image.
        Respond in JSON with keys "winner" ("A" or "B") and "reason". Don't use markdown fences! Just provide the JSON content.
        """

        messages = [{"role": "user", "content": prompt}]
        response = self.chitchat_fn(messages, self.api_key, self.api_type, self.api_endpoint)

        try:
            parsed = json.loads(response)
            winner = parsed.get("winner", "A")
            reason = parsed.get("reason", "")
        except Exception:
            # fallback: pick A by default if LLM answer not JSON
            logger.warning(f"Judge response not valid JSON: {response}")
            winner, reason = "A", "fallback"

        if winner not in ["A", "B"]:
            winner = "A"

        logger.info(f"Judge decided: {winner} ({reason})")

        return {"winner": "base" if winner == "A" else "retrieved", "reason": reason}
