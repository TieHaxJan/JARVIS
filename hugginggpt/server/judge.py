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
        You are a judge and an explainer. Your job is to compare two explanations of a system decision and, if possible, improve them.

        Task: {task_description}
        HuggingGPT Output: {hugginggpt_output}

        Explanation A (Base):
        {base_explanation}

        Explanation B (Retrieved):
        {retrieved_explanation}

        Step 1: Evaluate which explanation is clearer, more user-friendly, and more accurate for a non-technical audience.
        Also assess which one best matches the task context (e.g., image, text, or audio mentioned).

        Step 2: If both explanations have useful complementary information, combine them into a single improved explanation.
        This new explanation should be natural, human-readable, concise, and faithful to the actual system behavior.

        Respond in strict JSON format with the following keys:
        - "winner": "A" if the base explanation is better, "B" if the retrieved one is better, or "combined" if the improved version merges both.
        - "reason": a short justification for your choice.
        - "improved_explanation": the final, improved explanation (even if it's mostly from one source).

        Do not use markdown or extra commentary — only return the JSON object.
        """

        messages = [{"role": "user", "content": prompt}]
        response = self.chitchat_fn(messages, self.api_key, self.api_type, self.api_endpoint)
        response = response.strip()
        if response.startswith("```"):
            response = response.replace("```json", "")
            response = response.replace("```", "")
            response = response.strip()

        try:
            parsed = json.loads(response)
            winner = parsed.get("winner", "A")
            reason = parsed.get("reason", "")
            improved_explanation = parsed.get("improved_explanation", "")
        except Exception:
            logger.warning(f"Judge response not valid JSON: {response}")
            winner, reason, improved_explanation = "A", "fallback", ""

        # Normalize winner values
        if winner not in ["A", "B", "combined"]:
            winner = "A"

        # Map to internal labels
        if winner == "A":
            winner_label = "base"
        elif winner == "B":
            winner_label = "retrieved"
        else:
            winner_label = "combined"

        logger.info(f"Judge decided: {winner_label} ({reason})")

        return {
            "winner": winner_label,
            "reason": reason,
            "improved_explanation": improved_explanation.strip()
        }
