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
        Compare two explanations using an LLM judge.
        Adds:
        - A-F scoring for each explanation
        - Combining only when the merged explanation is strictly better
        - Using a single explanation if it is already sufficient
        """

        prompt = f"""
        You are an expert evaluator of explanations. 
        You evaluate two explanations (A and B) based on clarity, correctness, completeness, and usefulness.
        
        Bias control:
        - Do not prefer A or B due to label, origin, or writing style.
        - Anchor judgments to the System Output: reward only claims supported by it.
        - Avoid length bias: do not reward verbosity; prefer higher information density.
        - Penalize redundancy: if an explanation mostly restates what the other already said, score it lower.
        - If both are similarly correct, prefer the explanation that adds more verifiable, task-relevant detail and clearer next steps.

        Your tasks:
        1. Provide a letter-grade rating (A-F) for each explanation:
           - A = excellent
           - B = good
           - C = acceptable
           - D = weak
           - E = poor
           - F = unusable

        2. Decide whether A or B is better.
           If both are weak (C or worse) but contain complementary strengths, you MAY combine them.

        3. Only choose "combined" if the merged explanation is meaningfully better than both A and B individually.
           If a single explanation is already sufficient (B or better), choose ONLY that one.

        4. Produce an improved or final explanation:
           - If winner = "A": improved_explanation = Explanation A (possibly lightly polished)
           - If winner = "B": improved_explanation = Explanation B (possibly lightly polished)
           - If winner = "combined": merge them into a single, better explanation

        VERY IMPORTANT RULES:
        - The returned JSON MUST contain plain strings only.
        - "improved_explanation" MUST NOT be a dict, list, object, or nested JSON.
        - No markdown, no commentary, no code fences.

        Task description:
        {task_description}

        System Output:
        {hugginggpt_output}

        Explanation A:
        {base_explanation}

        Explanation B:
        {retrieved_explanation}

        Respond ONLY with strict JSON with keys:
        - "rating_A": letter A-F
        - "rating_B": letter A-F
        - "winner": "A", "B", or "combined"
        - "reason": short explanation of your choice
        - "improved_explanation": final natural-language explanation as a string
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
            rating_A = parsed.get("rating_A", "C")
            rating_B = parsed.get("rating_B", "C")
            improved_explanation = parsed.get("improved_explanation", "")
        except Exception:
            logger.warning(f"Judge response not valid JSON: {response}")
            winner, reason = "A", "fallback"
            rating_A, rating_B = "C", "C"
            improved_explanation = base_explanation

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

        logger.info(f"Judge decided: {winner_label} (A={rating_A}, B={rating_B}) reason={reason})")
        
        # --- SAFELY NORMALIZE THE IMPROVED EXPLANATION --------------------
        # LLMs sometimes return a dict, list, number, or mixed JSON instead of a string.
        # Guarantee that improved_explanation is ALWAYS a plain string.
        if isinstance(improved_explanation, dict):
            improved_explanation = json.dumps(improved_explanation, ensure_ascii=False)
        elif isinstance(improved_explanation, list):
            improved_explanation = " ".join(str(x) for x in improved_explanation)
        elif improved_explanation is None:
            improved_explanation = ""
        else:
            improved_explanation = str(improved_explanation)
        # --------------------------------------------------------------------

        return {
            "winner": winner_label,
            "reason": reason,
            "rating_A": rating_A,
            "rating_B": rating_B,
            "improved_explanation": improved_explanation.strip()
        }
