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

    def decide(self, base_explanation: str, retrieved_explanation: str) -> Dict[str, str]:
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
        - Prefer explanations that give correct, concrete next steps that are directly supported by the System Output.
        
        Path masking note:
        - Explanation B may contain masked placeholders like "[PATH]".
        - Do NOT reward Explanation A for containing concrete file paths.
        - Do NOT penalize B for having "[PATH]" when the surrounding instructions/content are otherwise correct.
        - When scoring correctness/completeness/clarity/usefulness, treat a correct “[PATH]” placeholder as equivalent to the corresponding concrete path (i.e., paths carry zero scoring weight unless the path itself is the point of the instruction, e.g., “open/save at X”).
        - When producing "improved_explanation", you MUST output a version with no “[PATH]” placeholders.
        - If Explanation B contains “[PATH]”, replace each placeholder with the correct concrete path copied verbatim from Explanation A whenever a matching path can be confidently identified from local context.
        - If the winner is A but A contains masked placeholders (rare), apply the same replacement logic using B as the source if it contains the concrete path.
        - If you cannot confidently identify a concrete path for a placeholder, rewrite that sentence to avoid needing a path (e.g., “the generated file was saved to the output directory”) rather than leaving “[PATH]” in the final text.
        - Do NOT replace by order. Match each placeholder to the best-fitting path using the local surrounding context (nearby filenames, tool names, modules, commands).

        Scoring rubric (0-5 each; integers only):
        - correctness: factual/technical correctness relative to System Output
        - completeness: covers the key points needed to understand the output and what to do next
        - clarity: easy to follow, well-structured, minimal ambiguity
        - usefulness: actionable next steps and diagnostic value
        Total score = sum (0-20).

        Decision rule:
        - Compute total score for A and B.
        - If one total score is higher by 3 or more, pick it.
        - After scoring, if B is the winner (or near-tie preference selects B), choose "winner": "B" even if you injected concrete paths from A into the final explanation.
        - Only choose "combined" if you genuinely merge substantive non-path content from both A and B (not just path substitution).
        - If totals differ by 0-1 (near-tie), prefer the explanation that is more grounded in System Output, has higher information density, and adds non-overlapping, decision-relevant detail (i.e., avoids redundant restatement).
        - If one explanation is mostly a stylistic variant of the other, prefer the one that provides clearer actionable steps and better generalizability.
        - Choose "combined" when merging yields new non-redundant coverage (missing steps, key caveats, or error diagnosis) and improves correctness/completeness/usefulness—not by adding paths or repeating content.

        Grade mapping from total score:
        - A: 18-20
        - B: 15-17
        - C: 12-14
        - D: 8-11
        - E: 4-7
        - F: 0-3

        VERY IMPORTANT RULES:
        - The returned JSON MUST contain plain strings only.
        - "improved_explanation" MUST NOT be a dict, list, object, or nested JSON.
        - No markdown, no commentary, no code fences.
        
        Respond ONLY with strict JSON with keys:
        - "rating_A": letter A-F
        - "rating_B": letter A-F
        - "winner": "A", "B", or "combined"
        - "reason": short explanation of your choice
        - "improved_explanation": final natural-language explanation as a string
        
        Explanation A:
        {base_explanation}

        Explanation B:
        {retrieved_explanation}
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
