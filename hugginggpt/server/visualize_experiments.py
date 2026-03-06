# ============================================================
# Headless, NeurIPS-style visualization script
# ============================================================

import os
import json
import glob
import math
import numpy as np
import pandas as pd
import argparse
from difflib import SequenceMatcher

# -----------------------------
# Force headless backend
# -----------------------------
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuration
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="runs/20260214_095426_e7f9",
                    help="Run directory, e.g. runs/20260214_095426_e7f9")
args = parser.parse_args()

RUN_DIR = args.run.rstrip("/")

LOG_PATTERN = os.path.join(RUN_DIR, "iterations", "iteration_*.jsonl")
OUT_DIR = os.path.join(RUN_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# Prompt subgroup for cosine stability plot
STABLE_PROMPT_IDS = [0, 1, 2, 3, 4, 5, 6, 20, 25]

# Grade mapping
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}

TEST_PROMPTS_PATH = "./test_prompts.json"
TEST_PROMPTS_REPHRASED_PATH = "./test_prompts_rephrased.json"

# Colorblind-friendly-ish colors (Okabe-Ito style)
CB_GREEN = "#009E73"  # bluish green
CB_RED   = "#D55E00"  # vermillion (reads as red/orange, very colorblind-safe)
CB_BLUE  = "#0072B2"
CB_PURPLE= "#CC79A7"
CB_GRAY  = "#666666"

from cycler import cycler
plt.rcParams["axes.prop_cycle"] = cycler(color=[CB_BLUE, CB_GREEN, CB_RED, CB_PURPLE, "black", CB_GRAY])

# ============================================================
# NeurIPS-style plotting defaults
# ============================================================

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300, 
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# Utility functions
# ============================================================

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    plt.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.close()

def mean_ci(series):
    """
    Returns mean and 95% confidence interval (normal approximation).
    """
    series = series.dropna()
    n = len(series)
    if n == 0:
        return np.nan, np.nan
    mean = series.mean()
    std = series.std(ddof=1)
    ci = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return mean, ci

REPHRASED_ITS = [6, 7]

def mark_rephrased_iters():
    """
    Visually highlight iterations that use rephrased prompts (6 and 7) on *all* plots.
    """
    ax = plt.gca()
    # Shade the region covering iterations 6 and 7
    ax.axvspan(5.5, 7.5, alpha=0.12)
    # Add vertical reference lines at 6 and 7
    for it in REPHRASED_ITS:
        ax.axvline(it, linestyle="--", linewidth=1.0, alpha=0.7)

# ============================================================
# Load data
# ============================================================

records = []
for file in glob.glob(LOG_PATTERN):
    with open(file, "r") as f:
        for line in f:
            records.append(json.loads(line))

if not records:
    raise RuntimeError("No experiment logs found.")

df = pd.DataFrame(records)

# ============================================================
# Preprocessing
# ============================================================

df["iteration"] = df["iteration"].astype(int)

df["grade_base_num"] = df["judge"].apply(
    lambda j: GRADE_MAP.get(j.get("grade_base"))
)
df["grade_retrieved_num"] = df["judge"].apply(
    lambda j: GRADE_MAP.get(j.get("grade_retrieved"))
)

df["winner"] = df["judge"].apply(lambda j: j.get("winner"))
df["cosine_similarity"] = df["retriever"].apply(
    lambda r: r.get("cosine_similarity", 0.0)
)

# --- Prompt match analysis ---
df["masked_prompt"] = df.get("masked_prompt")

df["retrieved_prompt"] = df["retriever"].apply(
    lambda r: r.get("retrieved_prompt") if isinstance(r, dict) else None
)

def close_ratio(a, b) -> float:
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
        return np.nan
    return SequenceMatcher(None, a, b).ratio()

df["prompt_match_ratio"] = df.apply(
    lambda row: close_ratio(row["masked_prompt"], row["retrieved_prompt"]),
    axis=1
)

# "same" means exact equality after stripping whitespace
df["prompt_same"] = df.apply(
    lambda row: (
        isinstance(row["masked_prompt"], str) and
        isinstance(row["retrieved_prompt"], str) and
        row["masked_prompt"].strip() == row["retrieved_prompt"].strip()
    ),
    axis=1
)

# Overall stats
same_count = int(df["prompt_same"].sum())
both_present = int(df[["masked_prompt", "retrieved_prompt"]].notna().all(axis=1).sum())
same_rate = (same_count / both_present) if both_present > 0 else 0.0

# Write mismatches
mismatch_path = os.path.join(OUT_DIR, "prompt_mismatches.txt")
mismatches = df[
    df[["masked_prompt", "retrieved_prompt"]].notna().all(axis=1) & (~df["prompt_same"])
].copy()

mismatches = mismatches.sort_values(
    by=["prompt_match_ratio"], ascending=True
)

with open(mismatch_path, "w", encoding="utf-8") as f:
    f.write(f"Run: {RUN_DIR}\n")
    f.write(f"Exact same (strip) count: {same_count}\n")
    f.write(f"Both present count: {both_present}\n")
    f.write(f"Exact same rate: {same_rate:.4f}\n\n")
    f.write("MISMATCHES (sorted by lowest close-match ratio first)\n")
    f.write("=" * 80 + "\n\n")

    for _, row in mismatches.iterrows():
        pid = row.get("prompt_id", "NA")
        it = row.get("iteration", "NA")
        ratio = row.get("prompt_match_ratio", np.nan)

        mp = row.get("masked_prompt", "")
        rp = row.get("retrieved_prompt", "")

        # short previews to keep file readable
        mp_prev = (mp[:300] + "…") if isinstance(mp, str) and len(mp) > 300 else mp
        rp_prev = (rp[:300] + "…") if isinstance(rp, str) and len(rp) > 300 else rp

        f.write(f"prompt_id={pid} iteration={it} match_ratio={ratio:.4f}\n")
        f.write("masked_prompt:\n")
        f.write(mp_prev + "\n")
        f.write("retrieved_prompt:\n")
        f.write(rp_prev + "\n")
        f.write("-" * 80 + "\n\n")

print(f"[OK] Prompt exact-same rate: {same_rate:.3f} ({same_count}/{both_present})")
print(f"[OK] Mismatches written to: {mismatch_path}")

# ============================================================
# 1. Average Grades per Iteration (with CI)
# ============================================================

stats = []

for it, g in df.groupby("iteration"):
    mb, cib = mean_ci(g["grade_base_num"])
    mr, cir = mean_ci(g["grade_retrieved_num"])
    stats.append({
        "iteration": it,
        "base_mean": mb,
        "base_ci": cib,
        "retr_mean": mr,
        "retr_ci": cir
    })

stats = pd.DataFrame(stats).sort_values("iteration")

plt.figure()

plt.errorbar(
    stats["iteration"], stats["base_mean"],
    yerr=stats["base_ci"],
    marker="o",
    label="Base",
    color="#D55E00"  # colorblind-safe red (vermillion)
)

plt.errorbar(
    stats["iteration"], stats["retr_mean"],
    yerr=stats["retr_ci"],
    marker="o",
    label="Retrieved",
    color="#009E73"  # colorblind-safe green
)

plt.xlabel("Iteration")
plt.ylabel("Average Grade")
plt.title("Average Explanation Quality per Iteration")

# Reverse grade scale: 1 at top, 6 at bottom
plt.ylim(2.4, 1.6)

plt.legend()
mark_rephrased_iters()
savefig("avg_grades_per_iteration")

# ============================================================
# 2. Base vs Retrieved (Judge Winner Count) per Iteration
# ============================================================

winner_counts = (
    df.groupby(["iteration", "winner"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

ordered_cols = [c for c in ["base", "combined", "retrieved"] if c in winner_counts.columns]
winner_counts = winner_counts[ordered_cols]

color_map = {
    "retrieved": CB_GREEN,
    "base": CB_RED,
    "combined": CB_BLUE,   # or CB_GRAY
}

ax = winner_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(6, 4),
    color=[color_map[c] for c in winner_counts.columns],
)

plt.xlabel("Iteration")
plt.ylabel("Count")
plt.title("Judge Decisions per Iteration")
plt.legend(title="Winner")

# Mark iterations 6 and 7 by bar *index* (works even if x is categorical)
# winner_counts.index contains the actual iteration numbers in order
idx = list(winner_counts.index)
for it in REPHRASED_ITS:
    if it in idx:
        j = idx.index(it)                  # bar position (0..n-1)
        ax.axvspan(j - 0.5, j + 0.5, alpha=0.12)
        ax.axvline(j, linestyle="--", linewidth=1.0, alpha=0.7)

savefig("judge_winner_counts")

# ============================================================
# 3. RAG Reliance per Iteration (winner == retrieved)
# ============================================================

# --- RAG Reliance per Iteration: stacked-area with two lines
# retrieved line at r
# combined line at r + c   (where c is your "combined" metric, NOT 1-r)

rag_stats = []
for it, g in df.groupby("iteration"):
    r = (g["winner"] == "retrieved").astype(int)
    c = (g["winner"] == "combined").astype(int)   # <-- assumes your logs use winner == "combined"
    r_mean, _ = mean_ci(r)
    c_mean, _ = mean_ci(c)
    rag_stats.append({"iteration": it, "retr_mean": r_mean, "comb_mean": c_mean})

rag_stats = pd.DataFrame(rag_stats).sort_values("iteration")

plt.figure()
x = rag_stats["iteration"].to_numpy()
y_retr = rag_stats["retr_mean"].to_numpy()
y_comb = rag_stats["comb_mean"].to_numpy()
y_top = y_retr + y_comb

# Area 1: under retrieved
plt.fill_between(x, 0, y_retr, alpha=0.85, label="Retrieved", color="#009E73")

# Area 2: between retrieved and retrieved+combined
plt.fill_between(x, y_retr, y_top, alpha=0.55, label="Combined (stacked)", color="#D55E00")

# Boundary lines (no points)
plt.plot(x, y_retr, linewidth=1.8, label="Retrieved (boundary)", color="#00916A")
plt.plot(x, y_top, linewidth=1.8, label="Retrieved + Combined (boundary)", color="#C55500")

plt.xlabel("Iteration")
plt.ylabel("Fraction")
plt.ylim(0, 1)
plt.title("RAG Reliance per Iteration")
plt.legend()
mark_rephrased_iters()
savefig("rag_reliance")

# ============================================================
# 4. Cosine Similarity per Iteration (with CI)
# ============================================================

cos_stats = []

for it, g in df.groupby("iteration"):
    mean, ci = mean_ci(g["cosine_similarity"])
    cos_stats.append({
        "iteration": it,
        "mean": mean,
        "ci": ci
    })

cos_stats = pd.DataFrame(cos_stats).sort_values("iteration")

plt.figure()
plt.errorbar(
    cos_stats["iteration"],
    cos_stats["mean"],
    yerr=cos_stats["ci"],
    marker="o",
    color="#009E73"
)
plt.xlabel("Iteration")
plt.ylabel("Cosine Similarity")
plt.title("Retriever Cosine Similarity per Iteration")
mark_rephrased_iters()
savefig("cosine_similarity_per_iteration")

# ============================================================
# 5. Stability of Cosine Similarity (Task Subgroup)
# ============================================================

sub_df = df[df["prompt_id"].isin(STABLE_PROMPT_IDS)]

plt.figure(figsize=(6, 4))
sns.lineplot(
    data=sub_df,
    x="iteration",
    y="cosine_similarity",
    hue="prompt_id",
    marker="o",
    legend="full",
    palette="colorblind"
)
plt.xlabel("Iteration")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity Stability (Task Subgroup)")
plt.legend(title="Prompt ID")
mark_rephrased_iters()
savefig("cosine_stability_subgroup")

# ============================================================
# 6. Prompt match per Iteration
# ============================================================

pm_stats = []
for it, g in df.groupby("iteration"):
    # exact same fraction (only where both strings exist)
    both = g[["masked_prompt", "retrieved_prompt"]].notna().all(axis=1)
    if both.any():
        exact_frac = float(g.loc[both, "prompt_same"].mean())
        mean_ratio, ci_ratio = mean_ci(g.loc[both, "prompt_match_ratio"])
    else:
        exact_frac = np.nan
        mean_ratio, ci_ratio = np.nan, np.nan

    pm_stats.append({
        "iteration": it,
        "exact_same_frac": exact_frac,
        "mean_ratio": mean_ratio,
        "ci_ratio": ci_ratio
    })

pm_stats = pd.DataFrame(pm_stats).sort_values("iteration")

# (a) exact same fraction
plt.figure()
plt.plot(pm_stats["iteration"], pm_stats["exact_same_frac"], marker="o", color=CB_GREEN)
plt.xlabel("Iteration")
plt.ylabel("Exact Same Fraction")
plt.ylim(0, 1)
plt.title("Exact Match: masked_prompt vs retrieved_prompt")
mark_rephrased_iters()
savefig("prompt_exact_match_per_iteration")

# (b) close-match ratio (with CI)
plt.figure()
plt.errorbar(
    pm_stats["iteration"], pm_stats["mean_ratio"],
    yerr=pm_stats["ci_ratio"], marker="o",
    color="#009E73"
)
plt.xlabel("Iteration")
plt.ylabel("Close-Match Ratio")
plt.ylim(0, 1)
plt.title("Close Match Ratio: masked_prompt vs retrieved_prompt")
mark_rephrased_iters()
savefig("prompt_close_match_ratio_per_iteration")

has_both = df[["masked_prompt", "retrieved_prompt"]].notna().all(axis=1)

rows = []
for it, g in df[has_both].groupby("iteration"):
    g_exact = g[g["prompt_same"]]
    g_wrong = g[~g["prompt_same"]]
    rows.append({
        "iteration": it,
        "win_rate_exact": (g_exact["winner"] == "retrieved").mean() if len(g_exact) else np.nan,
        "win_rate_wrong": (g_wrong["winner"] == "retrieved").mean() if len(g_wrong) else np.nan,
        "n_exact": len(g_exact),
        "n_wrong": len(g_wrong),
    })

wr = pd.DataFrame(rows).sort_values("iteration")

plt.figure()
plt.plot(
    wr["iteration"], wr["win_rate_exact"],
    marker="o",
    color=CB_GREEN,
    label="P(Judge picks Retrieved | exact retrieval)"
)
plt.plot(
    wr["iteration"], wr["win_rate_wrong"],
    marker="o",
    color=CB_RED,
    label="P(Judge picks Retrieved | non-exact retrieval)"
)
plt.xlabel("Iteration")
plt.ylabel("Retrieved win rate")
plt.ylim(0, 1)
plt.title("Judge preference for Retrieved under exact vs non-exact retrieval")
plt.legend()
savefig("retrieved_win_rate_exact_vs_wrong")

# --- Heatmap: 4 regimes per iteration (fractions) ---
use = df[has_both].copy()

use["retrieval_correct"] = np.where(use["prompt_same"], "Exact", "Non-exact")
use["judge_winner"] = np.where(use["winner"] == "retrieved", "Retrieved wins", "Base wins")

# Counts per iteration x (retrieval_correct, judge_winner)
counts = (
    use.groupby(["iteration", "retrieval_correct", "judge_winner"])
    .size()
    .reset_index(name="count")
)

# Pivot into 4 columns
pivot_counts = counts.pivot_table(
    index="iteration",
    columns=["retrieval_correct", "judge_winner"],
    values="count",
    fill_value=0
).sort_index()

# Convert to within-iteration fractions (so each iteration sums to 1)
pivot_frac = pivot_counts.div(pivot_counts.sum(axis=1), axis=0)

# Flatten column names for nicer heatmap labels
pivot_frac.columns = [f"{a} | {b}" for a, b in pivot_frac.columns]

n_iter = len(pivot_frac)

plt.figure(figsize=(10, max(6, 0.6 * n_iter)))
sns.heatmap(
    pivot_frac,
    vmin=0, vmax=1,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Fraction"}
)
plt.xlabel("Regime (retrieval correctness | judge decision)")
plt.ylabel("Iteration")
plt.title("RAG regimes per iteration")
mark_rephrased_iters()
savefig("heatmap_rag_regimes_per_iteration")

def load_expected_prompts(path: str) -> dict:
    """
    Returns mapping: prompt_string -> expected_models_list
    Accepts either:
      - JSON list of objects
      - JSON object with a top-level key containing the list
    Each item expected to have fields: "prompt", "models" (list of HF ids)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # try common containers
        for k in ["prompts", "data", "items", "test_prompts"]:
            if k in data and isinstance(data[k], list):
                data = data[k]
                break

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")

    m = {}
    for item in data:
        p = item.get("prompt")
        models = item.get("models", [])
        if isinstance(p, str):
            m[p.strip()] = [x.strip() for x in models if isinstance(x, str)]
    return m

expected_base = load_expected_prompts(TEST_PROMPTS_PATH)
expected_rephr = load_expected_prompts(TEST_PROMPTS_REPHRASED_PATH)

def extract_predicted_models(expert_output) -> list:
    """
    Extract all selected model ids from your log structure:
      expert_output: {"0": {"choose model result": {"id": "..."} ...}, ...}
    Returns list of model ids (may be empty).
    """
    models = []
    if not isinstance(expert_output, dict):
        return models

    for _, task_obj in expert_output.items():
        if not isinstance(task_obj, dict):
            continue

        cmr = task_obj.get("choose model result")
        if isinstance(cmr, dict):
            mid = cmr.get("id")
            if isinstance(mid, str) and mid.strip():
                models.append(mid.strip())
        elif isinstance(cmr, str) and cmr.strip():
            models.append(cmr.strip())

        # Some variants might store selected model under other keys
        mid2 = task_obj.get("model_id") or task_obj.get("model")
        if isinstance(mid2, str) and mid2.strip():
            models.append(mid2.strip())

    # de-duplicate while preserving order
    seen = set()
    out = []
    for m in models:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out

ERROR_KEYS = {"error", "exception", "traceback", "stderr"}
def task_inference_failed(task_obj: dict) -> bool:
    """
    Heuristic: failure if inference result missing/empty OR contains error-like keys.
    """
    if not isinstance(task_obj, dict):
        return True

    inf = task_obj.get("inference result", None)

    # explicit error-like keys anywhere at top level or inside inference result
    for k in ERROR_KEYS:
        if k in task_obj:
            return True
    if isinstance(inf, dict):
        for k in ERROR_KEYS:
            if k in inf:
                return True

    # missing or empty inference result
    if inf is None:
        return True
    if isinstance(inf, (dict, list, str)) and len(inf) == 0:
        return True

    return False

def record_inference_failure_rate(expert_output) -> float:
    """
    Returns fraction of tasks in expert_output that look failed.
    If there are no tasks, returns 1.0 (treat as failure).
    """
    if not isinstance(expert_output, dict) or len(expert_output) == 0:
        return 1.0

    fails = 0
    total = 0
    for _, task_obj in expert_output.items():
        if not isinstance(task_obj, dict):
            fails += 1
            total += 1
            continue
        total += 1
        if task_inference_failed(task_obj):
            fails += 1

    return fails / total if total > 0 else 1.0

def get_expected_models_for_row(row) -> list:
    """
    Use rephrased expected prompts only for iterations 6 and 7.
    Match exact prompt string.
    """
    p = row.get("prompt")
    if not isinstance(p, str):
        return []
    key = p.strip()
    it = int(row.get("iteration", -1))
    if it in REPHRASED_ITS:
        return expected_rephr.get(key, [])
    return expected_base.get(key, [])

df["expected_models"] = df.apply(get_expected_models_for_row, axis=1)
df["predicted_models"] = df["expert_output"].apply(extract_predicted_models)

def model_hit(expected, predicted) -> float:
    """
    HIT: at least one expected model is among predicted models.
    (robust when your system runs multiple helpers)
    """
    if not isinstance(expected, list) or not expected:
        return np.nan
    if not isinstance(predicted, list):
        return 0.0
    e = set(expected)
    p = set(predicted)
    return 1.0 if len(e.intersection(p)) > 0 else 0.0

def model_exact(expected, predicted) -> float:
    """
    EXACT: predicted model set equals expected model set.
    """
    if not isinstance(expected, list) or not expected:
        return np.nan
    if not isinstance(predicted, list):
        return 0.0
    return 1.0 if set(expected) == set(predicted) else 0.0

df["model_hit"] = df.apply(lambda r: model_hit(r["expected_models"], r["predicted_models"]), axis=1)
df["model_exact"] = df.apply(lambda r: model_exact(r["expected_models"], r["predicted_models"]), axis=1)
df["inference_failure_rate"] = df["expert_output"].apply(record_inference_failure_rate)

# ------------------------------------------------------------
# Write model mismatch report (for quick debugging)
# ------------------------------------------------------------
model_mismatch_path = os.path.join(OUT_DIR, "model_mismatches.txt")
mm = df[df["model_hit"].notna() & (df["model_hit"] == 0.0)].copy()

with open(model_mismatch_path, "w", encoding="utf-8") as f:
    f.write(f"Run: {RUN_DIR}\n")
    f.write(f"Model HIT==0 count: {len(mm)} / {int(df['model_hit'].notna().sum())}\n\n")
    f.write("MISSES (HIT==0)\n")
    f.write("=" * 80 + "\n\n")
    for _, row in mm.sort_values(["iteration", "prompt_id"]).iterrows():
        f.write(f"iteration={row.get('iteration')} prompt_id={row.get('prompt_id')}\n")
        f.write("prompt:\n")
        f.write((row.get("prompt") or "") + "\n")
        f.write(f"expected_models: {row.get('expected_models')}\n")
        f.write(f"predicted_models: {row.get('predicted_models')}\n")
        f.write("-" * 80 + "\n\n")

print(f"[OK] Model mismatches written to: {model_mismatch_path}")

# ============================================================
# NEW PLOTS
# ============================================================

# 7. Model Hit/Miss rate per iteration (with CI)
hit_stats = []
for it, g in df.groupby("iteration"):
    m, ci = mean_ci(g["model_hit"])
    hit_stats.append({"iteration": it, "mean": m, "ci": ci})
hit_stats = pd.DataFrame(hit_stats).sort_values("iteration")

plt.figure()
plt.errorbar(hit_stats["iteration"], hit_stats["mean"], yerr=hit_stats["ci"], marker="o", color="#009E73")
plt.xlabel("Iteration")
plt.ylabel("Hit rate")
plt.ylim(0, 1)
plt.title("Model Selection Hit Rate per Iteration")
mark_rephrased_iters()
savefig("model_hit_rate_per_iteration")

# 8. Model Exact-set match rate per iteration (with CI)
exact_stats = []
for it, g in df.groupby("iteration"):
    m, ci = mean_ci(g["model_exact"])
    exact_stats.append({"iteration": it, "mean": m, "ci": ci})
exact_stats = pd.DataFrame(exact_stats).sort_values("iteration")

plt.figure()
plt.errorbar(exact_stats["iteration"], exact_stats["mean"], yerr=exact_stats["ci"], marker="o", color="#009E73")
plt.xlabel("Iteration")
plt.ylabel("Exact match rate")
plt.ylim(0, 1)
plt.title("Model Selection Exact-Set Match per Iteration")
mark_rephrased_iters()
savefig("model_exact_match_rate_per_iteration")

# 9. Inference failure rate per iteration (with CI)
fail_stats = []
for it, g in df.groupby("iteration"):
    m, ci = mean_ci(g["inference_failure_rate"])
    fail_stats.append({"iteration": it, "mean": m, "ci": ci})
fail_stats = pd.DataFrame(fail_stats).sort_values("iteration")

plt.figure()
plt.errorbar(fail_stats["iteration"], fail_stats["mean"], yerr=fail_stats["ci"], marker="o", color="#009E73")
plt.xlabel("Iteration")
plt.ylabel("Failure rate")
plt.ylim(0, 1)
plt.title("Inference Failure Rate per Iteration")
mark_rephrased_iters()
savefig("inference_failure_rate_per_iteration")

# 10. (Optional) Stacked bar: Hits vs Misses per iteration
hm = (
    df[df["model_hit"].notna()]
    .assign(hit=lambda x: (x["model_hit"] == 1.0).astype(int),
            miss=lambda x: (x["model_hit"] == 0.0).astype(int))
    .groupby("iteration")[["hit", "miss"]]
    .sum()
    .sort_index()
)

ax = hm.plot(kind="bar", stacked=True, figsize=(6, 4),
             color=[CB_GREEN, CB_RED])
plt.xlabel("Iteration")
plt.ylabel("Count")
plt.title("Model Selection: Hit vs Miss (Counts)")
# highlight 6/7 in categorical space
idx = list(hm.index)
for it in REPHRASED_ITS:
    if it in idx:
        j = idx.index(it)
        ax.axvspan(j - 0.5, j + 0.5, alpha=0.12)
        ax.axvline(j, linestyle="--", linewidth=1.0, alpha=0.7)
savefig("model_hit_miss_counts")


# ============================================================
# Done
# ============================================================

print(f"[OK] Figures saved to ./{OUT_DIR}/ (PNG + PDF)")
