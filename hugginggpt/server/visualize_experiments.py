import os
import json
import glob
import math
import numpy as np
import pandas as pd
import argparse
from difflib import SequenceMatcher
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuration
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="runs/20260317_135158_f32d",
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

CB_GREEN = "#009E73"
CB_RED   = "#D55E00"
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
    "font.size": 22,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "legend.fontsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.dpi": 300, 
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# Utility functions
# ============================================================

FIGSIZE = (8, 5)
BAR_WIDTH = 0.75
ALPHA_MAIN = 0.85
ALPHA_COMP = 0.45
CAPSIZE = 4
COUNT_YMAX = 100
RATE_YMAX = 1.0

def savefig(name):
    plt.tight_layout(pad=0.6)
    plt.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), bbox_inches="tight", dpi=300)
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
    Highlight iterations 6 and 7 for plots with numeric x-axis.
    """
    ax = plt.gca()
    ax.axvspan(5.5, 7.5, alpha=0.12, color=CB_GRAY)
    for it in REPHRASED_ITS:
        ax.axvline(it, linestyle="--", linewidth=1.0, alpha=0.7, color="black")

def mark_rephrased_iters_categorical(index_values):
    """
    Highlight iterations 6 and 7 for bar plots with categorical x-axis positions.
    """
    ax = plt.gca()
    idx = list(index_values)
    for it in REPHRASED_ITS:
        if it in idx:
            j = idx.index(it)
            ax.axvspan(j - 0.5, j + 0.5, alpha=0.12, color=CB_GRAY)
            ax.axvline(j, linestyle="--", linewidth=1.0, alpha=0.7, color="black")

def highlight_human_eval_rephrased(df_sorted):
    """
    Highlights the background for any X-axis labels belonging to iterations 6 or 7.
    df_sorted: The DataFrame used for the plot, must have an 'iteration' column.
    """
    ax = plt.gca()
    # Find indices where iteration is 6 or 7
    rephrased_indices = df_sorted.index[df_sorted['iteration'].isin(REPHRASED_ITS)].tolist()
    
    if rephrased_indices:
        # Create spans for contiguous blocks of rephrased iterations
        # (Since it's sorted, 6 and 7 will be at the end)
        start = min(rephrased_indices) - 0.5
        end = max(rephrased_indices) + 0.5
        ax.axvspan(start, end, alpha=0.12, color=CB_GRAY)
        
        # Add vertical dashed lines for each specific rephrased point
        for idx in rephrased_indices:
            ax.axvline(idx, linestyle="--", linewidth=1.0, alpha=0.5, color="black")

def stacked_rate_bar(
    df_plot,
    x_col,
    y_col,
    ylabel,
    filename,
    color=CB_GREEN,
    yerr_col=None,
    ylim=None,
    invert_y=False
):
    """
    Single-rate plot shown as stacked green/red bars:
    green = observed rate
    red   = complement to 1
    """
    plt.figure(figsize=FIGSIZE)

    x = df_plot[x_col].to_numpy()
    y = df_plot[y_col].to_numpy()

    plt.bar(
        x,
        y,
        width=BAR_WIDTH,
        color=color,
        alpha=ALPHA_MAIN,
        yerr=df_plot[yerr_col].to_numpy() if yerr_col is not None else None,
        capsize=CAPSIZE if yerr_col is not None else 0
    )

    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.xticks(x, rotation=0)

    if ylim is not None:
        plt.ylim(*ylim)

    if invert_y:
        plt.gca().invert_yaxis()

    mark_rephrased_iters()
    savefig(filename)

def grouped_two_bar(
    df_plot,
    x_col,
    y1_col,
    y2_col,
    ylabel,
    filename,
    y1err_col=None,
    y2err_col=None,
    ylim=None,
    invert_y=False,
    label1="Series 1",
    label2="Series 2"
):
    """
    Two-series grouped bars with fully consistent width/style.
    """
    plt.figure(figsize=FIGSIZE)

    x = df_plot[x_col].to_numpy()
    width = 0.36

    plt.bar(
        x - width/2,
        df_plot[y1_col].to_numpy(),
        width=width,
        color=CB_RED,
        alpha=ALPHA_MAIN,
        yerr=df_plot[y1err_col].to_numpy() if y1err_col is not None else None,
        capsize=CAPSIZE if y1err_col is not None else 0,
        label=label1
    )

    plt.bar(
        x + width/2,
        df_plot[y2_col].to_numpy(),
        width=width,
        color=CB_GREEN,
        alpha=ALPHA_MAIN,
        yerr=df_plot[y2err_col].to_numpy() if y2err_col is not None else None,
        capsize=CAPSIZE if y2err_col is not None else 0,
        label=label2
    )

    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.xticks(x, rotation=0)

    if ylim is not None:
        plt.ylim(*ylim)

    if invert_y:
        plt.gca().invert_yaxis()

    plt.legend()
    mark_rephrased_iters()
    savefig(filename)
    
def stacked_count_bar(
    df_plot,
    x_col,
    y_col,
    ylabel,
    filename,
    ymax=100
):
    """
    Single count plot shown as stacked green/red bars:
    green = observed count
    red   = remaining count to ymax
    """
    plt.figure(figsize=FIGSIZE)

    x = df_plot[x_col].to_numpy()
    y = df_plot[y_col].to_numpy()
    comp = ymax - y

    plt.bar(
        x, y,
        width=BAR_WIDTH,
        color=CB_GREEN
    )

    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.xticks(x, rotation=0)
    plt.ylim(0, ymax)

    mark_rephrased_iters()
    savefig(filename)

# ============================================================
# Load data
# ============================================================

records = []
for file in glob.glob(LOG_PATTERN):
    with open(file, "r", encoding="utf-8") as f:
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

plt.figure(figsize=FIGSIZE)

plt.errorbar(
    stats["iteration"], stats["base_mean"],
    yerr=stats["base_ci"],
    fmt="o",
    label="Base",
    linestyle="none",
    capsize=CAPSIZE,
    color=CB_RED 
)

plt.errorbar(
    stats["iteration"], stats["retr_mean"],
    yerr=stats["retr_ci"],
    fmt="o",
    label="Retrieved",
    linestyle="none",
    capsize=CAPSIZE,
    color=CB_GREEN 
)

plt.xlabel("Iteration")
plt.ylabel("Average Grade")
plt.xticks(stats["iteration"], rotation=0)

# Reverse grade scale: 1 at top, 6 at bottom
plt.ylim(3, 1.4)

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

label_map = {
    "base": "Base",
    "combined": "Combined",
    "retrieved": "Retrieved",
}

legend_width = 1.8  # adjust if needed

fig, ax = plt.subplots(
    figsize=(FIGSIZE[0] + legend_width, FIGSIZE[1])
)

winner_counts.plot(
    kind="bar",
    stacked=True,
    width=BAR_WIDTH,
    ax=ax,
    color=[color_map[c] for c in winner_counts.columns],
)

ax.set_xlabel("Iteration")
ax.set_ylabel("Count")
ax.set_ylim(0, 100)
ax.set_xticklabels(winner_counts.index, rotation=0)

ax.legend(
    labels=[label_map[c] for c in winner_counts.columns],
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
)

mark_rephrased_iters_categorical(winner_counts.index)

fig.tight_layout()
savefig("judge_winner_counts")

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

stacked_rate_bar(
    cos_stats,
    x_col="iteration",
    y_col="mean",
    yerr_col="ci",
    ylabel="Cosine Similarity",
    filename="cosine_similarity_per_iteration"
)

# ============================================================
# 6. Prompt match per Iteration
# ============================================================

pm_stats = []
for it, g in df.groupby("iteration"):
    both = g[["masked_prompt", "retrieved_prompt"]].notna().all(axis=1)

    if both.any():
        exact_count = int(g.loc[both, "prompt_same"].sum())
        mean_ratio = float(g.loc[both, "prompt_match_ratio"].mean())
        n_total = int(both.sum())
    else:
        exact_count = np.nan
        mean_ratio = np.nan
        n_total = 0

    pm_stats.append({
        "iteration": it,
        "exact_same_count": exact_count,
        "mean_ratio": mean_ratio,
        "n_total": n_total
    })

pm_stats = pd.DataFrame(pm_stats).sort_values("iteration")

# Exact same prompt count per iteration
plt.figure(figsize=FIGSIZE)
x = pm_stats["iteration"].to_numpy()

plt.bar(
    x,
    pm_stats["exact_same_count"],
    width=BAR_WIDTH,
    color=CB_GREEN
)

plt.xlabel("Iteration")
plt.ylabel("Exact Same Count")
plt.xticks(x, rotation=0)
plt.ylim(0, 100)
mark_rephrased_iters()
savefig("prompt_exact_match_count_per_iteration")


# Mean close-match ratio per iteration
plt.figure(figsize=FIGSIZE)
x = pm_stats["iteration"].to_numpy()

plt.bar(
    x,
    pm_stats["mean_ratio"],
    width=BAR_WIDTH,
    color=CB_GREEN
)

plt.xlabel("Iteration")
plt.ylabel("Mean Close-Match Ratio")
plt.xticks(x, rotation=0)
plt.ylim(0, 1)
mark_rephrased_iters()
savefig("prompt_close_match_ratio_per_iteration")

has_both = df[["masked_prompt", "retrieved_prompt"]].notna().all(axis=1)

rows = []
for it, g in df[has_both].groupby("iteration"):
    retrieval_winners = g["winner"].isin(["retrieved", "combined"])
    exact_retrieval_win = ((g["prompt_same"]) & retrieval_winners).sum()
    wrong_retrieval_win = ((~g["prompt_same"]) & retrieval_winners).sum()

    rows.append({
        "iteration": it,
        "count_exact_retrieved": exact_retrieval_win,
        "count_wrong_retrieved": wrong_retrieval_win,
        "n_total": len(g),
    })

wr = pd.DataFrame(rows).sort_values("iteration")

plt.figure(figsize=FIGSIZE)
x = wr["iteration"].to_numpy()

plt.bar(
    x,
    wr["count_wrong_retrieved"],
    width=BAR_WIDTH,
    color=CB_RED,
    label="Non-exact retrieval"
)

plt.bar(
    x,
    wr["count_exact_retrieved"],
    width=BAR_WIDTH,
    bottom=wr["count_wrong_retrieved"],
    color=CB_GREEN,
    label="Exact retrieval"
)

plt.xlabel("Iteration")
plt.ylabel("Count")
plt.xticks(x, rotation=0)
plt.ylim(0, 100)
plt.legend()
mark_rephrased_iters()
savefig("retrieved_win_counts_exact_vs_wrong")

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

def normalize_models(models):
    if not isinstance(models, list):
        return []
    return [m.strip() for m in models if isinstance(m, str) and m.strip()]


def model_hit(expected, predicted) -> float:
    expected = normalize_models(expected)
    predicted = normalize_models(predicted)

    # Case 1: no expected model and no predicted model
    if not expected and not predicted:
        return 1.0

    # Case 2: no expected model, but ChatGPT/controller handled it directly
    if not expected and any("chatgpt" in m.lower() or "gpt" in m.lower() for m in predicted):
        return 1.0

    # Case 3: no expected model, but other model was predicted
    if not expected:
        return 0.0

    # Normal relaxed hit criterion
    return 1.0 if set(expected).intersection(set(predicted)) else 0.0


def model_exact(expected, predicted) -> float:
    expected = normalize_models(expected)
    predicted = normalize_models(predicted)

    # Treat empty expected and empty predicted as exact match
    if not expected and not predicted:
        return 1.0

    # Optional: treat ChatGPT/controller-only cases as exact for empty expected
    if not expected and any("chatgpt" in m.lower() or "gpt" in m.lower() for m in predicted):
        return 1.0

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

# 8. Model Exact-set match rate per iteration
exact_stats = []
for it, g in df.groupby("iteration"):
    exact_count = (g["model_exact"] == 1.0).sum()
    exact_stats.append({"iteration": it, "count": exact_count})
exact_stats = pd.DataFrame(exact_stats).sort_values("iteration")

stacked_count_bar(
    exact_stats,
    x_col="iteration",
    y_col="count",
    ylabel="Exact Match Count",
    filename="model_exact_match_rate_per_iteration",
    ymax=100
)

# 9. Model invocation failure rate per iteration
fail_stats = []
for it, g in df.groupby("iteration"):
    m = g["inference_failure_rate"].mean()
    fail_stats.append({
        "iteration": it,
        "mean": m
    })

fail_stats = pd.DataFrame(fail_stats).sort_values("iteration")

stacked_rate_bar(
    fail_stats,
    x_col="iteration",
    y_col="mean",
    ylabel="Model Invocation Failure Rate",
    filename="model_invocation_failure_rate_per_iteration",
    ylim=(0, 1)
)

# 10. Stacked bar: Hits vs Misses per iteration
hm = (
    df[df["model_hit"].notna()]
    .assign(hit=lambda x: (x["model_hit"] == 1.0).astype(int),
            miss=lambda x: (x["model_hit"] == 0.0).astype(int))
    .groupby("iteration")[["hit", "miss"]]
    .sum()
    .sort_index()
)

plt.figure(figsize=FIGSIZE)
x = hm.index.to_numpy()

plt.bar(
    x,
    hm["hit"].to_numpy(),
    width=BAR_WIDTH,
    color=CB_GREEN,
    label="Hit"
)
plt.bar(
    x,
    hm["miss"].to_numpy(),
    width=BAR_WIDTH,
    color=CB_RED,
    bottom=hm["hit"],
    label="Miss"
)

plt.xlabel("Iteration")
plt.ylabel("Count")
plt.xticks(x, rotation=0)
plt.ylim(0, 100)
plt.legend()
mark_rephrased_iters()
savefig("model_hit_miss_counts")

# ============================================================
# Build Quality Analytics Help
# ============================================================

from collections import defaultdict

# Group records by prompt_id
by_prompt = defaultdict(list)

for r in records:
    by_prompt[r["prompt_id"]].append(r)

target = ["base", "retrieved", "retrieved", "retrieved", "retrieved"]

for pid, items in by_prompt.items():
    # sort by iteration
    items_sorted = sorted(items, key=lambda x: x["iteration"])

    winners = [it["judge"]["winner"] for it in items_sorted]

    # check ordered subsequence
    i = 0
    for w in winners:
        if w == target[i]:
            i += 1
            if i == len(target):
                print(f"prompt_id {pid}: {winners}")
                break

from collections import defaultdict

TARGET_PROMPTS = {25, 39, 52, 69}
OUTPUT_FILE = OUT_DIR + "/qualitative_analysis.txt"

by_prompt = defaultdict(list)

for r in records:
    if r["prompt_id"] in TARGET_PROMPTS:
        by_prompt[r["prompt_id"]].append(r)

def latex_escape(text):
    if text is None:
        return ""
    return (
        text.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
            .replace("#", r"\#")
            .replace("$", r"\$")
            .replace("{", r"\{")
            .replace("}", r"\}")
    )

latex_output = []

for pid in sorted(TARGET_PROMPTS):
    if pid not in by_prompt:
        continue

    items = sorted(by_prompt[pid], key=lambda x: x["iteration"])
    latex_output.append(f"\\subsubsection{{Prompt {pid}}}\\label{{prompt:{pid}}}\n")

    for it in items:
        winner = it["judge"]["winner"]
        grade_base = it["judge"]["grade_base"]
        grade_ret = it["judge"]["grade_retrieved"]

        base = latex_escape(it.get("base_explanation", ""))
        retrieved = latex_escape(it.get("retriever", {}).get("retrieved_explanation", ""))
        combined = latex_escape(it.get("judge", {}).get("improved_explanation", ""))
        reason = latex_escape(it.get("judge", {}).get("reason", ""))

        iteration = it["iteration"]

        latex_output.append(
            f"\\paragraph{{Iteration {iteration} --- Winner: {winner}}}\\mbox{{}}\\\\"
        )
        
        latex_output.append(
            f"""
\\textbf{{Reason}}: {reason}"""
        )


        # special rule: first iteration + winner is base -> only print base
        if iteration == 1 and winner == "base":
            latex_output.append(
                f"""
\\textbf{{Base (Grade {grade_base})}}: {base}"""
            )
            continue

        # always show base
        latex_output.append(
            f"""
\\textbf{{Base (Grade {grade_base})}}: {base}"""
        )

        # show retrieved unless first-iteration/base special case already continued
        latex_output.append(
            f"""
\\textbf{{Retrieved (Grade {grade_ret})}}: {retrieved}"""
        )

        # only show combined if winner is actually combined
        if winner == "combined":
            latex_output.append(
                f"""
\\textbf{{Combined}}: {combined}"""
            )

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_output))

print(f"Saved qualitative analysis to {OUTPUT_FILE}")

# ============================================================
# Human Evaluation Sheet Generator
# ============================================================

import os
import random
from collections import defaultdict

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

OUTPUT_FILE = os.path.join(OUT_DIR, "human_eval_sheet.tex")

# ------------------------------------------------------------
# Helper: group records by (prompt_id, iteration)
# ------------------------------------------------------------
by_pair = {}
for r in records:
    key = (r["prompt_id"], r["iteration"])
    by_pair[key] = r

# ------------------------------------------------------------
# Fixed examples requested by user
#
# compare_mode:
#   - "base_vs_retrieved"
#   - "base_vs_combined"
#   - "retrieved_vs_combined"
#
# For combined-winning examples, base_vs_combined is usually
# the most interesting human comparison.
# ------------------------------------------------------------
FIXED_SELECTIONS = [
    # Prompt 25 (base-dominant)
    {"prompt_id": 25, "iteration": 2, "compare_mode": "base_vs_retrieved"},
    {"prompt_id": 25, "iteration": 6, "compare_mode": "base_vs_retrieved"},

    # Prompt 39 (failure case)
    {"prompt_id": 39, "iteration": 1, "compare_mode": "base_vs_combined"},
    {"prompt_id": 39, "iteration": 3, "compare_mode": "base_vs_retrieved"},
    {"prompt_id": 39, "iteration": 7, "compare_mode": "base_vs_combined"},

    # Prompt 52 (simple)
    {"prompt_id": 52, "iteration": 2, "compare_mode": "base_vs_retrieved"},
    {"prompt_id": 52, "iteration": 5, "compare_mode": "base_vs_retrieved"},

    # Prompt 69 (complex)
    {"prompt_id": 69, "iteration": 2, "compare_mode": "base_vs_combined"},
    {"prompt_id": 69, "iteration": 3, "compare_mode": "base_vs_retrieved"},
    {"prompt_id": 69, "iteration": 7, "compare_mode": "base_vs_combined"},
]

# ------------------------------------------------------------
# Latex escaping
# ------------------------------------------------------------
def latex_escape(text):
    if text is None:
        return ""
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )

# ------------------------------------------------------------
# Pick the two explanations to compare
# ------------------------------------------------------------
def get_explanations(rec, compare_mode):
    base = rec.get("base_explanation", "")
    retrieved = rec.get("retriever", {}).get("retrieved_explanation", "")
    combined = rec.get("judge", {}).get("improved_explanation", "")

    if compare_mode == "base_vs_retrieved":
        return ("Base", base, "Retrieved", retrieved)
    elif compare_mode == "base_vs_combined":
        return ("Base", base, "Combined", combined)
    elif compare_mode == "retrieved_vs_combined":
        return ("Retrieved", retrieved, "Combined", combined)
    else:
        raise ValueError(f"Unknown compare_mode: {compare_mode}")

# ------------------------------------------------------------
# Check whether a record is usable for a given mode
# ------------------------------------------------------------
def has_required_explanations(rec, compare_mode):
    _, text_a, _, text_b = get_explanations(rec, compare_mode)
    return bool(text_a and text_b)

# ------------------------------------------------------------
# Build fixed pool
# ------------------------------------------------------------
selected_keys = set()
survey_items = []

for item in FIXED_SELECTIONS:
    key = (item["prompt_id"], item["iteration"])
    rec = by_pair.get(key)
    if rec is None:
        print(f"[WARN] Missing fixed item: prompt {key[0]}, iteration {key[1]}")
        continue

    if not has_required_explanations(rec, item["compare_mode"]):
        print(f"[WARN] Fixed item lacks required explanations: {key}, mode={item['compare_mode']}")
        continue

    survey_items.append({
        "prompt_id": item["prompt_id"],
        "iteration": item["iteration"],
        "compare_mode": item["compare_mode"],
        "record": rec,
        "fixed": True,
    })
    selected_keys.add(key)

# ------------------------------------------------------------
# Randomly sample 10 more pairs
#
# Strategy:
# - Exclude already selected fixed items
# - Prefer base_vs_retrieved normally
# - If winner is combined and combined text exists, sometimes
#   compare base_vs_combined to get more interesting cases
# ------------------------------------------------------------
candidate_items = []

for (pid, it), rec in by_pair.items():
    if (pid, it) in selected_keys:
        continue

    winner = rec.get("judge", {}).get("winner", "")
    possible_modes = []

    if has_required_explanations(rec, "base_vs_retrieved"):
        possible_modes.append("base_vs_retrieved")

    if winner == "combined" and has_required_explanations(rec, "base_vs_combined"):
        possible_modes.append("base_vs_combined")

    if winner == "combined" and has_required_explanations(rec, "retrieved_vs_combined"):
        possible_modes.append("retrieved_vs_combined")

    if not possible_modes:
        continue

    # prefer base_vs_combined for combined winners, otherwise base_vs_retrieved
    if winner == "combined" and "base_vs_combined" in possible_modes:
        mode = "base_vs_combined"
    else:
        mode = possible_modes[0]

    candidate_items.append({
        "prompt_id": pid,
        "iteration": it,
        "compare_mode": mode,
        "record": rec,
        "fixed": False,
    })

if len(candidate_items) < 10:
    print(f"[WARN] Only {len(candidate_items)} random candidates available, not 10.")

random.shuffle(candidate_items)
survey_items.extend(candidate_items[:10])

# ------------------------------------------------------------
# Final shuffle so fixed/random are mixed in the sheet
# ------------------------------------------------------------
random.shuffle(survey_items)

# ------------------------------------------------------------
# Generate LaTeX blocks
# ------------------------------------------------------------
def make_item_block(idx, item):
    rec = item["record"]
    pid = item["prompt_id"]
    iteration = item["iteration"]
    compare_mode = item["compare_mode"]

    left_name, left_text, right_name, right_text = get_explanations(rec, compare_mode)

    # Randomize which side is A/B
    if random.random() < 0.5:
        a_label, a_text = left_name, left_text
        b_label, b_text = right_name, right_text
    else:
        a_label, a_text = right_name, right_text
        b_label, b_text = left_name, left_text

    prompt_text = rec.get("prompt", "")
    winner = rec.get("judge", {}).get("winner", "")
    reason = rec.get("judge", {}).get("reason", "")

    prompt_text = latex_escape(prompt_text)
    a_text = latex_escape(a_text)
    b_text = latex_escape(b_text)
    reason = latex_escape(reason)
    winner = latex_escape(winner)

    # internal metadata comment for later evaluation
    metadata_comment = (
        f"% item={idx} | prompt_id={pid} | iteration={iteration} | "
        f"compare_mode={compare_mode} | judge_winner={winner} | "
        f"A_source={a_label} | B_source={b_label}"
    )

    return f"""
{metadata_comment}
\\subsubsection*{{Item {idx}}}
\\textbf{{Prompt ID:}} {pid} \\hfill \\textbf{{Iteration:}} {iteration}

\\textbf{{Task Prompt:}} {prompt_text}

\\vspace{{0.75em}}
\\noindent
\\begin{{minipage}}[t]{{0.48\\textwidth}}
\\textbf{{Explanation A}}

\\vspace{{0.5em}}
\\fbox{{
\\parbox[t][0.30\\textheight][t]{{0.95\\linewidth}}{{{a_text}}}
}}
\\end{{minipage}}
\\hfill
\\begin{{minipage}}[t]{{0.48\\textwidth}}
\\textbf{{Explanation B}}

\\vspace{{0.5em}}
\\fbox{{
\\parbox[t][0.30\\textheight][t]{{0.95\\linewidth}}{{{b_text}}}
}}
\\end{{minipage}}

\\vspace{{1em}}

\\textbf{{Evaluation}}
\\begin{{itemize}}
    \\item \\textbf{{Better explanation:}} \\hspace{{1em}} $\\square$ A \\hspace{{1em}} $\\square$ B \\hspace{{1em}} $\\square$ Equal
    \\item \\textbf{{Grade A:}} \\underline{{\\hspace{{2cm}}}} \\hspace{{1em}} \\textbf{{Grade B:}} \\underline{{\\hspace{{2cm}}}}
    \\item \\textbf{{Reason:}} \\\\[4.5em]
\\end{{itemize}}

\\vspace{{1em}}
\\hrule
\\vspace{{1em}}
"""

# ------------------------------------------------------------
# Build whole document
# ------------------------------------------------------------
latex_output = []

latex_output.append(r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{enumitem}
\usepackage{parskip}
\setlength{\parindent}{0pt}
\setlength{\fboxsep}{8pt}
\setlength{\fboxrule}{0.5pt}
\renewcommand{\arraystretch}{1.2}

\begin{document}
""")

latex_output.append(r"\section*{Human Evaluation Sheet}")
latex_output.append(r"% Write your own intro above if needed.")
latex_output.append("")

for idx, item in enumerate(survey_items, start=1):
    latex_output.append(make_item_block(idx, item))

latex_output.append(r"\end{document}")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_output))

print(f"[OK] Human evaluation sheet saved to {OUTPUT_FILE}")
print(f"[OK] Total items: {len(survey_items)}")

# ============================================================
# Human vs LLM Comparison: 3 Targeted Graphs
# ============================================================

# 1. Prepare Data
target_pairs = [
    (78, 3), (75, 6), (11, 2), (39, 7), (9, 1), (69, 7),
    (25, 6), (39, 3), (52, 2), (38, 3), (52, 3), (10, 5), (25, 2),
    (69, 2), (39, 1), (19, 7), (3, 3), (69, 3), (75, 4), (55, 5)
]

HUMAN_EVAL_JSON = os.path.join(RUN_DIR, "human_eval_results.json")

if os.path.exists(HUMAN_EVAL_JSON):
    with open(HUMAN_EVAL_JSON, "r", encoding="utf-8") as f:
        h_data = json.load(f)
    df_h = pd.DataFrame(h_data)
    if "iteration_number" in df_h.columns:
        df_h = df_h.rename(columns={"iteration_number": "iteration"})

    # Map winners to numbers for plotting (0: base, 1: retrieved, 2: combined)
    WIN_MAP = {"base": 0, "retrieved": 1, "combined": 2}
    
    # 1. Prepare Data
    comp_list = []
    for pid, it in target_pairs:
        l_row = df[(df["prompt_id"] == pid) & (df["iteration"] == it)]
        h_row = df_h[(df_h["prompt_id"] == pid) & (df_h["iteration"] == it)]

        if not l_row.empty and not h_row.empty:
            l, h = l_row.iloc[0], h_row.iloc[0]
            comp_list.append({
                "prompt_id": pid,      # Keep as int for sorting
                "iteration": it,      # Keep as int for sorting
                "llm_win": WIN_MAP.get(l["winner"]),
                "hum_win": WIN_MAP.get(h["winner"]),
                "llm_base": l["grade_base_num"],
                "hum_base": GRADE_MAP.get(h["grade A"]),
                "llm_retr": l["grade_retrieved_num"],
                "hum_retr": GRADE_MAP.get(h["grade B"])
            })

    df_c = pd.DataFrame(comp_list)

    # --- CRITICAL CHANGE: MULTI-COLUMN SORT ---
    # Sort by Iteration (Low to High), then Prompt ID (Low to High)
    df_c = df_c.sort_values(by=["iteration", "prompt_id"]).reset_index(drop=True)

    # Create the display label AFTER sorting
    df_c["id"] = df_c.apply(lambda x: f"{int(x['prompt_id'])}-{int(x['iteration'])}", axis=1)

    # --- Plot 1: Winner Comparison ---
    plt.figure(figsize=FIGSIZE)
    plt.scatter(df_c["id"], df_c["llm_win"], color=CB_BLUE, s=120, label="LLM", marker='o')
    plt.scatter(df_c["id"], df_c["hum_win"], color=CB_RED, s=120, label="Human", marker='x', linewidths=2)
    highlight_human_eval_rephrased(df_c)
    plt.yticks([0, 1, 2], ["Base", "Retrieved", "Combined"])
    plt.xticks(rotation=90, ha='center')
    plt.ylabel("Winner Category")
    plt.xlabel("Prompt ID - Iteration")
    plt.legend(frameon=True)
    savefig("comp_winners")

    # --- Plot 2: Base (A) Grade Comparison ---
    plt.figure(figsize=FIGSIZE)
    plt.scatter(df_c["id"], df_c["llm_base"], color=CB_BLUE, s=120, label="LLM", marker='o')
    plt.scatter(df_c["id"], df_c["hum_base"], color=CB_RED, s=120, label="Human", marker='x', linewidths=2)
    highlight_human_eval_rephrased(df_c)
    plt.ylim(4.5, 0.5) # Numbers 1-6, A is top
    plt.xticks(rotation=90, ha='center')
    plt.ylabel("Grade")
    plt.xlabel("Prompt ID - Iteration")
    plt.legend(frameon=True)
    savefig("comp_base_grades")

    # --- Plot 3: Retrieved (B) Grade Comparison ---
    plt.figure(figsize=FIGSIZE)
    plt.scatter(df_c["id"], df_c["llm_retr"], color=CB_BLUE, s=120, label="LLM", marker='o')
    plt.scatter(df_c["id"], df_c["hum_retr"], color=CB_RED, s=120, label="Human", marker='x', linewidths=2)
    highlight_human_eval_rephrased(df_c)
    plt.ylim(4.5, 0.5)
    plt.xticks(rotation=90, ha='center')
    plt.ylabel("Grade")
    plt.xlabel("Prompt ID - Iteration")
    plt.legend(frameon=True)
    savefig("comp_retrieved_grades")

    print(f"[OK] Generated 3 comparison plots for {len(df_c)} matching pairs.")
# ============================================================
# Done
# ============================================================

print(f"[OK] PDF figures saved to ./{OUT_DIR}/")