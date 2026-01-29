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
parser.add_argument("--run", default="runs/20260115_164102_7e59",
                    help="Run directory, e.g. runs/20260115_164102_7e59")
args = parser.parse_args()

RUN_DIR = args.run.rstrip("/")

LOG_PATTERN = os.path.join(RUN_DIR, "iterations", "iteration_*.jsonl")
OUT_DIR = os.path.join(RUN_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# Prompt subgroup for cosine stability plot
STABLE_PROMPT_IDS = [0, 1, 2, 3, 4, 5, 6, 20, 25]

# Grade mapping
GRADE_MAP = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}

# ============================================================
# NeurIPS-style plotting defaults
# ============================================================

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
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
    #plt.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
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
    yerr=stats["base_ci"], marker="o", label="Base"
)
plt.errorbar(
    stats["iteration"], stats["retr_mean"],
    yerr=stats["retr_ci"], marker="o", label="Retrieved"
)
plt.xlabel("Iteration")
plt.ylabel("Average Grade")
plt.title("Average Explanation Quality per Iteration")
plt.legend()
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

winner_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(6, 4),
)
plt.xlabel("Iteration")
plt.ylabel("Count")
plt.title("Judge Decisions per Iteration")
plt.legend(title="Winner")
savefig("judge_winner_counts")

# ============================================================
# 3. RAG Reliance per Iteration (winner == retrieved)
# ============================================================

rag_stats = []

for it, g in df.groupby("iteration"):
    vals = (g["winner"] == "retrieved").astype(int)
    mean, ci = mean_ci(vals)
    rag_stats.append({
        "iteration": it,
        "mean": mean,
        "ci": ci
    })

rag_stats = pd.DataFrame(rag_stats).sort_values("iteration")

plt.figure()
plt.errorbar(
    rag_stats["iteration"],
    rag_stats["mean"],
    yerr=rag_stats["ci"],
    marker="o"
)
plt.xlabel("Iteration")
plt.ylabel("Fraction Retrieved Wins")
plt.ylim(0, 1)
plt.title("RAG Reliance per Iteration")
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
    marker="o"
)
plt.xlabel("Iteration")
plt.ylabel("Cosine Similarity")
plt.title("Retriever Cosine Similarity per Iteration")
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
    legend="full"
)
plt.xlabel("Iteration")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity Stability (Task Subgroup)")
plt.legend(title="Prompt ID")
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
plt.plot(pm_stats["iteration"], pm_stats["exact_same_frac"], marker="o")
plt.xlabel("Iteration")
plt.ylabel("Exact Same Fraction")
plt.ylim(0, 1)
plt.title("Exact Match: masked_prompt vs retrieved_prompt")
savefig("prompt_exact_match_per_iteration")

# (b) close-match ratio (with CI)
plt.figure()
plt.errorbar(
    pm_stats["iteration"], pm_stats["mean_ratio"],
    yerr=pm_stats["ci_ratio"], marker="o"
)
plt.xlabel("Iteration")
plt.ylabel("Close-Match Ratio")
plt.ylim(0, 1)
plt.title("Close Match Ratio: masked_prompt vs retrieved_prompt")
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
    label="P(Judge picks Retrieved | exact retrieval)"
)
plt.plot(
    wr["iteration"], wr["win_rate_wrong"],
    marker="o",
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
savefig("heatmap_rag_regimes_per_iteration")


# ============================================================
# Done
# ============================================================

print(f"[OK] Figures saved to ./{OUT_DIR}/ (PNG + PDF)")
