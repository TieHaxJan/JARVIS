# ============================================================
# Headless, NeurIPS-style visualization script
# ============================================================

import os
import json
import glob
import math
import numpy as np
import pandas as pd

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

LOG_PATTERN = "runs/20260115_164102_7e59/iterations/iteration_*.jsonl"
OUT_DIR = "runs/20260115_164102_7e59/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Prompt subgroup for cosine stability plot
STABLE_PROMPT_IDS = [0, 1, 2, 3, 4, 5, 6, 20, 25, 35, 56, 72, 74, 89, 90]

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
# Done
# ============================================================

print(f"[OK] Figures saved to ./{OUT_DIR}/ (PNG + PDF)")
