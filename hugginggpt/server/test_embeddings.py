import argparse
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

OUTDIR = Path("embedding_plots")
OUTDIR.mkdir(exist_ok=True)


# -------------------------
# your functions (copied)
# -------------------------
def mask_paths(s: str) -> str:
    if not isinstance(s, str):
        return s

    _PATH_EXTS = (
        "png","jpg","jpeg","gif","webp","bmp","tiff","svg",
        "mp3","wav","flac","m4a","ogg",
        "mp4","mov","avi","mkv","webm",
        "pdf","txt","json","csv","yaml","yml","xml","html",
        "zip","tar","gz","7z","doc","docx","ppt","pptx","xls","xlsx"
    )

    s = re.sub(r'https?://[^\s)>\]\'"]+', '[PATH]', s)
    s = re.sub(r'[A-Za-z]:\\[^\s\'"]+', '[PATH]', s)
    s = re.sub(r'(?<!\w)/[^\s\'"]+', '[PATH]', s)
    rel_ext_pattern = r'(?:(?:\./|\.\./)?(?:[\w\-\.]+/)+[\w\-.]+\.(?:' + "|".join(_PATH_EXTS) + '))'
    s = re.sub(rel_ext_pattern, '[PATH]', s, flags=re.IGNORECASE)
    s = re.sub(r'\"(?:[\\/][\w\-.]+)+(?:\.[\w]+)\"', '"[PATH]"', s)

    return s


def canonicalize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    t = mask_paths(t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[\"`']", "", t)
    t = re.sub(r"([!?.,:;])\1+", r"\1", t)
    return t


# -------------------------
# embed + normalize
# -------------------------
def embed_texts(model, texts, batch_size=32):
    embs = model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
    embs = embs.astype(np.float32, copy=False)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return embs


# -------------------------
# plotting helpers
# -------------------------
def plot_embedding_space(E, labels, method="pca", title="Embedding space (2D)"):
    if method == "pca":
        proj = PCA(n_components=2, random_state=0).fit_transform(E)
        subtitle = "PCA"
    elif method == "tsne":
        # TSNE is slower; good for <= ~1000 points
        proj = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=min(30, max(5, (len(E) - 1) // 3)),
            random_state=0
        ).fit_transform(E)
        subtitle = "t-SNE"
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    plt.figure(figsize=(7, 6))
    plt.scatter(proj[:, 0], proj[:, 1], s=18)
    plt.title(f"{title} – {subtitle}")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")

    n = len(E)
    step = max(1, n // 25)
    for i in range(0, n, step):
        plt.text(proj[i, 0], proj[i, 1], str(i), fontsize=8)

    fname = OUTDIR / f"embedding_space_{method}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"[saved] {fname}")


def plot_query_vs_all(sim, query_idx, threshold=0.95, title="Query vs all"):
    x = np.arange(len(sim))
    plt.figure(figsize=(10, 4))
    plt.plot(x, sim, marker="o", markersize=3, linewidth=1)
    plt.axhline(threshold, linestyle="--")
    plt.axvline(query_idx, linestyle=":")
    plt.title(f"{title} (query idx={query_idx})")
    plt.xlabel("index")
    plt.ylabel("cosine similarity")
    plt.ylim(-0.05, 1.05)

    fname = OUTDIR / f"query_{query_idx}_vs_all.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"[saved] {fname}")


def print_topk(sim, texts, k=10):
    order = np.argsort(-sim)
    for rank, j in enumerate(order[:k], start=1):
        print(f"{rank:2d}. idx={j:3d} sim={float(sim[j]):.4f}  text={texts[j][:130]}")


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="Path to explanations.jsonl")
    ap.add_argument("--query_idx", type=int, default=0, help="Index used as query for query-vs-all plot")
    ap.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"], help="2D projection method")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--max_items", type=int, default=300, help="Limit items to keep plots readable (0=all)")
    args = ap.parse_args()

    # load texts
    entries = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if args.max_items and len(entries) > args.max_items:
        entries = entries[:args.max_items]

    texts = []
    labels = []
    for e in entries:
        td = e.get("task_description", "")
        ho = e.get("hugginggpt_output", "")
        texts.append(mask_paths(f"{td}\n{ho}"))
        labels.append(e.get("task_id", ""))

    n = len(texts)
    if n == 0:
        raise SystemExit("No entries loaded.")

    if not (0 <= args.query_idx < n):
        raise SystemExit(f"query_idx must be in [0, {n-1}]")

    # model
    model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v2.0", trust_remote_code=True)

    # RAW embeddings
    E_raw = embed_texts(model, texts)
    S_raw = cosine_similarity(E_raw, E_raw)  # includes i==j on diagonal

    # CAN embeddings
    texts_can = [canonicalize(t) for t in texts]
    E_can = embed_texts(model, texts_can)
    S_can = cosine_similarity(E_can, E_can)  # includes i==j on diagonal

    # -------------------------
    # 1) show stats including diagonal
    # -------------------------
    raw_all = S_raw.flatten()
    can_all = S_can.flatten()

    print("\n=== Similarity stats (INCLUDING i==j diagonal) ===")
    print(f"RAW: mean={raw_all.mean():.4f}  min={raw_all.min():.4f}  max={raw_all.max():.4f}")
    print(f"CAN: mean={can_all.mean():.4f}  min={can_all.min():.4f}  max={can_all.max():.4f}")

    # off-diagonal stats for reference
    mask_off = ~np.eye(n, dtype=bool)
    raw_off = S_raw[mask_off]
    can_off = S_can[mask_off]

    print("\n=== Similarity stats (OFF-diagonal only) ===")
    print(f"RAW: mean={raw_off.mean():.4f}  min={raw_off.min():.4f}  max={raw_off.max():.4f}")
    print(f"CAN: mean={can_off.mean():.4f}  min={can_off.min():.4f}  max={can_off.max():.4f}")

    # -------------------------
    # 2) embedding space scatter
    # -------------------------
    plot_embedding_space(E_raw, labels, method=args.method, title="Embedding space (RAW)")
    plot_embedding_space(E_can, labels, method=args.method, title="Embedding space (CANON)")

    # -------------------------
    # 3) query vs all (line graph) for RAW + CAN
    # -------------------------
    sim_raw = S_raw[args.query_idx, :]  # includes self = 1
    sim_can = S_can[args.query_idx, :]

    print(f"\n=== Top neighbors for query idx={args.query_idx} (RAW) ===")
    print_topk(sim_raw, texts, k=10)

    print(f"\n=== Top neighbors for query idx={args.query_idx} (CANON) ===")
    print_topk(sim_can, texts, k=10)

    plot_query_vs_all(sim_raw, args.query_idx, threshold=args.threshold,
                      title=f"RAW cosine(query,{args.query_idx}) vs all")
    plot_query_vs_all(sim_can, args.query_idx, threshold=args.threshold,
                      title=f"CANON cosine(query,{args.query_idx}) vs all")


if __name__ == "__main__":
    main()
