"""P4 文本表示对比：哈希 TF-IDF vs 预训练句向量（BGE-small）。

在同一个分层分组抽样上，用相同的数值/类别预处理、相同的 5 折和相同的
线性 SVM，比较两种文本表示。抽样规模默认 12 万，控制 CPU 推理时间；
结论用于判断“换预训练模型是否值得”，不外推到全量。

用法：
    HF_ENDPOINT=https://hf-mirror.com python src/classify_embedding.py
    python src/classify_embedding.py --limit 60000 --force
    python src/classify_embedding.py --limit 60000 --folds 10 \
        --backend sentence-transformers --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .classify import CAT_COLS, MAX_TEXT_CHARS, MODEL_FEATURES, N_HASH, model_and_score
    from .config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES
except ImportError:
    from classify import CAT_COLS, MAX_TEXT_CHARS, MODEL_FEATURES, N_HASH, model_and_score
    from config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES

N_FOLDS = 5
N_BOOTSTRAP = 5_000
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
OUT_METRICS = TABLES / "classify_embedding_metrics.csv"
OUT_JSON = TABLES / "classify_embedding.json"
OUT_FIG = FIGURES / "fig38_embedding_compare.png"


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def load_frame(limit: int | None) -> pd.DataFrame:
    columns = ["asin", "label", "in_scope", "reviewText", "summary"] + MODEL_FEATURES
    frame = pq.read_table(FEATURES_PARQUET, columns=columns).to_pandas()
    frame = frame.loc[frame["in_scope"]].reset_index(drop=True)
    if limit is not None and limit < len(frame):
        rng = np.random.default_rng(SEED)
        index = rng.choice(len(frame), size=limit, replace=False)
        frame = frame.iloc[np.sort(index)].reset_index(drop=True)
    assert len(frame) >= 10_000, "样本过少"
    return frame


def make_folds(frame: pd.DataFrame) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.full(len(frame), -1, dtype=np.int8)
    for index, (_, valid) in enumerate(
        splitter.split(frame, frame["label"], groups=frame["asin"])
    ):
        folds[valid] = index
    assert (folds >= 0).all()
    return folds


def embed_texts(
    texts: list[str],
    batch_size: int = 256,
    backend: str = "fastembed",
    device: str = "cpu",
) -> np.ndarray:
    if backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBED_MODEL, device=device)
        matrix = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        assert matrix.shape == (len(texts), EMBED_DIM)
        return matrix

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=EMBED_MODEL)
    parts = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors = np.vstack(list(model.embed(batch, batch_size=batch_size)))
        parts.append(vectors.astype(np.float32))
        if start % (batch_size * 20) == 0:
            print(f"  嵌入 {start + len(batch):,}/{len(texts):,}")
    matrix = np.vstack(parts)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-12)
    assert matrix.shape == (len(texts), EMBED_DIM)
    return matrix


def evaluate(
    frame: pd.DataFrame,
    folds: np.ndarray,
    text_matrix,
    name: str,
    use_sparse: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame[CAT_COLS].fillna("(missing)").astype(str)
    numbers = frame[MODEL_FEATURES].drop(columns=CAT_COLS).astype("float32")
    rows = []
    scores = np.full(len(frame), np.nan)
    preds = np.full(len(frame), -1, dtype=np.int8)

    for fold in range(N_FOLDS):
        train = folds != fold
        valid = folds == fold
        train_index, valid_index = np.flatnonzero(train), np.flatnonzero(valid)
        y_train, y_valid = labels[train], labels[valid]

        imputer = SimpleImputer(strategy="median", add_indicator=True)
        number_train = imputer.fit_transform(numbers.iloc[train_index])
        number_valid = imputer.transform(numbers.iloc[valid_index])
        scaler = StandardScaler()
        number_train = scaler.fit_transform(number_train).astype(np.float32)
        number_valid = scaler.transform(number_valid).astype(np.float32)

        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=50,
            sparse_output=True,
            dtype=np.float32,
        )
        category_train = encoder.fit_transform(categories.iloc[train_index])
        category_valid = encoder.transform(categories.iloc[valid_index])

        if use_sparse:
            tfidf = TfidfTransformer(sublinear_tf=True)
            text_train = tfidf.fit_transform(text_matrix[train_index])
            text_valid = tfidf.transform(text_matrix[valid_index])
        else:
            text_train = sp.csr_matrix(text_matrix[train_index])
            text_valid = sp.csr_matrix(text_matrix[valid_index])

        x_train = sp.hstack(
            [sp.csr_matrix(number_train), category_train, text_train], format="csr"
        )
        x_valid = sp.hstack(
            [sp.csr_matrix(number_valid), category_valid, text_valid], format="csr"
        )
        _, score, pred = model_and_score("linear_svm", x_train, y_train, x_valid)
        scores[valid_index] = score
        preds[valid_index] = pred
        rows.append(
            {
                "representation": name,
                "fold": fold + 1,
                "n_train": len(y_train),
                "n_valid": len(y_valid),
                "accuracy": float(accuracy_score(y_valid, pred)),
                "macro_f1": float(f1_score(y_valid, pred, average="macro")),
                "pr_auc": float(average_precision_score(y_valid, score)),
                "roc_auc": float(roc_auc_score(y_valid, score)),
            }
        )
        print(f"  {name} 第 {fold + 1}/{N_FOLDS} 折完成")
        del x_train, x_valid, text_train, text_valid
        gc.collect()
    return pd.DataFrame(rows), scores, preds


def paired_stats(diff: np.ndarray) -> dict:
    from scipy.stats import wilcoxon

    diff = np.asarray(diff, dtype=float)
    _, p_value = wilcoxon(diff)
    rng = np.random.default_rng(SEED)
    means = np.array([rng.choice(diff, len(diff)).mean() for _ in range(N_BOOTSTRAP)])
    return {
        "n_pairs": int(len(diff)),
        "mean_difference": float(diff.mean()),
        "cohen_dz": float(diff.mean() / diff.std(ddof=1)),
        "p_value": float(p_value),
        "bootstrap_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def plot_compare(table: pd.DataFrame, path: Path) -> None:
    metrics = ["macro_f1", "pr_auc", "roc_auc"]
    labels = {"macro_f1": "宏 F1", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for axis, metric in zip(axes, metrics):
        means = table.groupby("representation")[metric].mean()
        stds = table.groupby("representation")[metric].std(ddof=1)
        axis.bar(means.index, means.values, yerr=stds.values, capsize=4, color=["#4c72b0", "#dd8452"])
        axis.set_title(labels[metric])
        axis.set_ylim(0.5, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=15)
    fig.suptitle(f"文本表示对比（同一样本、同一 {N_FOLDS} 折、线性 SVM）")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF 与预训练句向量对比")
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument(
        "--full",
        action="store_true",
        help="使用全部 in_scope 评论，忽略 --limit",
    )
    parser.add_argument(
        "--text-chars", type=int, default=600,
        help="截断字符数；CPU 句向量推理的 token 数近似随之线性变化",
    )
    parser.add_argument("--folds", type=int, default=5, choices=[5, 10])
    parser.add_argument(
        "--backend",
        default="fastembed",
        choices=["fastembed", "sentence-transformers"],
        help="fastembed=ONNX CPU；sentence-transformers=PyTorch，可用 --device cuda",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="仅 sentence-transformers 后端生效",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    global N_FOLDS
    N_FOLDS = args.folds
    if args.backend == "sentence-transformers" and args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("--device cuda 不可用：torch.cuda.is_available() 为 False")
    setup_fonts()
    started = time.perf_counter()
    limit = None if args.full else args.limit
    frame = load_frame(limit)
    folds = make_folds(frame)
    scope = "全量" if args.full else "抽样"
    print(f"{scope} {len(frame):,}，正类 {frame['label'].mean():.4f}，{N_FOLDS} 折")

    texts = (
        frame["reviewText"].fillna("") + " " + frame["summary"].fillna("")
    ).str.slice(0, args.text_chars).tolist()
    # 文本已经抽成 list，释放宽表里的两列大字符串，降低全量运行的内存峰值。
    frame = frame.drop(columns=["reviewText", "summary"])

    print("构建哈希 TF-IDF 计数")
    vectorizer = HashingVectorizer(
        n_features=N_HASH,
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    counts = vectorizer.transform(texts)
    table_tfidf, _, _ = evaluate(frame, folds, counts, "hash_tfidf", use_sparse=True)
    del counts
    gc.collect()

    print(f"生成 BGE 句向量（{EMBED_MODEL}）")
    embed_started = time.perf_counter()
    embeddings = embed_texts(
        texts,
        batch_size=args.batch_size,
        backend=args.backend,
        device=args.device,
    )
    embed_seconds = time.perf_counter() - embed_started
    del texts
    gc.collect()
    table_bge, _, _ = evaluate(frame, folds, embeddings, "bge_small", use_sparse=False)

    table = pd.concat([table_tfidf, table_bge], ignore_index=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    plot_compare(table, OUT_FIG)

    summary = {}
    for name in ["hash_tfidf", "bge_small"]:
        part = table.loc[table["representation"] == name]
        summary[name] = {
            metric: {
                "mean": round(float(part[metric].mean()), 4),
                "std": round(float(part[metric].std(ddof=1)), 4),
            }
            for metric in ["accuracy", "macro_f1", "pr_auc", "roc_auc"]
        }
    test = paired_stats(
        table_bge["macro_f1"].to_numpy() - table_tfidf["macro_f1"].to_numpy()
    )
    test["comparison"] = "bge_small - hash_tfidf"

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "embedding_model": EMBED_MODEL,
        "embedding_dim": EMBED_DIM,
        "embedding_backend": args.backend,
        "embedding_device": args.device,
        "sample_size": int(len(frame)),
        "full_data": bool(args.full),
        "positive_rate": round(float(frame["label"].mean()), 4),
        "n_folds": N_FOLDS,
        "models": summary,
        "paired_test": test,
        "embedding_seconds": round(embed_seconds, 1),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "gpu": _gpu_name(),
        },
        "outputs": {"metrics": str(OUT_METRICS), "figure": str(OUT_FIG)},
        "text_chars": args.text_chars,
        "notes": [
            (
                f"全量 {len(frame):,}、文本截断 {args.text_chars} 字符；"
                if args.full
                else f"抽样 {len(frame):,}、文本截断 {args.text_chars} 字符，结论不外推到全量；"
            ),
            "两种表示使用同一批折、同一数值/类别预处理与同一线性 SVM；",
            f"BGE 后端 {args.backend}/{args.device}，向量已 L2 归一化，未做微调。",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
