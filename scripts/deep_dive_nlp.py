"""Build dependency-light semantic clustering and comment sentiment tables."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiktok_semantic.io import read_config, write_table  # noqa: E402, I001


POSITIVE_WORDS = {
    "action",
    "amazing",
    "beautiful",
    "better",
    "care",
    "cool",
    "encourage",
    "good",
    "great",
    "help",
    "hope",
    "love",
    "progress",
    "save",
    "solution",
    "thank",
    "true",
    "yes",
}

NEGATIVE_WORDS = {
    "angry",
    "bad",
    "crisis",
    "die",
    "disaster",
    "fear",
    "hard",
    "hate",
    "hoax",
    "kill",
    "late",
    "pollution",
    "problem",
    "sad",
    "scam",
    "terrible",
    "threat",
    "wrong",
}

EMOTION_PATTERNS = {
    "hope": r"\b(hope|solution|can do|help|save|progress|better|work)\b",
    "anxiety": r"\b(scared|fear|worried|anxious|too late|hard|crisis|disaster|die|kill)\b",
    "confusion": r"\b(how|what|why|where|don't know|dont know|can i|would|should)\b|\?",
    "anger": r"\b(angry|hate|blame|hypocrite|stupid|wrong|greedy|scam|hoax)\b",
    "support": r"\b(love|thank|yes|true|beautiful|amazing|agree|same)\b",
}


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def choose_text_column(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "semantic_text",
        "text_for_nlp",
        "overall_narrative_message",
        "summary_text",
        "title",
    ]
    present = [col for col in candidates if col in df.columns]
    if not present:
        return pd.Series([""] * len(df), index=df.index)
    return df[present].fillna("").astype(str).agg(" ".join, axis=1).map(clean_text)


def auto_cluster_count(n_docs: int) -> int:
    if n_docs < 4:
        return max(1, n_docs)
    return min(8, max(2, round(math.sqrt(n_docs))))


def semantic_clusters(
    summaries: pd.DataFrame,
    post_metrics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if summaries.empty:
        return {
            "semantic_clusters": pd.DataFrame(),
            "semantic_cluster_performance": pd.DataFrame(),
        }

    df = summaries.copy()
    df["cluster_text"] = choose_text_column(df)
    df = df[df["cluster_text"].str.len() > 20].copy()
    if len(df) < 2:
        df["cluster_id"] = 0
        df["cluster_terms"] = ""
        df["cluster_name"] = "single cluster"
    else:
        vectorizer = TfidfVectorizer(
            min_df=1,
            max_df=0.85,
            ngram_range=(1, 2),
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(df["cluster_text"])
        n_clusters = auto_cluster_count(len(df))
        best_labels = None
        best_score = -1.0
        for k in range(2, min(n_clusters + 2, len(df))):
            labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(matrix)
            score = silhouette_score(matrix, labels) if len(set(labels)) > 1 else -1.0
            if score > best_score:
                best_score = score
                best_labels = labels
        df["cluster_id"] = best_labels if best_labels is not None else 0

        terms = np.array(vectorizer.get_feature_names_out())
        cluster_terms = {}
        for cid in sorted(df["cluster_id"].unique()):
            idx = np.where(df["cluster_id"].to_numpy() == cid)[0]
            weights = np.asarray(matrix[idx].mean(axis=0)).ravel()
            top = terms[weights.argsort()[::-1][:8]]
            cluster_terms[cid] = ", ".join(top)
        df["cluster_terms"] = df["cluster_id"].map(cluster_terms)
        df["cluster_name"] = df["cluster_terms"].str.split(",").str[:3].str.join(" / ")

        centroids = {
            cid: np.asarray(matrix[df["cluster_id"].to_numpy() == cid].mean(axis=0))
            for cid in sorted(df["cluster_id"].unique())
        }
        reps = []
        for cid, centroid in centroids.items():
            idx = np.where(df["cluster_id"].to_numpy() == cid)[0]
            sims = cosine_similarity(matrix[idx], centroid).ravel()
            reps.append((cid, df.iloc[idx[int(sims.argmax())]]["video_id"]))
        rep_map = dict(reps)
        df["is_cluster_representative"] = df.apply(
            lambda r: r["video_id"] == rep_map.get(r["cluster_id"]),
            axis=1,
        )

    keep = [
        "video_id",
        "content_type",
        "cluster_id",
        "cluster_name",
        "cluster_terms",
        "is_cluster_representative",
        "overall_narrative_message",
        "overall_mood_tone",
    ]
    keep = [col for col in keep if col in df.columns]
    clusters = df[keep].copy()

    perf = clusters.merge(post_metrics, on="video_id", how="left")
    cluster_perf = (
        perf.groupby(["cluster_id", "cluster_name"], dropna=False)
        .agg(
            posts=("video_id", "nunique"),
            median_views=("play_count", "median"),
            total_views=("play_count", "sum"),
            median_eng_per_1k=("eng_per_1k_views", "median"),
            total_shares=("share_count", "sum"),
            total_comments=("comment_count", "sum"),
        )
        .reset_index()
        .sort_values(["median_views", "median_eng_per_1k"], ascending=False)
    )
    return {
        "semantic_clusters": clusters,
        "semantic_cluster_performance": cluster_perf,
    }


def sentiment_score(text: str) -> tuple[str, float]:
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return "neutral", 0.0
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    score = (pos - neg) / math.sqrt(len(tokens))
    if score > 0.15:
        return "positive", score
    if score < -0.15:
        return "negative", score
    return "neutral", score


def emotion_label(text: str) -> str:
    value = text.lower()
    for label, pattern in EMOTION_PATTERNS.items():
        if re.search(pattern, value):
            return label
    return "neutral"


def comment_sentiment_tables(
    comments: pd.DataFrame,
    post_metrics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if comments.empty or "text" not in comments.columns:
        return {
            "comment_sentiment_emotion": pd.DataFrame(),
            "comment_sentiment_summary": pd.DataFrame(),
            "post_comment_sentiment": pd.DataFrame(),
        }

    df = comments.copy()
    df["comment_text_clean"] = df["text"].map(clean_text)
    scored = df["comment_text_clean"].map(sentiment_score)
    df["sentiment_label"] = [label for label, _ in scored]
    df["sentiment_score"] = [score for _, score in scored]
    df["emotion_label"] = df["comment_text_clean"].map(emotion_label)

    summary = (
        df.groupby(["sentiment_label", "emotion_label"], dropna=False)
        .agg(
            comments=("comment_id", "count"),
            avg_comment_likes=("digg_count", "mean"),
            median_mins_since_post=("mins_since_post", "median"),
        )
        .reset_index()
        .sort_values("comments", ascending=False)
    )

    post_summary = (
        df.groupby("video_id", dropna=False)
        .agg(
            comments_scored=("comment_id", "count"),
            avg_sentiment_score=("sentiment_score", "mean"),
            pct_positive=("sentiment_label", lambda s: (s == "positive").mean()),
            pct_negative=("sentiment_label", lambda s: (s == "negative").mean()),
            top_emotion=("emotion_label", lambda s: s.value_counts().index[0] if len(s) else None),
        )
        .reset_index()
    )
    keep = [
        "video_id",
        "title",
        "play_count",
        "comment_count",
        "share_count",
        "eng_per_1k_views",
    ]
    keep = [col for col in keep if col in post_metrics.columns]
    if keep:
        post_summary = post_summary.merge(post_metrics[keep], on="video_id", how="left")
    post_summary = post_summary.sort_values(
        ["comments_scored", "avg_sentiment_score"],
        ascending=False,
    )
    keep_comments = [
        "video_id",
        "comment_id",
        "text",
        "digg_count",
        "mins_since_post",
        "sentiment_label",
        "sentiment_score",
        "emotion_label",
    ]
    keep_comments = [col for col in keep_comments if col in df.columns]
    return {
        "comment_sentiment_emotion": df[keep_comments],
        "comment_sentiment_summary": summary,
        "post_comment_sentiment": post_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build semantic clustering and comment sentiment tables."
    )
    parser.add_argument("--config", default="configs/sample.yaml")
    args = parser.parse_args()

    cfg = read_config(args.config)
    processed = Path(cfg["processed_dir"])
    core = processed / "core"
    analytics = processed / "analytics"
    post_metrics = pd.read_parquet(core / "post_metrics.parquet")
    summaries_path = core / "posts_summaries.parquet"
    comments_path = core / "posts_comments.parquet"
    summaries = pd.read_parquet(summaries_path) if summaries_path.exists() else pd.DataFrame()
    comments = pd.read_parquet(comments_path) if comments_path.exists() else pd.DataFrame()

    tables = {}
    tables.update(semantic_clusters(summaries, post_metrics))
    tables.update(comment_sentiment_tables(comments, post_metrics))
    for name, df in tables.items():
        write_table(df, analytics / f"{name}.csv")

    print(f"Built {len(tables)} deep-dive NLP tables in {analytics}")


if __name__ == "__main__":
    main()
