from __future__ import annotations

import numpy as np
import pandas as pd
import re


COMMENT_INTENT_PATTERNS = {
    "action_oriented": r"\b(i can|i will|i'll|start|try|plant|garden|bike|bus|recycle|eco|meat|vegan|vegetarian|do this|make it work|help out)\b",
    "barrier_or_constraint": r"\b(hard|can't|cannot|don't know|parents|school|expensive|afford|where|how|but|problem)\b",
    "skeptic_or_counter": r"\b(fake|not real|scam|hoax|china|india|doesn't matter|too late|hypocrite)\b",
    "info_question": r"\?",
    "support_affect": r"\b(love|beautiful|amazing|true|yes|thank|save|needs us|sad|cry|wow)\b",
}


def top_posts(post_metrics: pd.DataFrame, min_views: int = 1000, n: int = 20) -> dict[str, pd.DataFrame]:
    df = post_metrics.copy()
    by_reach = df.sort_values("play_count", ascending=False).head(n)
    eligible = df[df["play_count"].fillna(0) >= min_views].copy()
    by_efficiency = eligible.sort_values("eng_per_1k_views", ascending=False).head(n)
    by_share = df.sort_values("share_count", ascending=False).head(n)
    return {
        "top_posts_by_reach": by_reach,
        "top_posts_by_engagement_efficiency": by_efficiency,
        "top_posts_by_shares": by_share,
    }


def hashtag_performance(post_metrics: pd.DataFrame, hashtags: pd.DataFrame) -> pd.DataFrame:
    if hashtags.empty:
        return pd.DataFrame()
    df = hashtags.merge(
        post_metrics[
            [
                "video_id",
                "play_count",
                "digg_count",
                "comment_count",
                "share_count",
                "collect_count",
                "eng_total",
                "eng_per_1k_views",
            ]
        ],
        on="video_id",
        how="left",
    )
    return (
        df.groupby("hashtag")
        .agg(
            posts=("video_id", "nunique"),
            median_views=("play_count", "median"),
            mean_views=("play_count", "mean"),
            median_eng_per_1k=("eng_per_1k_views", "median"),
            total_shares=("share_count", "sum"),
            total_comments=("comment_count", "sum"),
        )
        .sort_values(["posts", "median_eng_per_1k"], ascending=[False, False])
        .reset_index()
    )


def region_performance(post_metrics: pd.DataFrame) -> pd.DataFrame:
    if post_metrics.empty or "region" not in post_metrics.columns:
        return pd.DataFrame()
    return (
        post_metrics.groupby("region", dropna=False)
        .agg(
            posts=("video_id", "nunique"),
            total_views=("play_count", "sum"),
            median_views=("play_count", "median"),
            total_shares=("share_count", "sum"),
            total_comments=("comment_count", "sum"),
            median_eng_per_1k=("eng_per_1k_views", "median"),
        )
        .reset_index()
        .sort_values("total_views", ascending=False)
    )
def content_type_performance(post_metrics: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    if summaries.empty:
        return pd.DataFrame()
    df = post_metrics.merge(summaries[["video_id", "content_type"]], on="video_id", how="left")
    return (
        df.groupby("content_type", dropna=False)
        .agg(
            posts=("video_id", "nunique"),
            median_views=("play_count", "median"),
            median_eng_per_1k=("eng_per_1k_views", "median"),
            mean_shares=("share_count", "mean"),
            mean_comments=("comment_count", "mean"),
        )
        .reset_index()
        .sort_values("posts", ascending=False)
    )


def theme_performance(posts_enriched: pd.DataFrame) -> pd.DataFrame:
    required = {"video_id", "primary_theme"}
    if posts_enriched.empty or not required.issubset(posts_enriched.columns):
        return pd.DataFrame()
    return (
        posts_enriched.groupby(["primary_theme", "action_frame", "mood_label"], dropna=False)
        .agg(
            posts=("video_id", "nunique"),
            median_views=("play_count", "median"),
            median_eng_per_1k=("eng_per_1k_views", "median"),
            mean_shares=("share_count", "mean"),
            mean_comments=("comment_count", "mean"),
            total_shares=("share_count", "sum"),
        )
        .reset_index()
        .sort_values(["median_eng_per_1k", "median_views"], ascending=False)
    )


def messaging_recommendations(posts_enriched: pd.DataFrame, min_views: int = 1000) -> pd.DataFrame:
    if posts_enriched.empty or "primary_theme" not in posts_enriched.columns:
        return pd.DataFrame()

    df = posts_enriched[posts_enriched["play_count"].fillna(0) >= min_views].copy()
    if df.empty:
        df = posts_enriched.copy()

    keys = ["primary_theme", "action_frame", "content_type"]
    grouped = (
        df.groupby(keys, dropna=False)
        .agg(
            posts=("video_id", "nunique"),
            median_views=("play_count", "median"),
            median_eng_per_1k=("eng_per_1k_views", "median"),
            median_shares=("share_count", "median"),
        )
        .reset_index()
    )
    if grouped.empty:
        return grouped

    grouped["views_rank_pct"] = grouped["median_views"].rank(pct=True)
    grouped["eng_rank_pct"] = grouped["median_eng_per_1k"].rank(pct=True)
    grouped["share_rank_pct"] = grouped["median_shares"].rank(pct=True)
    grouped["opportunity_score"] = np.round(
        0.40 * grouped["eng_rank_pct"]
        + 0.35 * grouped["views_rank_pct"]
        + 0.25 * grouped["share_rank_pct"],
        3,
    )
    return grouped.sort_values("opportunity_score", ascending=False)


def classify_comment_intent(text: object) -> str:
    value = str(text).lower() if isinstance(text, str) else ""
    for label, pattern in COMMENT_INTENT_PATTERNS.items():
        if re.search(pattern, value):
            return label
    return "general_reaction"


def comment_intent_tables(comments: pd.DataFrame, post_metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if comments.empty or "text" not in comments.columns:
        return {
            "comment_intent_summary": pd.DataFrame(),
            "post_comment_intents": pd.DataFrame(),
        }

    scored = comments.copy()
    scored["rough_intent"] = scored["text"].apply(classify_comment_intent)
    if "digg_count" in scored.columns:
        scored["digg_count"] = pd.to_numeric(scored["digg_count"], errors="coerce").fillna(0)

    intent_summary = (
        scored.groupby("rough_intent", dropna=False)
        .agg(
            comments=("comment_id", "count"),
            avg_comment_likes=("digg_count", "mean"),
            median_mins_since_post=("mins_since_post", "median"),
        )
        .reset_index()
        .sort_values("comments", ascending=False)
    )

    post_summary = (
        scored.groupby("video_id", dropna=False)
        .agg(
            captured_comments=("comment_id", "count"),
            avg_comment_likes=("digg_count", "mean"),
            early_comments_60m=("mins_since_post", lambda s: int((s <= 60).sum())),
            top_comment_intent=("rough_intent", lambda s: s.value_counts().index[0] if len(s) else None),
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
    keep = [c for c in keep if c in post_metrics.columns]
    if keep:
        post_summary = post_summary.merge(post_metrics[keep], on="video_id", how="left")
    post_summary = post_summary.sort_values(["captured_comments", "avg_comment_likes"], ascending=False)
    return {
        "comment_intent_summary": intent_summary,
        "post_comment_intents": post_summary,
    }


def creator_leverage(
    posts_author: pd.DataFrame,
    authors: pd.DataFrame,
    authorpost_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if posts_author.empty or authors.empty:
        return pd.DataFrame()
    df = posts_author.merge(
        authors[
            [
                c
                for c in [
                    "author_id",
                    "user_uniqueId",
                    "user_nickname",
                    "stats_followerCount",
                    "stats_heartCount",
                    "stats_videoCount",
                ]
                if c in authors.columns
            ]
        ],
        on="author_id",
        how="left",
    )
    if not authorpost_metrics.empty:
        df = df.merge(authorpost_metrics, on="author_id", how="left")

    followers = pd.to_numeric(df.get("stats_followerCount", 0), errors="coerce").replace({0: np.nan})
    avg_play = pd.to_numeric(df.get("avg_play", 0), errors="coerce").fillna(0)
    df["views_per_follower_history"] = (avg_play / followers).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["creator_scale"] = pd.cut(
        followers.fillna(0),
        bins=[-1, 1_000, 10_000, 100_000, 1_000_000, np.inf],
        labels=["nano", "micro", "mid", "macro", "mega"],
    )
    cols = [
        "video_id",
        "author_id",
        "author_unique_id",
        "author_nickname",
        "creator_scale",
        "stats_followerCount",
        "stats_heartCount",
        "posts_total",
        "active_months",
        "posts_per_month",
        "avg_play",
        "avg_share",
        "eng_per_1k_views_mean",
        "views_per_follower_history",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("views_per_follower_history", ascending=False)
