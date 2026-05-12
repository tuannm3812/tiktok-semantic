from __future__ import annotations

import numpy as np
import pandas as pd


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
