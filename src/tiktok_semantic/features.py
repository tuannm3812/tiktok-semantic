from __future__ import annotations

import numpy as np
import pandas as pd

from .normalize import (
    add_time_features,
    coerce_numeric,
    dedupe_columns,
    dict_to_frame,
    explode_list_of_dicts,
    explode_record_dicts,
    extract_hashtags,
    flatten_dict_column,
    scalar_columns,
)


COUNT_COLS = ["play_count", "digg_count", "comment_count", "share_count", "collect_count"]


def normalize_posts(raw_posts: object) -> dict[str, pd.DataFrame]:
    posts_raw = add_time_features(dict_to_frame(raw_posts))
    posts = posts_raw[scalar_columns(posts_raw)].copy()
    posts = coerce_numeric(posts, COUNT_COLS + ["duration", "download_count"])
    posts["has_video"] = posts.get("duration", 0).astype(float).gt(0)
    posts["eng_total"] = posts[[c for c in COUNT_COLS if c in posts]].sum(axis=1)
    posts["eng_per_1k_views"] = (
        1000 * posts["eng_total"] / posts["play_count"].replace({0: np.nan})
    ).fillna(0.0)
    posts["url"] = "https://www.tiktok.com/@tiktok/video/" + posts["video_id"].astype(str)

    posts_author = flatten_dict_column(posts_raw, ["video_id"], "author", "author_")
    posts_music = flatten_dict_column(posts_raw, ["video_id"], "music_info", "music_")
    posts_commerce = flatten_dict_column(posts_raw, ["video_id"], "commerce_info", "commerce_")

    comments = explode_list_of_dicts(posts_raw, "comments", ["video_id"])
    comments = dedupe_columns(comments)
    if "user" in comments.columns:
        user = pd.json_normalize(comments["user"].apply(lambda x: x if isinstance(x, dict) else {}).tolist())
        user.columns = [f"user_{c}" for c in user.columns]
        comments = pd.concat([comments.drop(columns=["user"]), user], axis=1)
    if not comments.empty:
        comments = comments.rename(columns={"id": "comment_id"})
        comments["comment_ts"] = pd.to_datetime(
            comments.get("create_time"), unit="s", utc=True, errors="coerce"
        )
        if "ts" in posts.columns:
            comments = comments.merge(posts[["video_id", "ts"]], on="video_id", how="left")
            comments["mins_since_post"] = (
                comments["comment_ts"] - comments["ts"]
            ).dt.total_seconds() / 60

    images = posts_raw[["video_id", "images"]].explode("images", ignore_index=True)
    images = images.dropna(subset=["images"]).rename(columns={"images": "image_url"})
    if not images.empty:
        images["image_idx"] = images.groupby("video_id").cumcount() + 1

    hashtags = posts[["video_id", "title"]].copy()
    hashtags["hashtag"] = hashtags["title"].apply(extract_hashtags)
    hashtags = hashtags.explode("hashtag").dropna(subset=["hashtag"])
    hashtags = hashtags[["video_id", "hashtag"]].drop_duplicates()

    post_metrics = build_post_metrics(posts, comments, images, hashtags)
    return {
        "posts": posts,
        "posts_author": posts_author,
        "posts_music": posts_music,
        "posts_commerce": posts_commerce,
        "posts_comments": comments,
        "posts_images": images,
        "posts_hashtags": hashtags,
        "post_metrics": post_metrics,
    }


def build_post_metrics(
    posts: pd.DataFrame,
    comments: pd.DataFrame,
    images: pd.DataFrame,
    hashtags: pd.DataFrame,
) -> pd.DataFrame:
    out = posts.copy()
    if not hashtags.empty:
        out = out.merge(hashtags.groupby("video_id").size().rename("n_hashtags"), on="video_id", how="left")
    if not images.empty:
        out = out.merge(images.groupby("video_id").size().rename("n_images"), on="video_id", how="left")
    if not comments.empty:
        out = out.merge(
            comments.groupby("video_id").size().rename("n_comments_captured"),
            on="video_id",
            how="left",
        )
        if "digg_count" in comments.columns:
            c = comments.copy()
            c["digg_count"] = pd.to_numeric(c["digg_count"], errors="coerce").fillna(0)
            out = out.merge(
                c.groupby("video_id")["digg_count"].mean().rename("avg_comment_diggs"),
                on="video_id",
                how="left",
            )
    for col in ("n_hashtags", "n_images", "n_comments_captured"):
        out[col] = out.get(col, 0).fillna(0).astype(int)
    out["avg_comment_diggs"] = out.get("avg_comment_diggs", 0).fillna(0.0)
    return out


def normalize_authors(raw_authors: object) -> dict[str, pd.DataFrame]:
    authors_raw = dict_to_frame(raw_authors, key_name="author_id", unify_video_id=False)
    user = flatten_dict_column(authors_raw, ["author_id"], "user", "user_")
    stats = flatten_dict_column(authors_raw, ["author_id"], "stats", "stats_").drop(columns=["author_id"])
    authors = pd.concat([user.reset_index(drop=True), stats.reset_index(drop=True)], axis=1)
    author_metrics = build_author_metrics(authors)
    return {"authors": authors, "author_metrics": author_metrics}


def build_author_metrics(authors: pd.DataFrame) -> pd.DataFrame:
    out = authors.copy()
    stats_cols = [
        "stats_followingCount",
        "stats_followerCount",
        "stats_heartCount",
        "stats_videoCount",
        "stats_diggCount",
    ]
    out = coerce_numeric(out, stats_cols)
    metrics = out[["author_id"] + [c for c in stats_cols if c in out.columns]].copy()
    followers = metrics.get("stats_followerCount", pd.Series(0, index=metrics.index)).replace({0: np.nan})
    videos = metrics.get("stats_videoCount", pd.Series(0, index=metrics.index)).replace({0: np.nan})
    hearts = metrics.get("stats_heartCount", pd.Series(0, index=metrics.index))
    metrics["avg_hearts_per_video"] = (hearts / videos).fillna(0.0)
    metrics["hearts_per_follower"] = (hearts / followers).fillna(0.0)
    metrics["videos_per_1k_followers"] = (1000 * metrics.get("stats_videoCount", 0) / followers).fillna(0.0)
    return metrics


def normalize_author_activity(name: str, raw: object) -> dict[str, pd.DataFrame]:
    frame = dict_to_frame(raw, key_name="author_id", unify_video_id=False)
    exploded = add_time_features(explode_record_dicts(frame, "author_id"))
    exploded = coerce_numeric(exploded, COUNT_COLS + ["duration"])
    if not exploded.empty and "play_count" in exploded.columns:
        exploded["eng_total"] = exploded[[c for c in COUNT_COLS if c in exploded]].sum(axis=1)
        exploded["eng_per_1k_views"] = (
            1000 * exploded["eng_total"] / exploded["play_count"].replace({0: np.nan})
        ).fillna(0.0)
        if "duration" in exploded.columns:
            exploded["has_video"] = exploded["duration"].astype(float).gt(0)
        else:
            exploded["has_video"] = False

    tables = {name: exploded}
    if name == "authorposts" and not exploded.empty:
        tables["authorpost_metrics"] = build_authorpost_metrics(exploded)
    return tables


def build_authorpost_metrics(authorposts: pd.DataFrame) -> pd.DataFrame:
    grp = authorposts.groupby("author_id", dropna=False)
    out = grp.size().rename("posts_total").to_frame()
    if "month" in authorposts.columns:
        out["active_months"] = grp["month"].nunique()
        out["posts_per_month"] = (out["posts_total"] / out["active_months"].replace(0, np.nan)).fillna(0)
    for col in COUNT_COLS:
        if col in authorposts.columns:
            out[f"sum_{col.removesuffix('_count')}"] = grp[col].sum()
            out[f"avg_{col.removesuffix('_count')}"] = grp[col].mean()
    if "eng_per_1k_views" in authorposts.columns:
        out["eng_per_1k_views_mean"] = grp["eng_per_1k_views"].mean()
    return out.reset_index()


def normalize_network(name: str, raw: object, entity_prefix: str) -> dict[str, pd.DataFrame]:
    frame = dict_to_frame(raw, key_name="author_id", unify_video_id=False)
    exploded = explode_record_dicts(frame, "author_id")
    rename = {
        "id": f"{entity_prefix}_id",
        "unique_id": f"{entity_prefix}_unique_id",
        "uniqueId": f"{entity_prefix}_unique_id",
        "nickname": f"{entity_prefix}_nickname",
        "region": f"{entity_prefix}_region",
    }
    exploded = exploded.rename(columns=rename)
    key_cols = [
        "author_id",
        f"{entity_prefix}_id",
        f"{entity_prefix}_unique_id",
        f"{entity_prefix}_nickname",
        f"{entity_prefix}_region",
    ]
    keep = [c for c in key_cols if c in exploded.columns]
    slim = exploded[keep].copy() if keep else exploded
    tables = {name: slim}
    if not slim.empty and f"{entity_prefix}_region" in slim.columns:
        region_col = f"{entity_prefix}_region"
        metrics = slim.groupby("author_id").agg(
            captured=(region_col, "size"),
            distinct_regions=(region_col, "nunique"),
        )
        top_region = (
            slim.groupby(["author_id", region_col]).size().rename("top_region_count").reset_index()
            .sort_values(["author_id", "top_region_count"], ascending=[True, False])
            .drop_duplicates("author_id")
            .rename(columns={region_col: "top_region"})
        )
        metrics = metrics.reset_index().merge(top_region, on="author_id", how="left")
        tables[f"{name}_metrics"] = metrics
    return tables


def build_core_tables(raw: dict[str, object]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables.update(normalize_posts(raw["dtaPosts"]))
    tables.update(normalize_authors(raw["dtaAuthors"]))
    tables.update(normalize_author_activity("authorposts", raw["dtaAuthorPosts"]))
    tables.update(normalize_author_activity("authorfavorites", raw["dtaAuthorFavorites"]))
    tables.update(normalize_network("authorfollowers", raw["dtaAuthorFollowers"], "follower"))
    tables.update(normalize_network("authorfollowings", raw["dtaAuthorFollowings"], "followed"))
    return tables
