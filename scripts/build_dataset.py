"""Build normalized core tables and competition-oriented analytics outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiktok_semantic.features import build_core_tables  # noqa: E402
from tiktok_semantic.insights import (
    comment_intent_tables,
    content_type_performance,
    creator_bridge_metrics,
    creator_leverage,
    hashtag_performance,
    messaging_recommendations,
    region_performance,
    theme_performance,
    top_posts,
)
from tiktok_semantic.io import (
    load_raw_pickles,
    read_config,
    resolve_data_root,
    write_manifest,
    write_table,
)  # noqa: E402
from tiktok_semantic.summaries import (  # noqa: E402
    enrich_posts_with_summaries,
    read_video_summaries,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build normalized TikTok climate analytics tables."
    )
    parser.add_argument("--config", default="configs/sample.yaml")
    args = parser.parse_args()

    cfg = read_config(args.config)
    raw_root = resolve_data_root(cfg["raw_dir"])
    processed = Path(cfg["processed_dir"])
    core_dir = processed / "core"
    analytics_dir = processed / "analytics"

    raw = load_raw_pickles(raw_root)
    tables = build_core_tables(raw)

    videos_dir = raw_root / "Videos"
    summary_filename = cfg.get("summary", {}).get("summary_filename", "summary.txt")
    summaries = read_video_summaries(videos_dir, summary_filename)
    if not summaries.empty:
        tables["posts_summaries"] = summaries
        tables["posts_enriched"] = enrich_posts_with_summaries(tables["post_metrics"], summaries)

    manifest_rows = []
    for name, df in sorted(tables.items()):
        path = core_dir / f"{name}.parquet"
        write_table(df, path)
        manifest_rows.append(
            {
                "table": name,
                "category": "core",
                "rows": len(df),
                "cols": len(df.columns),
                "path": str(path),
            }
        )

    min_views = cfg.get("engagement", {}).get("min_views_for_efficiency_rank", 1000)
    insight_tables = {}
    insight_tables.update(top_posts(tables["post_metrics"], min_views=min_views))
    insight_tables["region_performance"] = region_performance(tables["post_metrics"])
    insight_tables["hashtag_performance"] = hashtag_performance(
        tables["post_metrics"], tables["posts_hashtags"]
    )
    insight_tables.update(comment_intent_tables(tables["posts_comments"], tables["post_metrics"]))
    insight_tables["creator_leverage"] = creator_leverage(
        tables["posts_author"],
        tables["authors"],
        tables["authorpost_metrics"],
    )
    insight_tables["creator_bridge_metrics"] = creator_bridge_metrics(
        tables["posts_author"],
        tables["authors"],
        tables["authorfollowers"],
        tables["authorfollowings"],
        tables["post_metrics"],
    )
    if not summaries.empty:
        insight_tables["content_type_performance"] = content_type_performance(
            tables["post_metrics"], summaries
        )
        insight_tables["theme_performance"] = theme_performance(tables["posts_enriched"])
        insight_tables["messaging_recommendations"] = messaging_recommendations(
            tables["posts_enriched"],
            min_views=min_views,
        )

    for name, df in sorted(insight_tables.items()):
        path = analytics_dir / f"{name}.csv"
        write_table(df, path)
        manifest_rows.append(
            {
                "table": name,
                "category": "analytics",
                "rows": len(df),
                "cols": len(df.columns),
                "path": str(path),
            }
        )

    manifest = write_manifest(manifest_rows, processed / "manifest.csv")
    print(f"Built {len(manifest)} tables from {raw_root}")
    print(f"Manifest: {processed / 'manifest.csv'}")


if __name__ == "__main__":
    main()
