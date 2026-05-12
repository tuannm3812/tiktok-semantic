# TikTok Semantic

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Pandas](https://img.shields.io/badge/Pandas-analytics-150458?logo=pandas&logoColor=white)](pyproject.toml)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-clustering-F7931E?logo=scikitlearn&logoColor=white)](pyproject.toml)
[![NetworkX](https://img.shields.io/badge/NetworkX-creator%20graphs-2D6A4F)](pyproject.toml)
[![Hackathon](https://img.shields.io/badge/UNSW-2025%20Marketing%20Analytics-00A6D6)](docs/competition_insights.md)
[![Data](https://img.shields.io/badge/data-local%20raw%20files-lightgrey)](.gitignore)

<p align="center">
  <img src="https://p16-tiktokcdn-com.akamaized.net/obj/tiktok-obj/d6415accb8f0986a125f880088f4964f.png" alt="TikTok icon" width="96">
</p>

Semantic marketing analytics for climate-action TikToks.

This repo turns the UNSW Marketing Analytics Hackathon TikTok dataset into reusable analysis tables and insight briefs for answering:

> How can marketing analytics help uncover insights to more effectively promote climate actions on TikTok?

The starter pipeline is based on the supplied notebook and keeps the useful pieces: pickle loading, nested TikTok object normalization, post/author/comment/hashtag metrics, multimodal summary extraction, and first-pass insight tables.

Current analysis is sample-first: the checked workflow runs on the available 10-post `Data_Sample`, and the same scripts are designed to rerun when the full 1,597-post dataset is available locally.

## Repo Layout

- `configs/sample.yaml` points to the supplied `Data_Sample.zip` on Google Drive.
- `scripts/prepare_data.py` extracts the raw zip into `data/raw/Data_Sample/`.
- `scripts/build_dataset.py` builds parquet core tables and CSV insight tables.
- `scripts/deep_dive_nlp.py` builds dependency-light semantic clusters and comment sentiment/emotion tables.
- `src/tiktok_semantic/` contains reusable loaders, normalizers, feature builders, summary parsers, and insight helpers.
- `notebooks/01_competition_analysis.ipynb` is the refined competition notebook for insight generation.
- `docs/competition_insights.md` summarizes the current evidence and recommended competition storyline.
- `data/` and `reports/` are git-ignored so large media, pickles, and outputs stay local.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/prepare_data.py --config configs/sample.yaml
python scripts/build_dataset.py --config configs/sample.yaml
python scripts/deep_dive_nlp.py --config configs/sample.yaml
```

If the pickle files and `Videos/` folders already exist in `data/raw/`, you can skip `prepare_data.py`.

Outputs:

- `data/processed/core/*.parquet`
- `data/processed/analytics/*.csv`
- `data/processed/manifest.csv`

## Suggested Analysis Tracks

1. **Content that travels**: compare reach and `eng_per_1k_views` by content type, duration, hashtags, and multimodal summary themes.
2. **Action messaging**: identify whether posts frame climate as personal action, systemic action, crisis impacts, nature appreciation, or accountability.
3. **Creator leverage**: combine author metrics with post outcomes to find creators whose audience size underpredicts engagement.
4. **Audience response**: roll up comment volume, timing, likes, sentiment, and recurring objections/supportive language.
5. **Network opportunity**: use follower/following samples to map creator overlap, bridge accounts, and regional clusters.

## Competition-Ready Tables

The build script creates a lightweight semantic layer from Gemini's multimodal `summary.txt` files:

- `posts_summaries.parquet`: content type, parsed narrative/mood fields, theme labels, action frame, and local media flags.
- `posts_enriched.parquet`: post metrics joined to the semantic labels.
- `theme_performance.csv`: performance by theme, action frame, and mood.
- `messaging_recommendations.csv`: ranked message/content combinations using reach, engagement efficiency, and shares.
- `comment_intent_summary.csv`: lightweight audience response categories from captured comments.
- `post_comment_intents.csv`: post-level comment response metrics, including early-comment counts.
- `creator_leverage.csv`: creator history and reach-over-follower indicators for activation planning.
- `region_performance.csv`: descriptive regional performance summary.
- `semantic_clusters.csv` and `semantic_cluster_performance.csv`: organic narrative clusters from multimodal summary text.
- `comment_sentiment_emotion.csv`, `comment_sentiment_summary.csv`, and `post_comment_sentiment.csv`: audience sentiment and emotion readouts from captured comments.
- `creator_bridge_metrics.csv`: follower/following graph metrics for identifying bridge creators.

These tables are designed to support claims about how climate-action TikToks should be framed, not just which posts were popular.

## Notes

The original notebook includes heavier NLP steps such as sentence-transformer embeddings, YAKE keyphrases, zero-shot themes, sentiment, emotion, and toxicity. Those belong in an optional NLP layer because they download large models and take longer to run. The base pipeline prepares the clean tables those models need.

Optional transformer NLP is not required for the default sample workflow. Install it only when you want model-based embeddings or sentiment/emotion scoring:

```powershell
python -m pip install -e ".[nlp]"
```
