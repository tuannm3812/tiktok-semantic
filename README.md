# TikTok Climate Change Marketing Analytics

This repo turns the UNSW Marketing Analytics Hackathon TikTok dataset into reusable analysis tables for answering:

> How can marketing analytics help uncover insights to more effectively promote climate actions on TikTok?

The starter pipeline is based on the supplied notebook and keeps the useful pieces: pickle loading, nested TikTok object normalization, post/author/comment/hashtag metrics, multimodal summary extraction, and first-pass insight tables.

## Repo Layout

- `configs/sample.yaml` points to the supplied `Data_Sample.zip` on Google Drive.
- `scripts/prepare_data.py` extracts the raw zip into `data/raw/Data_Sample/`.
- `scripts/build_dataset.py` builds parquet core tables and CSV insight tables.
- `src/tiktok_semantic/` contains reusable loaders, normalizers, feature builders, summary parsers, and insight helpers.
- `notebooks/01_competition_analysis.ipynb` is the refined competition notebook for insight generation.
- `data/` and `reports/` are git-ignored so large media, pickles, and outputs stay local.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/prepare_data.py --config configs/sample.yaml
python scripts/build_dataset.py --config configs/sample.yaml
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

These tables are designed to support claims about how climate-action TikToks should be framed, not just which posts were popular.

## Notes

The original notebook includes heavier NLP steps such as sentence-transformer embeddings, YAKE keyphrases, zero-shot themes, sentiment, emotion, and toxicity. Those belong in an optional NLP layer because they download large models and take longer to run. The base pipeline prepares the clean tables those models need.
