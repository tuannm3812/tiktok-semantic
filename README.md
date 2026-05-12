# TikTok Semantic

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Pandas](https://img.shields.io/badge/Pandas-analytics-150458?logo=pandas&logoColor=white)](pyproject.toml)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-clustering-F7931E?logo=scikitlearn&logoColor=white)](pyproject.toml)
[![NetworkX](https://img.shields.io/badge/NetworkX-creator%20graphs-2D6A4F)](pyproject.toml)
[![Hackathon](https://img.shields.io/badge/UNSW-2025%20Marketing%20Analytics-00A6D6)](docs/competition_insights.md)
[![Data](https://img.shields.io/badge/data-local%20raw%20files-lightgrey)](.gitignore)

<p align="center">
  <img src="https://p16-va-tiktok.ibyteimg.com/obj/musically-maliva-obj/3736e47fd36b7b1015749101ecab1891.jpeg" alt="TikTok climate analytics visual" width="160">
</p>

Semantic marketing analytics for climate-action TikTok content.

This repository supports the UNSW Marketing Analytics Hackathon 2025 challenge:

> How can marketing analytics help uncover insights to more effectively promote climate actions on TikTok?

The project converts raw TikTok post, creator, comment, hashtag, follower/following, and multimodal summary data into reusable analysis tables, charts, and an evidence-backed competition narrative.

## Current Scope

The checked workflow is **sample-first**. It currently runs on the available 10-post `Data_Sample`, while keeping the pipeline ready to rerun when the full 1,597-post dataset is available locally.

Because the current dataset is small, results should be framed as directional evidence and a reusable analytical framework rather than final population-level claims.

## What This Project Does

- Normalizes nested TikTok pickle objects into tidy post, author, comment, hashtag, image, music, and network tables.
- Extracts semantic features from Gemini-generated `summary.txt` files for each post.
- Builds post-level performance metrics such as reach, shares, comments, and engagement per 1K views.
- Creates sample-first deep dives for message framing, content format, audience response, creator leverage, and creator bridge potential.
- Produces an executed notebook with charts and a concise insight brief for presentation development.

## Repository Structure

```text
configs/
  sample.yaml                  # Local paths and pipeline settings
docs/
  competition_insights.md      # Competition-ready insight narrative
notebooks/
  01_competition_analysis.ipynb
scripts/
  prepare_data.py              # Extract raw data zip
  build_dataset.py             # Build normalized and analytical tables
  deep_dive_nlp.py             # Build semantic clusters and comment sentiment/emotion tables
src/tiktok_semantic/
  features.py                  # Core feature builders
  insights.py                  # Analytical rollups and campaign tables
  io.py                        # Config, pickle, parquet, and CSV helpers
  normalize.py                 # Reusable normalization utilities
  summaries.py                 # Multimodal summary parsing and labels
```

`data/` and `reports/` are intentionally git-ignored so raw media, pickle files, and generated outputs stay local.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If the supplied raw zip has not been extracted:

```powershell
python scripts/prepare_data.py --config configs/sample.yaml
```

Build the analysis tables:

```powershell
python scripts/build_dataset.py --config configs/sample.yaml
python scripts/deep_dive_nlp.py --config configs/sample.yaml
```

Then open:

```text
notebooks/01_competition_analysis.ipynb
docs/competition_insights.md
```

## Outputs

The build scripts write local outputs to:

- `data/processed/core/*.parquet`
- `data/processed/analytics/*.csv`
- `data/processed/manifest.csv`

Key analytical outputs include:

| Output | Purpose |
| --- | --- |
| `posts_enriched.parquet` | Post metrics joined to content type, theme, mood, and action-frame labels |
| `theme_performance.csv` | Performance by theme, action frame, and mood |
| `messaging_recommendations.csv` | Ranked message/content combinations using reach, engagement efficiency, and shares |
| `comment_intent_summary.csv` | Lightweight audience response categories from captured comments |
| `post_comment_intents.csv` | Post-level comment response metrics and early-comment counts |
| `semantic_clusters.csv` | Organic narrative clusters from multimodal summary text |
| `semantic_cluster_performance.csv` | Cluster-level reach, engagement, share, and comment performance |
| `comment_sentiment_summary.csv` | Comment sentiment and emotion rollups |
| `post_comment_sentiment.csv` | Post-level sentiment and emotion indicators |
| `creator_leverage.csv` | Creator history and reach-over-follower indicators |
| `creator_bridge_metrics.csv` | Follower/following graph metrics for bridge-creator discovery |
| `region_performance.csv` | Descriptive regional performance summary |

## Current Sample Findings

The sample supports a campaign framing that is useful to test on the full dataset:

> Reach comes from hooks, persuasion comes from frames, and conversion opportunities come from comments.

Directional findings:

- Video-led climate narratives carry the strongest reach in the sample.
- Practical action framing can outperform awareness-only framing on engagement efficiency.
- Information questions and confusion in comments are valuable signals for follow-up content.
- Creator selection should consider bridge potential and historical overperformance, not follower count alone.
- Broad climate hashtags need contextual tags that clarify the content lane, such as transport, eco-lifestyle, politics, extreme weather, or pop-culture analogy.

See [docs/competition_insights.md](docs/competition_insights.md) for the full sample-first insight brief.

## Optional NLP Layer

The default workflow avoids large model downloads and uses transparent, dependency-light methods:

- TF-IDF + KMeans for semantic clustering.
- Lexicon/rule-based sentiment and emotion labels for comments.

For a larger dataset or final presentation robustness, install the optional NLP dependencies:

```powershell
python -m pip install -e ".[nlp]"
```

Future upgrades can replace the default methods with sentence-transformer embeddings and transformer-based sentiment/emotion classifiers while preserving the same output table structure.

## Data Notes

- Raw data is expected under `data/raw`.
- Generated files under `data/processed` are local artifacts and are not committed.
- The repository currently documents sample results only because the full 1,597-post dataset is not available in the local workspace.

## Suggested Next Steps

1. Add the full 1,597-post dataset to `data/raw`.
2. Rerun `build_dataset.py` and `deep_dive_nlp.py`.
3. Re-execute the notebook to refresh charts and rankings.
4. Convert the strongest findings into a slide-ready creative playbook: hook, format, frame, creator type, hashtag bundle, and comment follow-up.
