from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

THEME_PATTERNS = {
    "climate_impacts": r"\b(flood|fire|wildfire|drought|heatwave|storm|disaster|melting|sea level|extreme weather|climate crisis)\b",
    "solutions_action": r"\b(solution|action|reduce|reuse|recycle|renewable|solar|wind|plant|vote|policy|petition|protest|activism)\b",
    "accountability_policy": r"\b(government|policy|corporate|company|fossil fuel|oil|gas|emissions|carbon|greenwash|accountability)\b",
    "education_awareness": r"\b(explain|learn|science|fact|evidence|awareness|educat|inform|myth|understand)\b",
    "nature_biodiversity": r"\b(nature|animal|wildlife|ocean|forest|tree|reef|species|biodiversity|planet)\b",
    "lifestyle_consumer": r"\b(food|vegan|diet|fashion|thrift|plastic|waste|travel|consumer|lifestyle|home)\b",
}

MOOD_PATTERNS = {
    "urgent_alarm": r"\b(urgent|alarming|grave|fear|anxious|dire|warning|crisis|catastrophic)\b",
    "hopeful_empowering": r"\b(hope|optimistic|empower|inspiring|solution|positive|encouraging)\b",
    "humorous_ironic": r"\b(humor|funny|satirical|sarcastic|ironic|playful|meme)\b",
    "angry_critical": r"\b(angry|frustrated|critical|outrage|blame|condemn)\b",
    "reflective_somber": r"\b(somber|serious|reflective|sad|melancholy|concerned)\b",
}

ACTION_PATTERNS = {
    "individual_action": r"\b(you can|individual|personal|at home|lifestyle|recycle|diet|shop|reduce|reuse)\b",
    "collective_action": r"\b(we can|community|together|collective|movement|join|protest|campaign)\b",
    "policy_systemic": r"\b(policy|government|systemic|corporate|regulation|vote|petition|fossil fuel|accountability)\b",
    "awareness_only": r"\b(awareness|inform|educat|explain|shows|highlights|depicts)\b",
}


SUMMARY_PATTERNS = {
    "overall_mood_tone": r"\*\*.*?overall mood.*?\*\*\n?(.*?)(?=\n\*\*|$)",
    "overall_narrative_message": r"\*\*.*?narrative.*?\*\*\n?(.*?)(?=\n\*\*|$)",
    "prominent_objects_actions": r"\*\*.*?(?:key events|prominent objects).*?\*\*\n?(.*?)(?=\n\*\*|$)",
    "description_storyline": r"\*\*.*?(?:describe the video|storyline).*?\*\*\n?(.*?)(?=\n\*\*|$)",
    "music_description": r"\*\*.*?(?:music track|audio used).*?\*\*\n?(.*?)(?=\n\*\*|$)",
    "audio_nature": r"\*\*.*?(?:nature of the audio).*?:\*\*\s*(.*?)(?=\n\*\*|$)",
    "prominent_audio_content": r"\*\*.*?(?:prominent sounds|spoken content|audio content|prominent lines).*?:\*\*\n?(.*?)(?=\n\*\*|$)",
}


def classify_content_type(folder: Path) -> str:
    has_video = (folder / "video.mp4").exists()
    has_audio = (folder / "audio.mp3").exists()
    has_images = any(folder.glob("image_*.jpg"))
    if has_video:
        return "video"
    if has_audio and has_images:
        return "images_audio"
    if has_images:
        return "images_music"
    return "unknown"


def extract_summary_fields(text: str | None) -> dict[str, str | None]:
    result = {field: None for field in SUMMARY_PATTERNS}
    if not isinstance(text, str) or not text.strip():
        return result
    for field, pattern in SUMMARY_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            result[field] = match.group(match.lastindex or 1).strip()
    return result


def compact_text(*parts: object) -> str:
    return " ".join(str(part).strip() for part in parts if isinstance(part, str) and part.strip())


def first_regex_label(text: str, patterns: dict[str, str], default: str) -> str:
    for label, pattern in patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return default


def all_regex_labels(text: str, patterns: dict[str, str]) -> list[str]:
    return [
        label
        for label, pattern in patterns.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


def read_video_summaries(videos_dir: str | Path, summary_filename: str = "summary.txt") -> pd.DataFrame:
    videos_dir = Path(videos_dir)
    rows = []
    if not videos_dir.exists():
        return pd.DataFrame(columns=["video_id", "content_type", "summary_text"])
    for folder in sorted(p for p in videos_dir.iterdir() if p.is_dir()):
        summary_path = folder / summary_filename
        text = summary_path.read_text(encoding="utf-8", errors="replace") if summary_path.exists() else None
        row = {
            "video_id": folder.name,
            "content_type": classify_content_type(folder),
            "summary_text": text,
            "n_image_files": len(list(folder.glob("image_*.jpg"))),
            "has_local_video": (folder / "video.mp4").exists(),
            "has_local_audio": (folder / "audio.mp3").exists(),
        }
        row.update(extract_summary_fields(text))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["semantic_text"] = out.apply(
        lambda r: compact_text(
            r.get("overall_narrative_message"),
            r.get("prominent_objects_actions"),
            r.get("overall_mood_tone"),
            r.get("summary_text"),
        ),
        axis=1,
    )
    out["theme_labels"] = out["semantic_text"].apply(lambda x: all_regex_labels(x, THEME_PATTERNS))
    out["primary_theme"] = out["theme_labels"].apply(lambda x: x[0] if x else "uncategorized")
    out["mood_label"] = out["semantic_text"].apply(
        lambda x: first_regex_label(x, MOOD_PATTERNS, "neutral_or_mixed")
    )
    out["action_frame"] = out["semantic_text"].apply(
        lambda x: first_regex_label(x, ACTION_PATTERNS, "no_clear_action")
    )
    out["n_theme_labels"] = out["theme_labels"].apply(len)
    out["theme_labels"] = out["theme_labels"].apply(lambda labels: ", ".join(labels))
    return out


def enrich_posts_with_summaries(post_metrics: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "video_id",
        "content_type",
        "primary_theme",
        "theme_labels",
        "mood_label",
        "action_frame",
        "n_theme_labels",
        "overall_narrative_message",
        "overall_mood_tone",
    ]
    keep = [col for col in summary_cols if col in summaries.columns]
    out = post_metrics.merge(summaries[keep], on="video_id", how="left")
    for col in ("content_type", "primary_theme", "mood_label", "action_frame"):
        if col in out.columns:
            out[col] = out[col].fillna("missing_summary")
    return out
