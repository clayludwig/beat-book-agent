"""
clean.py — Convert alt-format RSS-style JSONs into the standard story format.

Reads all JSON files in the source-stories-alt/ folder tree, converts each
entry to the standard schema used by the Beat Book pipeline, strips HTML from
summaries, and writes one merged output file.

Usage:
    cd "Beat Book Topic Categories"
    python3 source-stories-alt/clean.py

Output:
    source-stories-alt/cleaned_stories.json
"""

import json
import glob
import re
import os
from pathlib import Path
from datetime import datetime


def strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#x27;|&#39;", "'", text)
    text = re.sub(r"&nbsp;", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_date(entry: dict) -> dict:
    """Extract year, month, day and ISO date string."""
    raw = entry.get("published_parsed") or entry.get("published", "")
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        try:
            dt = datetime.strptime(raw[:10], "%Y-%m-%d")
        except Exception:
            return {"date": raw[:10] if len(raw) >= 10 else "", "year": 0, "month": 0, "day": 0}
    return {
        "date": dt.strftime("%Y-%m-%d"),
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
    }


def convert_entry(entry: dict) -> dict:
    """Convert one RSS-style entry to the standard story dict."""
    dates = parse_date(entry)

    content = strip_html(entry.get("summary", ""))
    title   = entry.get("title", "").strip()
    author  = entry.get("author", "").strip()
    link    = entry.get("link", "")
    tags    = entry.get("tags", [])

    # Build a content string with header metadata
    header_parts = [title]
    if dates["date"]:
        header_parts.append(dates["date"])
    if author:
        header_parts.append(f"Author/Byline: {author}")
    content_full = "\n\n".join(header_parts) + "\n\n" + content

    return {
        "title": title,
        "date": dates["date"],
        "author": author,
        "content": content_full,
        "docref": link,
        "article_id": entry.get("id", link),
        "content_source": "rss_summary",
        "year": dates["year"],
        "month": dates["month"],
        "day": dates["day"],
        "tags": tags if isinstance(tags, list) else [],
    }


def main():
    script_dir = Path(__file__).parent
    pattern    = str(script_dir / "**" / "*.json")
    out_file   = script_dir / "cleaned_stories.json"

    all_stories = []
    seen_ids    = set()

    files = sorted(glob.glob(pattern, recursive=True))
    # Exclude our own output file
    files = [f for f in files if Path(f).name != "cleaned_stories.json"]

    print(f"Found {len(files)} JSON files in {script_dir}")

    for filepath in files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠ Skipping {filepath}: {e}")
            continue

        entries = data.get("entries", [])
        if not entries:
            continue

        converted = 0
        for entry in entries:
            story = convert_entry(entry)

            # Deduplicate by article_id
            aid = story["article_id"]
            if aid in seen_ids:
                continue
            seen_ids.add(aid)

            # Skip entries with no real content
            if len(story["content"]) < 50:
                continue

            all_stories.append(story)
            converted += 1

        print(f"  ✓ {Path(filepath).name}: {converted} stories (from {len(entries)} entries)")

    # Sort by date
    all_stories.sort(key=lambda s: s.get("date", ""))

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_stories, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Wrote {len(all_stories)} stories to {out_file}")


if __name__ == "__main__":
    main()
