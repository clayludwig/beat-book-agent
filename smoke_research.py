"""
Smoke test for research_agent.run_research_agent.

Creates a temp sandbox with a tiny fake beat book, runs the agent with lowered
limits (fewer turns, fewer web searches/fetches) to keep token spend small,
and prints the revised markdown plus a short event trail.

Run: python smoke_research.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

import research_agent

# Dial down cost before importing anything that reads these constants.
research_agent.MAX_TURNS = 8
research_agent.WEB_SEARCH_MAX_USES = 3
research_agent.WEB_FETCH_MAX_USES = 3

from research_agent import run_research_agent  # noqa: E402


SANDBOX = Path(__file__).parent / "output" / "sandboxes" / "_smoke_test"
MARKDOWN_FILE = "beat_book.md"

FAKE_MARKDOWN = """# San Francisco Board of Education — Beat Book

A reporting guide for covering the San Francisco Unified School District
(SFUSD) Board of Education.

## Overview

The San Francisco Board of Education is the elected seven-member body that
governs SFUSD, the school district serving the city of San Francisco.

## Key People

- Board president (name TBD)
- Superintendent (name TBD)

## Recent Coverage Themes

- School closures and consolidation debates
- Budget shortfalls
- Enrollment decline

## Open Questions

- What are the board's upcoming meeting dates this spring?
- Who are the current commissioners?
"""

INTERVIEW_LOG = [
    {
        "intro": "Quick questions to calibrate the beat book.",
        "questions": [],
        "answers": [
            {
                "question": "What is your primary angle on this beat?",
                "answer": "Accountability reporting — I'm especially interested in how budget shortfalls are driving school-closure decisions.",
            },
            {
                "question": "What level of reader are you writing for?",
                "answer": "General city audience, not education specialists.",
            },
            {
                "question": "Any stories you already know you want to pursue?",
                "answer": "A long feature on the superintendent's first year in office.",
            },
        ],
    }
]


def make_sandbox() -> Path:
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    SANDBOX.mkdir(parents=True, exist_ok=True)
    (SANDBOX / MARKDOWN_FILE).write_text(FAKE_MARKDOWN, encoding="utf-8")
    return SANDBOX


async def main() -> int:
    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 1

    sandbox = make_sandbox()
    print(f"Sandbox: {sandbox}")

    events: list[str] = []

    async def on_progress(stage: str, detail: str) -> None:
        line = f"[progress] {stage}: {detail}"
        print(line)
        events.append(line)

    async def on_tool_status(tool_name: str, desc: str, detail: str) -> None:
        line = f"[tool] {tool_name} — {detail}"[:200]
        print(line)
        events.append(line)

    async def on_text(text: str) -> None:
        snippet = text if len(text) < 200 else text[:200] + "…"
        line = f"[text] {snippet}"
        print(line)
        events.append(line)

    try:
        final_markdown = await run_research_agent(
            sandbox_dir=sandbox,
            markdown_filename=MARKDOWN_FILE,
            interview_log=INTERVIEW_LOG,
            anthropic_api_key=api_key,
            on_progress=on_progress,
            on_tool_status=on_tool_status,
            on_text=on_text,
        )
    except Exception as e:
        print(f"\nERROR: run_research_agent raised: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    original = FAKE_MARKDOWN
    print("\n" + "=" * 72)
    print("LENGTH: original", len(original), "→ revised", len(final_markdown))
    print("CHANGED:", final_markdown != original)
    print("=" * 72)
    print("\n--- REVISED MARKDOWN (first 2000 chars) ---\n")
    print(final_markdown[:2000])
    if len(final_markdown) > 2000:
        print(f"\n[... {len(final_markdown) - 2000} more chars ...]")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
