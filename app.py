"""
app.py
------
FastAPI web app.

- POST /upload          → upload JSON files, run embedding/clustering pipeline
- WS   /ws/{session_id} → WebSocket for the agent conversation
- GET  /                → serves the frontend
"""

import html
import json
import os
import re
import uuid
import asyncio
import queue
from pathlib import Path
from typing import List, Dict
from urllib.parse import quote

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Load .env
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from pipeline import run_pipeline, PipelineResult
from agent import run_agent
from citation_matcher import (
    embed_source_stories,
    markdown_to_beatbook_entries,
    build_sources_file,
)

# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Beat Book Builder")


def _strip_html(raw: str) -> str:
    """Unescape HTML entities, remove tags, and preserve paragraph breaks.

    Block-level tags become double newlines so the viewer can re-split the
    content into paragraphs. Everything else is collapsed to single spaces.
    """
    # Decode entities first so encoded markup (&lt;p&gt;) becomes real tags.
    text = html.unescape(raw)
    # A second unescape catches double-encoded sources.
    text = html.unescape(text)

    # Convert block-level tags into paragraph/line breaks before stripping.
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(
        r"(?i)</\s*(p|div|li|h[1-6]|blockquote|tr|article|section)\s*>",
        "\n\n",
        text,
    )
    text = re.sub(
        r"(?i)<\s*(p|div|li|h[1-6]|blockquote|tr|article|section)(\s[^>]*)?>",
        "\n\n",
        text,
    )

    # Strip any remaining tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Collapse runs of spaces/tabs but keep newlines, then collapse 3+ newlines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_story(entry: dict) -> dict:
    """Normalize a Chicago Public Media RSS entry (or pass through an already-normalized story)."""
    # Already in the expected format
    if "content" in entry and "title" in entry:
        content = entry.get("content") or ""
        if "<" in content or "&lt;" in content:
            entry = dict(entry)
            entry["content"] = _strip_html(content)
        return entry

    title = entry.get("title", "")
    author = entry.get("author", "")

    # Date: prefer published_parsed, fall back to published, then date
    date = entry.get("published_parsed", "") or entry.get("published", "") or entry.get("date", "")
    # Trim to YYYY-MM-DD if it's an ISO datetime
    if isinstance(date, str) and len(date) > 10:
        date = date[:10]

    # Content: use summary (strip HTML) or content field
    raw_content = entry.get("summary", "") or entry.get("content", "")
    content = _strip_html(raw_content) if ("<" in raw_content or "&lt;" in raw_content) else raw_content

    story = {"title": title, "date": date, "author": author, "content": content}

    # Carry over extra fields that may be useful
    if entry.get("link"):
        story["link"] = entry["link"]
    if entry.get("tags"):
        story["tags"] = entry["tags"]

    return story

# In-memory session store: session_id → PipelineResult
sessions: Dict[str, PipelineResult] = {}

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_stories(files: List[UploadFile] = File(...)):
    """Accept one or more JSON files, merge all stories, run the pipeline.
    Returns a text/event-stream with progress updates, then a final JSON result."""
    all_stories = []
    for f in files:
        raw = await f.read()
        data = json.loads(raw)
        if isinstance(data, list):
            all_stories.extend(data)
        elif isinstance(data, dict):
            # Chicago Public Media format: {"date": ..., "entries": [...]}
            if "entries" in data and isinstance(data["entries"], list):
                all_stories.extend(data["entries"])
            else:
                all_stories.append(data)

    # Normalize all stories to a consistent format
    all_stories = [_normalize_story(s) for s in all_stories]

    if not all_stories:
        return JSONResponse({"error": "No stories found in uploaded files."}, status_code=400)

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        return JSONResponse({"error": "OPENAI_API_KEY not configured."}, status_code=500)

    # Thread-safe queue for progress updates from the pipeline thread
    progress_queue = queue.Queue()

    def on_progress(step: str, fraction: float, detail: str):
        progress_queue.put({"step": step, "fraction": fraction, "detail": detail})

    async def event_stream():
        loop = asyncio.get_event_loop()

        # Start the pipeline in a background thread
        future = loop.run_in_executor(
            None, run_pipeline, all_stories, openai_key, on_progress
        )

        # Stream progress events while the pipeline runs
        while not future.done():
            try:
                msg = progress_queue.get_nowait()
                yield f"data: {json.dumps({'type': 'progress', **msg})}\n\n"
            except queue.Empty:
                pass
            await asyncio.sleep(0.15)

        # Drain any remaining progress messages
        while not progress_queue.empty():
            msg = progress_queue.get_nowait()
            yield f"data: {json.dumps({'type': 'progress', **msg})}\n\n"

        # Get the result — surface any pipeline exception as an SSE error event
        # so the browser doesn't see a truncated stream.
        try:
            result = future.result()
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': f'{type(e).__name__}: {e}'})}\n\n"
            return

        session_id = str(uuid.uuid4())[:8]
        sessions[session_id] = result

        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'num_stories': len(all_stories), 'num_topics': len(result.topics), 'broad_topics': {k: len(v) for k, v in result.broad_topics.items()}, 'specific_topics': {k: len(v) for k, v in result.specific_topics.items()}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET — Agent conversation
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def agent_ws(ws: WebSocket, session_id: str):
    await ws.accept()

    pipeline_result = sessions.get(session_id)
    if not pipeline_result:
        await ws.send_json({"type": "error", "text": "Invalid session. Please upload stories first."})
        await ws.close()
        return

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        await ws.send_json({"type": "error", "text": "ANTHROPIC_API_KEY not configured."})
        await ws.close()
        return

    # ── Callbacks ─────────────────────────────────────────────────────────

    async def on_message(text: str):
        """Send agent text to the frontend."""
        await ws.send_json({"type": "message", "text": text})

    async def on_interview(interview_data: dict) -> str:
        """Send a batch of questions to the frontend, wait for all answers,
        return a single formatted string for the agent to read."""
        questions = interview_data.get("questions", [])
        await ws.send_json({
            "type": "questions",
            "intro": interview_data.get("intro", ""),
            "questions": questions,
        })

        response = await ws.receive_json()
        answers = response.get("answers", [])

        lines = ["Reporter's answers:", ""]
        for i, item in enumerate(answers, 1):
            q = item.get("question", "")
            a = item.get("answer", "")
            if isinstance(a, list):
                a = ", ".join(str(x) for x in a) if a else "(no answer)"
            lines.append(f"{i}. {q}")
            lines.append(f"   → {a}")
            lines.append("")
        return "\n".join(lines)

    async def on_beat_book(filename: str, markdown: str):
        """Save markdown, run the citation matcher, save JSON + sources, send viewer URL."""
        # Save raw markdown first so the user can always fall back to it.
        filepath = OUTPUT_DIR / filename
        filepath.write_text(markdown, encoding="utf-8")
        await ws.send_json({
            "type": "beat_book_markdown_saved",
            "filename": filename,
        })

        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key:
            await ws.send_json({
                "type": "error",
                "text": "OPENAI_API_KEY not configured; skipping citation matching.",
            })
            return

        stem = filepath.stem
        stories = pipeline_result.stories

        citation_progress_queue: queue.Queue = queue.Queue()

        def on_matcher_progress(stage: str, fraction: float, detail: str):
            citation_progress_queue.put({"stage": stage, "fraction": fraction, "detail": detail})

        def run_matcher():
            source_embeddings = embed_source_stories(stories, openai_key, on_matcher_progress)
            entries = markdown_to_beatbook_entries(markdown, source_embeddings, openai_key, on_matcher_progress)
            sources = build_sources_file(stories, source_embeddings)
            return entries, sources

        await ws.send_json({
            "type": "citation_progress",
            "stage": "starting",
            "fraction": 0.0,
            "detail": "Embedding source sentences…",
        })

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, run_matcher)

        while not future.done():
            try:
                msg = citation_progress_queue.get_nowait()
                await ws.send_json({"type": "citation_progress", **msg})
            except queue.Empty:
                await asyncio.sleep(0.15)

        while not citation_progress_queue.empty():
            msg = citation_progress_queue.get_nowait()
            await ws.send_json({"type": "citation_progress", **msg})

        try:
            entries, sources = future.result()
        except Exception as e:
            await ws.send_json({
                "type": "error",
                "text": f"Citation matching failed: {e}. The raw Markdown is still available at /output/{filename}.",
            })
            return

        json_path = OUTPUT_DIR / f"{stem}.json"
        sources_path = OUTPUT_DIR / f"{stem}_sources.json"
        json_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")

        await ws.send_json({
            "type": "beat_book",
            "filename": filename,
            "markdown_path": f"/output/{quote(filename)}",
            "viewer_url": f"/static/viewer/viewer.html?book={quote(stem)}",
            "stem": stem,
        })

    async def on_tool_status(tool_name: str, tool_desc: str, detail: str):
        """Send tool execution status to frontend."""
        await ws.send_json({
            "type": "tool_status",
            "tool_name": tool_name,
            "tool": tool_desc,
            "detail": detail,
        })

    # ── Run agent ─────────────────────────────────────────────────────────

    try:
        await run_agent(
            pipeline_result=pipeline_result,
            anthropic_api_key=anthropic_key,
            on_interview=on_interview,
            on_message=on_message,
            on_beat_book=on_beat_book,
            on_tool_status=on_tool_status,
        )
    except WebSocketDisconnect:
        print(f"Session {session_id}: client disconnected.")
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "text": f"Agent error: {str(e)}"})
        except Exception:
            pass
        raise


# ─────────────────────────────────────────────────────────────────────────────
# STATIC FILES (must be last so it doesn't shadow routes)
# ─────────────────────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")
