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

# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Beat Book Builder")


def _strip_html(raw: str) -> str:
    """Unescape HTML entities, remove tags, and collapse whitespace."""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_story(entry: dict) -> dict:
    """Normalize a Chicago Public Media RSS entry (or pass through an already-normalized story)."""
    # Already in the expected format
    if "content" in entry and "title" in entry:
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
    content = _strip_html(raw_content) if "<" in raw_content else raw_content

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

        # Get the result
        result = future.result()
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

    async def on_interview(question_data: dict) -> str:
        """Send question to frontend, wait for answer."""
        await ws.send_json({
            "type": "question",
            "question": question_data.get("question", ""),
            "question_type": question_data.get("question_type", "free_response"),
            "options": question_data.get("options", []),
        })
        # Wait for the user's response
        answer_msg = await ws.receive_json()
        return answer_msg.get("answer", "")

    async def on_beat_book(filename: str, markdown: str):
        """Save beat book and notify frontend."""
        filepath = OUTPUT_DIR / filename
        filepath.write_text(markdown, encoding="utf-8")
        await ws.send_json({
            "type": "beat_book",
            "filename": filename,
            "content": markdown,
        })

    async def on_tool_status(tool_desc: str, detail: str):
        """Send tool execution status to frontend."""
        await ws.send_json({
            "type": "tool_status",
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
