# Beat Book Builder

A web application that turns a corpus of news articles into an interactive, AI-generated **beat book** — a practical reporting guide for journalists covering a specific topic area. Upload JSON story files (e.g. from an RSS feed), and the system automatically discovers topics via embedding and clustering, then walks the reporter through an AI-guided interview to produce a tailored beat book.

Built for [Chicago Public Media](https://chicago.suntimes.com/) story data, but works with any news corpus in a supported JSON format.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture Overview](#architecture-overview)
- [Pipeline: Step by Step](#pipeline-step-by-step)
  - [1. Upload & Ingest](#1-upload--ingest)
  - [2. Text Extraction & Normalization](#2-text-extraction--normalization)
  - [3. Embedding](#3-embedding)
  - [4. Dimensionality Reduction](#4-dimensionality-reduction)
  - [5. Clustering](#5-clustering)
  - [6. Topic Labeling](#6-topic-labeling)
- [Agent: The Interview & Beat Book Generation](#agent-the-interview--beat-book-generation)
  - [Agent Tools](#agent-tools)
  - [Agent Loop](#agent-loop)
- [Frontend](#frontend)
- [Accepted JSON Formats](#accepted-json-formats)
- [Tech Stack](#tech-stack)
- [Setup & Running](#setup--running)
- [Project Structure](#project-structure)

---

## How It Works

1. **Upload** — The reporter uploads one or more JSON files containing news articles through a drag-and-drop web interface.
2. **Analyze** — The server runs each article through an NLP pipeline: embed the text, reduce dimensions, cluster into topics at two granularities (broad and specific), and label each cluster with an LLM.
3. **Interview** — A Claude-powered AI agent connects over WebSocket, explores the discovered topics, and asks the reporter 3-5 targeted questions about their beat, audience, and goals.
4. **Generate** — The agent synthesizes everything — topics, article content, and reporter answers — into a polished Markdown beat book with sources, story ideas, context, and reporting tips.

---

## Architecture Overview

```
Browser (static HTML/JS/CSS)
    │
    ├── POST /upload          → streams SSE progress events
    │       │
    │       └── pipeline.py   → embed → UMAP → HDBSCAN → LLM label
    │
    └── WS /ws/{session_id}   → bidirectional agent conversation
            │
            └── agent.py      → Claude tool-use loop
                               (reads topics/stories, interviews user, writes beat book)
```

The app server is **FastAPI** running on **Uvicorn**. The pipeline runs CPU-bound work (embedding API calls, UMAP, HDBSCAN) in a thread pool so the async server stays responsive. The agent conversation happens over a WebSocket, allowing real-time back-and-forth between the Claude agent and the reporter's browser.

---

## Pipeline: Step by Step

The pipeline lives in `pipeline.py` and is called by the `/upload` endpoint in `app.py`. It takes a list of story dicts and returns a `PipelineResult` containing stories, topics, and lookup structures.

### 1. Upload & Ingest

**File:** `app.py` — `upload_stories()`

When the reporter uploads JSON files, the server reads each file and detects its format. It handles two structures:

- **Flat array** — `[{story}, {story}, ...]` — each object is a story.
- **Chicago Public Media / RSS wrapper** — `{"date": "...", "entries": [...]}` — the `entries` array is unpacked.

All entries are then normalized into a consistent internal format (see [Accepted JSON Formats](#accepted-json-formats)).

**Tech:** FastAPI's `UploadFile` with `python-multipart` for multipart form parsing.

### 2. Text Extraction & Normalization

**File:** `app.py` — `_normalize_story()`, `_strip_html()`

Each story is normalized to a flat dict with these fields:

| Field     | Source (Chicago/RSS format)                   | Source (flat format) |
|-----------|----------------------------------------------|----------------------|
| `title`   | `entry.title`                                | `story.title`        |
| `date`    | `entry.published_parsed` or `entry.published`, trimmed to `YYYY-MM-DD` | `story.date` |
| `author`  | `entry.author`                               | `story.author`       |
| `content` | `entry.summary`, HTML-unescaped and tag-stripped | `story.content`   |
| `link`    | `entry.link` (carried through if present)    | —                    |
| `tags`    | `entry.tags` (carried through if present)    | —                    |

HTML processing uses Python's `html.unescape()` to decode entities (`&lt;` -> `<`), then a regex to strip all tags, then whitespace collapsing.

### 3. Embedding

**File:** `pipeline.py` — `_story_to_text()`, `_embed_batch()`, `_load_or_embed()`

Each story is converted to a text representation for embedding:
- The **title**, a **section line** (if found in the first 10 lines of content), and the **first 400 words** of content are concatenated.

These text blocks are sent to the **OpenAI Embeddings API** in batches of 100.

- **Model:** `text-embedding-3-small` — produces 1536-dimensional vectors.
- **Caching:** Embeddings are cached to `.cache/embeddings.pkl` keyed by a hash of the first 10 texts + model name. On re-runs with the same data, embeddings load instantly from cache.

**Tech:** OpenAI Python SDK (`openai`), NumPy for vector storage.

### 4. Dimensionality Reduction

**File:** `pipeline.py` — `_reduce()`, `_umap_params()`

The high-dimensional embedding vectors (1536-d) are projected down to a lower-dimensional space to make clustering feasible and to capture local structure.

- **Algorithm:** [UMAP](https://umap-learn.readthedocs.io/) (Uniform Manifold Approximation and Projection)
- **Parameters are adaptive** based on corpus size `n`:
  - `n_components`: `min(15, max(5, n // 40))` — more dimensions for larger corpora
  - `n_neighbors`: `min(30, max(5, int(n ** 0.55)))` — balances local vs. global structure
  - `min_dist`: `0.0` — allows tight clusters
  - `metric`: `cosine` — standard for text embeddings

UMAP preserves local neighborhood relationships from the high-dimensional space, meaning articles that are semantically similar end up near each other in the reduced space.

**Tech:** `umap-learn` library (built on NumPy/SciPy/scikit-learn).

### 5. Clustering

**File:** `pipeline.py` — `_cluster()`, `_assign_outliers()`, `_cluster_sizes()`

Stories are clustered at **two granularities** to produce both broad themes and specific sub-topics:

- **Algorithm:** [HDBSCAN](https://hdbscan.readthedocs.io/) (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
- **Broad clusters:** `min_cluster_size = max(4, n // 25)` — produces fewer, larger groups
- **Specific clusters:** `min_cluster_size = max(2, n // 60)` — produces more, finer-grained groups
- Both use `min_samples=2`, Euclidean distance on the UMAP-reduced space, and the "excess of mass" (`eom`) cluster selection method.

**Outlier reassignment:** HDBSCAN labels some points as noise (`-1`). After clustering, every noise point is assigned to its nearest cluster by Euclidean distance to cluster centroids. This ensures every story belongs to at least one topic.

**Why HDBSCAN over K-Means?** HDBSCAN doesn't require specifying the number of clusters in advance — it discovers them from the data's density structure. This is critical because we don't know how many topics a given news corpus will contain.

**Tech:** `hdbscan` library, NumPy.

### 6. Topic Labeling

**File:** `pipeline.py` — `_label_cluster()`, `_label_all()`

Each cluster gets a human-readable topic label generated by an LLM:

1. For each cluster, the **most representative stories** are selected — the ones closest to the cluster's centroid in the reduced space (up to 8 stories).
2. Their headlines and a 30-word excerpt are formatted into a prompt.
3. The LLM is asked to return a concise 2-5 word topic label describing the shared subject matter (e.g. "High School Basketball", "City Budget Disputes", "Immigration Policy").

- **Model:** `gpt-5-mini` (configurable in `pipeline.py`)
- **Prompt engineering:** The prompt explicitly instructs the model to focus on *what* the articles are about, not *where* they're from, to avoid generic geographic labels.

This runs once for broad clusters and once for specific clusters.

**Tech:** OpenAI Chat Completions API.

---

## Agent: The Interview & Beat Book Generation

**File:** `agent.py`

After the pipeline finishes, the reporter's browser opens a WebSocket connection and a Claude-powered agent takes over. The agent uses Anthropic's [tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) feature to interact with the pipeline results and the reporter.

### Agent Tools

| Tool | Type | Description |
|------|------|-------------|
| `view_topics` | Local | Returns all broad and specific topics with story counts |
| `list_stories_in_topic` | Local | Lists stories belonging to a given topic |
| `read_story` | Local | Reads full content of a story by index (truncated to 3000 chars) |
| `search_stories` | Local | Keyword search across story titles and content |
| `interview_user` | Interactive | Sends a question to the reporter via WebSocket and awaits their response. Supports `checklist`, `single_choice`, `multiple_choice`, and `free_response` question types |
| `generate_beat_book` | Output | Writes the final Markdown beat book to the `output/` directory and delivers it to the browser |

### Agent Loop

1. The agent receives a system prompt defining its role as a journalism mentor.
2. It starts by calling `view_topics` to survey the topic landscape.
3. It reads representative stories to understand the coverage.
4. It uses `interview_user` to ask the reporter 3-5 questions — starting with a checklist of topics for them to select their beat, then follow-ups about audience, experience, and needs.
5. It digs deeper into relevant stories using `read_story` and `search_stories`.
6. It calls `generate_beat_book` with a complete Markdown document.

The loop runs for up to 40 turns. If the model hits the token limit mid-generation, it's prompted to continue. The loop exits when the beat book is saved or the model stops calling tools.

- **Model:** `claude-sonnet-4-6`
- **Max tokens per turn:** 16,384

**Tech:** Anthropic Python SDK (`anthropic`), async/await for WebSocket communication.

---

## Frontend

**Files:** `static/index.html`, `static/app.js`, `static/style.css`

The frontend is a single-page app with two screens:

### Upload Screen
- Drag-and-drop or file-picker for JSON files
- Real-time progress bar during pipeline execution via Server-Sent Events (SSE)
- Progress is broken into weighted stages: embedding (30%), reducing (10%), clustering (10%), labeling (50%)

### Chat Screen
- Real-time WebSocket connection to the agent
- Renders agent messages, thinking indicators (with tool status), and interactive question UI
- Question types render as: checkboxes (checklist/multiple choice), radio buttons (single choice), or a textarea (free response)
- Beat book delivery shows a download link to the generated Markdown file

**Tech:** Vanilla JavaScript (no framework), CSS custom properties for theming (dark mode).

---

## Accepted JSON Formats

### Format 1: Flat Array (generic)

```json
[
  {
    "title": "Article headline",
    "date": "2026-02-06",
    "author": "Reporter Name",
    "content": "Full article text..."
  }
]
```

### Format 2: Chicago Public Media / RSS Wrapper

Produced by the RSS parser in `beat_book_work-main/chicago-public-media/rss_parser.py`:

```json
{
  "date": "2026-02-06",
  "entry_count": 65,
  "entries": [
    {
      "title": "Article headline",
      "link": "https://...",
      "published": "2026-02-06T14:30:12.711-06:00",
      "published_parsed": "2026-02-06T20:30:12",
      "summary": "<p>HTML article content...</p>",
      "author": "Reporter Name",
      "id": "https://...",
      "tags": ["tag1", "tag2"]
    }
  ],
  "last_updated": "2026-02-06T21:00:00"
}
```

The `summary` field is automatically unescaped and stripped of HTML tags during normalization.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web server** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | Async HTTP + WebSocket server |
| **Embeddings** | [OpenAI API](https://platform.openai.com/docs/guides/embeddings) (`text-embedding-3-small`) | Convert article text to 1536-d vectors |
| **Dimensionality reduction** | [UMAP](https://umap-learn.readthedocs.io/) | Project embeddings to lower dimensions for clustering |
| **Clustering** | [HDBSCAN](https://hdbscan.readthedocs.io/) | Density-based topic discovery at two granularities |
| **Topic labeling** | [OpenAI API](https://platform.openai.com/docs/guides/chat) (`gpt-5-mini`) | Generate human-readable labels for each cluster |
| **Agent** | [Anthropic API](https://docs.anthropic.com/) (`claude-sonnet-4-6`) | Tool-using agent for interview and beat book generation |
| **Numerical** | [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [scikit-learn](https://scikit-learn.org/) | Vector math, distance calculations, preprocessing |
| **Frontend** | Vanilla HTML/CSS/JS | No-framework single-page app |

---

## Setup & Running

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys) (for embeddings and topic labeling)
- An [Anthropic API key](https://console.anthropic.com/) (for the Claude agent)

### Install

```bash
pip install -r requirements.txt
pip install anthropic python-multipart
```

### Configure

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Run

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Project Structure

```
Beat Book Topic Categories/
├── app.py                  # FastAPI server — routes, upload handling, WebSocket, normalization
├── pipeline.py             # NLP pipeline — embedding, UMAP, HDBSCAN, LLM labeling
├── agent.py                # Claude agent — tool definitions, system prompt, agent loop
├── assign_topics.py        # Standalone CLI script for topic assignment (no web server)
├── inspect_topics.py       # Utility to print topic distributions from a labeled JSON
├── requirements.txt        # Python dependencies
├── static/
│   ├── index.html          # Single-page app markup
│   ├── app.js              # Frontend logic — upload, SSE, WebSocket, chat UI
│   └── style.css           # Dark-themed styles
├── source-stories/         # Input story files (upload via the web UI)
├── source-stories-alt/     # Alternate story sources with a cleaning script
│   └── clean.py            # Converts RSS-format JSONs to the standard schema
├── output/                 # Generated beat books (Markdown files)
└── .cache/                 # Embedding cache (auto-generated)
```
