"""
assign_topics.py
----------------
Reads source stories from a JSON file, generates embeddings via the OpenAI API,
clusters them with UMAP + HDBSCAN at two granularities (broad and specific),
labels each cluster with an LLM, and writes the results back to a new JSON file
with a `topics` field added to each story.

Usage:
    export OPENAI_API_KEY=sk-...
    python3 assign_topics.py

Output:
    source-stories/chicago_public_media_with_topics.json
"""

import json
import os
import sys
import hashlib
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Load .env file if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── OpenAI ───────────────────────────────────────────────────────────────────
from openai import OpenAI

# ── Dimensionality reduction + clustering ────────────────────────────────────
import umap
import hdbscan
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE  = Path("source-stories/chicago_public_media.json")
OUTPUT_FILE = Path("source-stories/chicago_public_media_with_topics.json")
CACHE_FILE  = Path(".embeddings_cache.pkl")   # speeds up reruns

EMBED_MODEL = "text-embedding-3-small"
LABEL_MODEL = "gpt-5-mini"

# How many representative stories to show the LLM per cluster for labeling
SAMPLE_SIZE_FOR_LABEL = 8

# All UMAP and HDBSCAN parameters are computed dynamically in main() based on
# corpus size so the pipeline works well from ~100 to ~1000+ stories.

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def story_to_text(story: dict) -> str:
    """Return the title + section + first ~400 words of content for embedding."""
    title   = story.get("title", "")
    content = story.get("content", "")

    # Pull the section label out of the content header (e.g. "Section: Sports")
    section = ""
    for line in content.splitlines()[:10]:
        if line.strip().lower().startswith("section:"):
            section = line.strip()
            break

    words   = content.split()
    snippet = " ".join(words[:400])
    parts   = [p for p in [title, section, snippet] if p]
    return "\n\n".join(parts)


def embed_batch(client: OpenAI, texts: list[str], model: str) -> np.ndarray:
    """Embed a list of texts in batches of 100 and return a (N, D) float32 array."""
    all_vectors = []
    batch_size  = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        chunk  = texts[i : i + batch_size]
        resp   = client.embeddings.create(input=chunk, model=model)
        vecs   = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        all_vectors.extend(vecs)
    return np.array(all_vectors, dtype=np.float32)


def get_cache_key(texts: list[str]) -> str:
    combined = "\n---\n".join(texts[:10])  # fingerprint on first 10 docs
    return hashlib.md5((combined + EMBED_MODEL).encode()).hexdigest()


def load_or_embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Return embeddings from cache if texts match, otherwise call the API."""
    key = get_cache_key(texts)
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == key and len(cached.get("vectors", [])) == len(texts):
            print("✓ Loaded embeddings from cache.")
            return cached["vectors"]

    print(f"Generating embeddings for {len(texts)} stories via OpenAI...")
    vectors = embed_batch(client, texts, EMBED_MODEL)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"key": key, "vectors": vectors}, f)
    print("✓ Embeddings saved to cache.")
    return vectors


def umap_params(n: int) -> dict:
    """Return UMAP parameters scaled to corpus size."""
    # n_components: more dims helps larger corpora separate; cap at 15
    n_components = min(15, max(5, n // 40))
    # n_neighbors: controls local vs global structure; scale with sqrt(n)
    n_neighbors  = min(30, max(5, int(n ** 0.55)))
    return {"n_components": n_components, "n_neighbors": n_neighbors}


def cluster_sizes(n: int) -> tuple[int, int]:
    """Return (broad_min, specific_min) cluster sizes scaled to corpus size."""
    broad    = max(4, n // 25)   # ~25 broad clusters per 1000 docs  → e.g. 200→8,  1000→40
    specific = max(2, n // 60)   # ~17 specific clusters per 1000 docs → e.g. 200→3,  1000→17
    return broad, specific


def reduce_dimensions(vectors: np.ndarray) -> np.ndarray:
    """UMAP dimensionality reduction with corpus-size-scaled parameters."""
    n      = len(vectors)
    params = umap_params(n)
    print(f"Running UMAP (n_components={params['n_components']}, n_neighbors={params['n_neighbors']}) on {n} docs...")
    reducer = umap.UMAP(
        n_components=params["n_components"],
        n_neighbors=params["n_neighbors"],
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(vectors)
    print("✓ UMAP complete.")
    return reduced


def cluster(reduced: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Run HDBSCAN and return integer cluster labels (-1 = noise/outlier)."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"  → {n_clusters} clusters, {n_noise} noise points (min_cluster_size={min_cluster_size})")
    return labels, clusterer


def assign_outliers(reduced: np.ndarray, labels: np.ndarray, clusterer) -> np.ndarray:
    """Assign noise points (-1) to their nearest cluster centroid."""
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels

    labels          = labels.copy()
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    if not unique_clusters:
        return labels

    cluster_means = np.stack([
        reduced[labels == c].mean(axis=0) for c in unique_clusters
    ])

    for idx in np.where(noise_mask)[0]:
        dists = np.linalg.norm(cluster_means - reduced[idx], axis=1)
        labels[idx] = unique_clusters[int(dists.argmin())]

    return labels


def label_cluster(
    client: OpenAI,
    stories: list[dict],
    cluster_indices: list[int],
    reduced: np.ndarray,
) -> str:
    """Ask the LLM to give a short topic label for a cluster of stories.

    Picks the stories closest to the cluster centroid so the LLM sees the most
    representative examples rather than arbitrary list order.
    """
    # Sort indices by distance to cluster centroid so we feed the best examples
    cluster_vecs = reduced[cluster_indices]
    centroid     = cluster_vecs.mean(axis=0)
    dists        = np.linalg.norm(cluster_vecs - centroid, axis=1)
    order        = np.argsort(dists)
    sampled      = [cluster_indices[i] for i in order[:SAMPLE_SIZE_FOR_LABEL]]

    snippets = []
    for i in sampled:
        s = stories[i]
        # Use just the title + a tight 30-word excerpt for clarity
        words   = s.get("content", "").split()
        excerpt = " ".join(words[10:40])  # skip the repeated header boilerplate
        snippets.append(f"• {s['title']} — {excerpt}")

    prompt = (
        "You are labeling clusters of news articles from Chicago Public Media.\n"
        "Below are the most representative headlines and excerpts from one cluster.\n\n"
        + "\n".join(snippets)
        + "\n\nReturn ONLY a concise topic label (2–5 words) describing the SUBJECT MATTER "
        "these articles share. Focus on WHAT happens, not WHERE — avoid labels like "
        "'Chicago news', 'local community news', or 'Illinois news' unless the "
        "geography itself is the distinguishing feature (e.g. 'Lake Michigan environment'). "
        "Good labels: 'High School Basketball', 'City Budget Disputes', "
        "'Immigration Policy', 'Crime and Sentencing', 'City Council', 'Transit'."
    )

    resp = client.chat.completions.create(
        model=LABEL_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip().strip('"').strip("'")


def label_all_clusters(
    client: OpenAI,
    stories: list[dict],
    labels: np.ndarray,
    reduced: np.ndarray,
    level_name: str,
) -> dict[int, str]:
    """Return a mapping of cluster_id → topic string."""
    unique = sorted(c for c in np.unique(labels) if c != -1)
    print(f"\nLabeling {len(unique)} {level_name} clusters via LLM...")
    cluster_label_map = {}
    for cid in tqdm(unique, desc=f"Labeling ({level_name})"):
        indices = list(np.where(labels == cid)[0])
        cluster_label_map[cid] = label_cluster(client, stories, indices, reduced)
    return cluster_label_map


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # ── 1. Load stories ──────────────────────────────────────────────────────
    print(f"\nLoading stories from {INPUT_FILE}…")
    with open(INPUT_FILE, "r") as f:
        stories = json.load(f)
    print(f"✓ {len(stories)} stories loaded.")

    # ── 2. Embed ─────────────────────────────────────────────────────────────
    texts   = [story_to_text(s) for s in stories]
    vectors = load_or_embed(client, texts)

    # ── 3. Reduce dimensions ─────────────────────────────────────────────────
    reduced = reduce_dimensions(vectors)

    # ── 4. Cluster at two granularities (sizes scale with corpus) ─────────────
    broad_min, specific_min = cluster_sizes(len(stories))
    print(f"\nCluster sizes: broad_min={broad_min}, specific_min={specific_min}")

    print("\nClustering (broad)…")
    broad_labels, broad_clusterer = cluster(reduced, broad_min)
    broad_labels = assign_outliers(reduced, broad_labels, broad_clusterer)

    print("\nClustering (specific)…")
    spec_labels, spec_clusterer = cluster(reduced, specific_min)
    spec_labels = assign_outliers(reduced, spec_labels, spec_clusterer)

    # ── 5. LLM-label each cluster ─────────────────────────────────────────────
    broad_map = label_all_clusters(client, stories, broad_labels, reduced, "broad")
    spec_map  = label_all_clusters(client, stories, spec_labels,  reduced, "specific")

    # ── 6. Print summary ──────────────────────────────────────────────────────
    print("\n── Broad topics ──────────────────────────────────────────────")
    for cid, label in sorted(broad_map.items()):
        count = int((broad_labels == cid).sum())
        print(f"  [{count:3d} stories]  {label}")

    print("\n── Specific topics ───────────────────────────────────────────")
    for cid, label in sorted(spec_map.items()):
        count = int((spec_labels == cid).sum())
        print(f"  [{count:3d} stories]  {label}")

    # ── 7. Attach topics to each story ────────────────────────────────────────
    for i, story in enumerate(stories):
        broad_topic = broad_map.get(int(broad_labels[i]), "Uncategorized")
        spec_topic  = spec_map.get(int(spec_labels[i]),   "Uncategorized")

        # De-duplicate in case both levels produce the same label
        if spec_topic.lower() == broad_topic.lower():
            story["topics"] = [broad_topic]
        else:
            story["topics"] = [broad_topic, spec_topic]

    # ── 8. Write output ──────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(stories, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done! Output written to {OUTPUT_FILE}")
    print("  Example assignments:")
    for story in stories[:5]:
        print(f"  {story['title'][:60]:<60}  →  {story['topics']}")


if __name__ == "__main__":
    main()
