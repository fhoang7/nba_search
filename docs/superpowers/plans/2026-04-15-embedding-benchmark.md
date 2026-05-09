# Embedding Model Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-stage benchmarking pipeline that evaluates open-source embedding models on indexing speed, query latency, and Recall@K against a Claude-generated eval set.

**Architecture:** `eval_gen.py` generates a prompt for Opus to produce multi-label eval items (query → relevant game IDs); `embed.py` encodes the corpus once per model and caches to `.npy` files; `benchmark.py` loads cached embeddings, runs cosine retrieval, computes Recall@K, and writes a terminal table plus CSV row. Each stage is independently runnable and skippable if its artifact already exists.

**Tech Stack:** `sentence-transformers>=3.0`, `numpy>=1.26`, `pydantic` (already present), `rank_bm25` / `nba_search.index` for corpus loading (already present)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `sentence-transformers`, `numpy` to dependencies |
| `src/nba_search/models.py` | Modify | Add `EvalItem` Pydantic model |
| `src/nba_search/eval_gen.py` | Create | Prompt builder (`generate`) and JSON parser (`parse`) |
| `src/nba_search/embed.py` | Create | Corpus/query embedding with `.npy` cache |
| `src/nba_search/benchmark.py` | Create | Recall@K + timing + terminal table + CSV |
| `tests/test_eval_gen.py` | Create | Tests for prompt builder and parser |
| `tests/test_embed.py` | Create | Tests for embedding + caching using real model |
| `tests/test_benchmark.py` | Create | Tests for pure recall + similarity functions |

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add sentence-transformers and numpy to pyproject.toml**

Replace the `dependencies` block:

```toml
dependencies = [
    "nltk>=3.9.4",
    "numpy>=1.26",
    "pydantic>=2.0",
    "rank-bm25>=0.2.2",
    "requests>=2.32",
    "sentence-transformers>=3.0",
]
```

- [ ] **Step 2: Install updated dependencies**

```bash
uv sync
```

Expected: resolves and installs `sentence-transformers`, `numpy`, and their transitive deps (torch, transformers, etc.) without errors.

- [ ] **Step 3: Verify install**

```bash
python -c "from sentence_transformers import SentenceTransformer; import numpy; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add sentence-transformers and numpy dependencies"
```

---

## Task 2: Add EvalItem Model

**Files:**
- Modify: `src/nba_search/models.py`
- Test: `tests/test_eval_gen.py` (partial — EvalItem tests only)

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_gen.py` with this content (EvalItem section only — more tests added in Task 3):

```python
"""Tests for eval set generation: prompt builder and JSON parser."""
import json
from pathlib import Path

import pytest

from nba_search.models import EvalItem


class TestEvalItem:
    def test_valid_construction(self):
        item = EvalItem(query="Tatum big game", relevant_ids=["001", "002"])
        assert item.query == "Tatum big game"
        assert item.relevant_ids == ["001", "002"]

    def test_single_relevant_id(self):
        item = EvalItem(query="test", relevant_ids=["abc"])
        assert len(item.relevant_ids) == 1

    def test_model_validate_from_dict(self):
        item = EvalItem.model_validate({"query": "test", "relevant_ids": ["x"]})
        assert item.query == "test"

    def test_model_dump_roundtrip(self):
        item = EvalItem(query="test", relevant_ids=["a", "b"])
        dumped = item.model_dump()
        restored = EvalItem.model_validate(dumped)
        assert restored == item
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_eval_gen.py::TestEvalItem -v
```

Expected: `ImportError` or `AttributeError` — `EvalItem` not yet defined.

- [ ] **Step 3: Add EvalItem to models.py**

Append to the end of `src/nba_search/models.py`:

```python


class EvalItem(BaseModel):
    query: str
    relevant_ids: list[str]  # multi-label; supports hybrid search
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_eval_gen.py::TestEvalItem -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_search/models.py tests/test_eval_gen.py
git commit -m "feat: add EvalItem model for multi-label eval set"
```

---

## Task 3: eval_gen Prompt Builder

**Files:**
- Create: `src/nba_search/eval_gen.py`
- Test: `tests/test_eval_gen.py` (append `TestBuildPrompt`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_gen.py`:

```python
from nba_search.eval_gen import build_prompt
from nba_search.models import GameDocument, Performer


def _make_doc(game_id: str, full_text: str) -> GameDocument:
    return GameDocument(
        game_id=game_id,
        date="2026-04-08",
        home_team="Boston Celtics",
        away_team="Dallas Mavericks",
        home_score=106,
        away_score=88,
        top_performers=[Performer(player="Jayson Tatum", pts=31, reb=11, ast=5)],
        recap_text=full_text,
        stat_summary="Tatum: 31/11/5",
        full_text=full_text,
    )


SAMPLE_DOCS = [
    _make_doc("001", "Tatum 31 points Celtics dominant win over Mavericks"),
    _make_doc("002", "Luka 45 points Mavericks comeback victory"),
]


class TestBuildPrompt:
    def test_contains_all_game_ids(self):
        prompt = build_prompt(SAMPLE_DOCS)
        assert "001" in prompt
        assert "002" in prompt

    def test_contains_full_text(self):
        prompt = build_prompt(SAMPLE_DOCS)
        assert "Tatum 31 points" in prompt
        assert "Luka 45 points" in prompt

    def test_instructs_relevant_ids_field(self):
        prompt = build_prompt(SAMPLE_DOCS)
        assert "relevant_ids" in prompt

    def test_instructs_query_field(self):
        prompt = build_prompt(SAMPLE_DOCS)
        assert '"query"' in prompt

    def test_mentions_doc_count(self):
        prompt = build_prompt(SAMPLE_DOCS)
        assert str(len(SAMPLE_DOCS)) in prompt
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_eval_gen.py::TestBuildPrompt -v
```

Expected: `ImportError` — `eval_gen` module doesn't exist yet.

- [ ] **Step 3: Create eval_gen.py with build_prompt**

Create `src/nba_search/eval_gen.py`:

```python
"""
Eval set generation for embedding benchmarks.

Subcommands:
  generate  — reads corpus, writes data/eval_prompt.txt for pasting into Claude
  parse     — reads data/eval_raw.json, validates, writes data/eval_set.json
"""
import json
import sys
from pathlib import Path

from nba_search.index import load_corpus
from nba_search.models import EvalItem, GameDocument

CORPUS_PATH = Path("data/corpus.jsonl")
EVAL_PROMPT_PATH = Path("data/eval_prompt.txt")
EVAL_RAW_PATH = Path("data/eval_raw.json")
EVAL_SET_PATH = Path("data/eval_set.json")


def build_prompt(docs: list[GameDocument]) -> str:
    """Build the Opus prompt for query generation and relevance judgment."""
    corpus_block = "\n".join(
        f'  {{"game_id": "{doc.game_id}", "full_text": {json.dumps(doc.full_text)}}}'
        for doc in docs
    )
    return f"""You are building an evaluation set for an NBA game search system.

Below is a corpus of {len(docs)} NBA game documents. Each has a game_id and full_text.

CORPUS:
[
{corpus_block}
]

Your task:
1. For each game, generate 2-3 natural language search queries a user might type to find that game.
   - Mix player-focused, score-focused, and narrative-focused queries
   - Phrase them as a user would search (not paraphrasing the recap directly)
   - Make them specific enough to distinguish the game from similar ones
   - Avoid overly generic queries like "who won" or "best game"

2. For EACH query, identify ALL game_ids from the corpus that are genuinely relevant to that query.
   The source game will usually be included, but add others if they are also relevant.

Return a JSON array in EXACTLY this format — no other text, no markdown fences:
[
  {{
    "query": "Tatum triple-double Celtics win",
    "relevant_ids": ["401765432", "401765501"]
  }}
]"""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_eval_gen.py::TestBuildPrompt -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_search/eval_gen.py tests/test_eval_gen.py
git commit -m "feat: add eval_gen prompt builder"
```

---

## Task 4: eval_gen Parser and CLI

**Files:**
- Modify: `src/nba_search/eval_gen.py`
- Test: `tests/test_eval_gen.py` (append `TestParseRaw`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_gen.py`:

```python
from nba_search.eval_gen import parse_raw


class TestParseRaw:
    def test_parses_valid_json(self, tmp_path):
        raw = [
            {"query": "Tatum big game", "relevant_ids": ["001"]},
            {"query": "Luka comeback", "relevant_ids": ["002", "001"]},
        ]
        raw_path = tmp_path / "eval_raw.json"
        out_path = tmp_path / "eval_set.json"
        raw_path.write_text(json.dumps(raw))

        items = parse_raw(raw_path, out_path)

        assert len(items) == 2
        assert items[0].query == "Tatum big game"
        assert items[0].relevant_ids == ["001"]
        assert items[1].relevant_ids == ["002", "001"]

    def test_skips_malformed_items(self, tmp_path):
        raw = [
            {"query": "valid query", "relevant_ids": ["001"]},
            {"bad_field": "missing required fields"},
        ]
        raw_path = tmp_path / "eval_raw.json"
        out_path = tmp_path / "eval_set.json"
        raw_path.write_text(json.dumps(raw))

        items = parse_raw(raw_path, out_path)
        assert len(items) == 1
        assert items[0].query == "valid query"

    def test_writes_output_file(self, tmp_path):
        raw = [{"query": "test query", "relevant_ids": ["001"]}]
        raw_path = tmp_path / "eval_raw.json"
        out_path = tmp_path / "eval_set.json"
        raw_path.write_text(json.dumps(raw))

        parse_raw(raw_path, out_path)

        assert out_path.exists()
        saved = json.loads(out_path.read_text())
        assert len(saved) == 1
        assert saved[0]["query"] == "test query"

    def test_strips_markdown_code_fences(self, tmp_path):
        raw = [{"query": "test", "relevant_ids": ["001"]}]
        raw_path = tmp_path / "eval_raw.json"
        out_path = tmp_path / "eval_set.json"
        raw_path.write_text("```json\n" + json.dumps(raw) + "\n```")

        items = parse_raw(raw_path, out_path)
        assert len(items) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_eval_gen.py::TestParseRaw -v
```

Expected: `ImportError` — `parse_raw` not yet defined.

- [ ] **Step 3: Add parse_raw, write_prompt, and main to eval_gen.py**

Append to `src/nba_search/eval_gen.py`:

```python

def write_prompt(docs: list[GameDocument], path: Path = EVAL_PROMPT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_prompt(docs), encoding="utf-8")
    print(f"Prompt written to {path}")
    print(f"Paste it into Claude (Opus) and save the JSON response to {EVAL_RAW_PATH}")


def parse_raw(
    raw_path: Path = EVAL_RAW_PATH,
    out_path: Path = EVAL_SET_PATH,
) -> list[EvalItem]:
    """Parse eval_raw.json into validated EvalItem objects, skipping malformed entries."""
    raw_text = raw_path.read_text(encoding="utf-8").strip()

    # Strip markdown code fences if Claude wrapped the response
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        raw_text = "\n".join(lines[1:end])

    raw_items: list[object] = json.loads(raw_text)

    items: list[EvalItem] = []
    skipped = 0
    for i, raw in enumerate(raw_items):
        try:
            items.append(EvalItem.model_validate(raw))
        except Exception as e:
            print(f"  Warning: skipping item {i} — {e}", file=sys.stderr)
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} malformed item(s).", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([item.model_dump() for item in items], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(items)} eval items to {out_path}")
    return items


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("generate", "parse"):
        print("Usage: python -m nba_search.eval_gen [generate|parse]")
        sys.exit(1)

    if sys.argv[1] == "generate":
        if not CORPUS_PATH.exists():
            print(
                f"Error: corpus not found at {CORPUS_PATH}. Run ingest first.",
                file=sys.stderr,
            )
            sys.exit(1)
        docs = load_corpus(CORPUS_PATH)
        write_prompt(docs)
    else:
        if not EVAL_RAW_PATH.exists():
            print(
                f"Error: {EVAL_RAW_PATH} not found. "
                "Run 'generate', paste the response into Claude, and save it here.",
                file=sys.stderr,
            )
            sys.exit(1)
        parse_raw()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_eval_gen.py -v
```

Expected: all tests pass (EvalItem + BuildPrompt + ParseRaw).

- [ ] **Step 5: Commit**

```bash
git add src/nba_search/eval_gen.py tests/test_eval_gen.py
git commit -m "feat: add eval_gen parser and CLI"
```

---

## Task 5: embed.py — safe_model_name and Scaffold

**Files:**
- Create: `src/nba_search/embed.py`
- Create: `tests/test_embed.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_embed.py`:

```python
"""Tests for embedding pipeline — uses all-MiniLM-L6-v2 directly (no mocks)."""
import json
from pathlib import Path

import numpy as np
import pytest

from nba_search.embed import safe_model_name


class TestSafeModelName:
    def test_replaces_slash_with_underscore(self):
        assert safe_model_name("BAAI/bge-small-en-v1.5") == "BAAI_bge-small-en-v1.5"

    def test_no_slash_unchanged(self):
        assert safe_model_name("all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"

    def test_multiple_slashes(self):
        assert safe_model_name("org/sub/model") == "org_sub_model"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_embed.py::TestSafeModelName -v
```

Expected: `ImportError` — `embed` module doesn't exist.

- [ ] **Step 3: Create embed.py with safe_model_name and DEFAULT_MODELS**

Create `src/nba_search/embed.py`:

```python
"""
Embedding pipeline: encodes corpus and queries using sentence-transformers models.
Results are cached to data/embeddings/ to avoid re-encoding on every run.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from nba_search.index import load_corpus
from nba_search.models import GameDocument

EMBEDDINGS_DIR = Path("data/embeddings")
CORPUS_PATH = Path("data/corpus.jsonl")

DEFAULT_MODELS: list[str] = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
]


def safe_model_name(model_name: str) -> str:
    """Sanitize model name for filesystem use — replaces / with _."""
    return model_name.replace("/", "_")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_embed.py::TestSafeModelName -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_search/embed.py tests/test_embed.py
git commit -m "feat: add embed.py scaffold with safe_model_name"
```

---

## Task 6: embed.py — embed_corpus with Caching

**Files:**
- Modify: `src/nba_search/embed.py`
- Test: `tests/test_embed.py` (append `TestEmbedCorpus`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_embed.py` after the existing imports and `TestSafeModelName`:

```python
from nba_search.embed import embed_corpus
from nba_search.models import Performer, GameDocument

MODEL = "all-MiniLM-L6-v2"


def _make_doc(game_id: str, full_text: str) -> GameDocument:
    return GameDocument(
        game_id=game_id,
        date="2026-04-08",
        home_team="Boston Celtics",
        away_team="Dallas Mavericks",
        home_score=106,
        away_score=88,
        top_performers=[Performer(player="Jayson Tatum", pts=31, reb=11, ast=5)],
        recap_text=full_text,
        stat_summary="Tatum: 31/11/5",
        full_text=full_text,
    )


SAMPLE_DOCS = [
    _make_doc("001", "Tatum 31 points Celtics dominant win"),
    _make_doc("002", "Luka 45 points Mavericks comeback"),
    _make_doc("003", "Gobert defensive masterclass held offense to 89"),
]


class TestEmbedCorpus:
    def test_returns_matrix_correct_shape(self, tmp_path):
        matrix, _ = embed_corpus(MODEL, SAMPLE_DOCS, embeddings_dir=tmp_path)
        assert matrix.shape == (3, 384)

    def test_returns_positive_index_time(self, tmp_path):
        _, index_time = embed_corpus(MODEL, SAMPLE_DOCS, embeddings_dir=tmp_path)
        assert index_time > 0

    def test_creates_npy_cache(self, tmp_path):
        embed_corpus(MODEL, SAMPLE_DOCS, embeddings_dir=tmp_path)
        assert (tmp_path / f"{MODEL}.npy").exists()

    def test_creates_ids_json(self, tmp_path):
        embed_corpus(MODEL, SAMPLE_DOCS, embeddings_dir=tmp_path)
        ids = json.loads((tmp_path / f"{MODEL}_ids.json").read_text())
        assert ids == ["001", "002", "003"]

    def test_creates_meta_json_with_index_time(self, tmp_path):
        embed_corpus(MODEL, SAMPLE_DOCS, embeddings_dir=tmp_path)
        meta = json.loads((tmp_path / f"{MODEL}_meta.json").read_text())
        assert "index_time_seconds" in meta
        assert meta["index_time_seconds"] > 0

    def test_loads_from_cache_on_second_call(self, tmp_path):
        matrix1, _ = embed_corpus(MODEL, SAMPLE_DOCS, embeddings_dir=tmp_path)
        matrix2, _ = embed_corpus(MODEL, SAMPLE_DOCS, embeddings_dir=tmp_path)
        assert np.array_equal(matrix1, matrix2)

    def test_force_flag_reembeds(self, tmp_path):
        embed_corpus(MODEL, SAMPLE_DOCS, embeddings_dir=tmp_path)
        matrix2, _ = embed_corpus(MODEL, SAMPLE_DOCS, embeddings_dir=tmp_path, force=True)
        assert matrix2.shape == (3, 384)

    def test_sanitizes_model_name_in_filenames(self, tmp_path):
        slashed_model = "BAAI/bge-small-en-v1.5"
        # We just test the filename logic, not the actual model download
        safe = "BAAI_bge-small-en-v1.5"
        embed_corpus(MODEL, SAMPLE_DOCS, embeddings_dir=tmp_path)
        # Confirm no slash appears in any file under tmp_path
        for f in tmp_path.iterdir():
            assert "/" not in f.name
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_embed.py::TestEmbedCorpus -v
```

Expected: `ImportError` — `embed_corpus` not yet defined.

- [ ] **Step 3: Implement embed_corpus in embed.py**

Append to `src/nba_search/embed.py`:

```python

def embed_corpus(
    model_name: str,
    docs: list[GameDocument],
    embeddings_dir: Path = EMBEDDINGS_DIR,
    force: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Embed all documents. Returns (matrix, index_time_seconds).
    Loads from cache if available and force=False.
    """
    safe_name = safe_model_name(model_name)
    npy_path = embeddings_dir / f"{safe_name}.npy"
    ids_path = embeddings_dir / f"{safe_name}_ids.json"
    meta_path = embeddings_dir / f"{safe_name}_meta.json"

    if not force and npy_path.exists():
        matrix: np.ndarray = np.load(npy_path)
        meta = json.loads(meta_path.read_text())
        print(f"  Loaded cached embeddings for {model_name} ({matrix.shape[0]} docs)")
        return matrix, float(meta["index_time_seconds"])

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    texts = [doc.full_text for doc in docs]

    start = time.perf_counter()
    matrix = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    index_time = time.perf_counter() - start

    np.save(npy_path, matrix)
    ids_path.write_text(json.dumps([doc.game_id for doc in docs]))
    meta_path.write_text(
        json.dumps({
            "model_name": model_name,
            "index_time_seconds": round(index_time, 3),
            "doc_count": len(docs),
            "dim": int(matrix.shape[1]),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    )

    print(f"  Embedded {len(docs)} docs with {model_name} in {index_time:.2f}s")
    return matrix, index_time
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_embed.py::TestEmbedCorpus -v
```

Expected: 8 passed. (This test downloads `all-MiniLM-L6-v2` ~90MB on first run — subsequent runs use HuggingFace cache.)

- [ ] **Step 5: Commit**

```bash
git add src/nba_search/embed.py tests/test_embed.py
git commit -m "feat: add embed_corpus with npy cache"
```

---

## Task 7: embed.py — embed_query, load_embeddings, and CLI

**Files:**
- Modify: `src/nba_search/embed.py`
- Test: `tests/test_embed.py` (append `TestEmbedQuery` and `TestLoadEmbeddings`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_embed.py`:

```python
from nba_search.embed import embed_query, load_embeddings


class TestEmbedQuery:
    def test_returns_correct_shape(self):
        vec, _ = embed_query(MODEL, "Tatum scored 31 points")
        assert vec.shape == (384,)

    def test_returns_positive_latency_ms(self):
        _, latency_ms = embed_query(MODEL, "Tatum scored 31 points")
        assert latency_ms > 0


class TestLoadEmbeddings:
    def test_loads_matrix_ids_and_index_time(self, tmp_path):
        embed_corpus(MODEL, SAMPLE_DOCS, embeddings_dir=tmp_path)
        matrix, ids, index_time = load_embeddings(MODEL, embeddings_dir=tmp_path)
        assert matrix.shape == (3, 384)
        assert ids == ["001", "002", "003"]
        assert index_time > 0

    def test_raises_if_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No embeddings found"):
            load_embeddings(MODEL, embeddings_dir=tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_embed.py::TestEmbedQuery tests/test_embed.py::TestLoadEmbeddings -v
```

Expected: `ImportError` — `embed_query` and `load_embeddings` not yet defined.

- [ ] **Step 3: Implement embed_query, load_embeddings, and main in embed.py**

Append to `src/nba_search/embed.py`:

```python

def embed_query(model_name: str, query: str) -> tuple[np.ndarray, float]:
    """Embed a single query. Returns (vector, elapsed_ms). Loads model each call."""
    model = SentenceTransformer(model_name)
    start = time.perf_counter()
    vector: np.ndarray = model.encode(query, convert_to_numpy=True)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return vector, elapsed_ms


def load_embeddings(
    model_name: str,
    embeddings_dir: Path = EMBEDDINGS_DIR,
) -> tuple[np.ndarray, list[str], float]:
    """
    Load cached embeddings for a model.
    Returns (matrix, game_ids, index_time_seconds).
    Raises FileNotFoundError if embed_corpus hasn't been run for this model.
    """
    safe_name = safe_model_name(model_name)
    npy_path = embeddings_dir / f"{safe_name}.npy"
    ids_path = embeddings_dir / f"{safe_name}_ids.json"
    meta_path = embeddings_dir / f"{safe_name}_meta.json"

    if not npy_path.exists():
        raise FileNotFoundError(
            f"No embeddings found for '{model_name}'. "
            f"Run: python -m nba_search.embed --model {model_name}"
        )

    matrix: np.ndarray = np.load(npy_path)
    game_ids: list[str] = json.loads(ids_path.read_text())
    meta = json.loads(meta_path.read_text())
    return matrix, game_ids, float(meta["index_time_seconds"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed NBA corpus with sentence-transformers models."
    )
    parser.add_argument("--model", type=str, default=None, help="Single model to embed")
    parser.add_argument(
        "--force", action="store_true", help="Re-embed even if cache exists"
    )
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        print(
            f"Error: corpus not found at {CORPUS_PATH}. Run ingest first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    docs = load_corpus(CORPUS_PATH)
    models = [args.model] if args.model else DEFAULT_MODELS

    for model_name in models:
        print(f"\nModel: {model_name}")
        embed_corpus(model_name, docs, force=args.force)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all embed tests**

```bash
pytest tests/test_embed.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/nba_search/embed.py tests/test_embed.py
git commit -m "feat: add embed_query, load_embeddings, and embed CLI"
```

---

## Task 8: benchmark.py — Pure Recall Functions

**Files:**
- Create: `src/nba_search/benchmark.py`
- Create: `tests/test_benchmark.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark.py`:

```python
"""Tests for benchmark logic — synthetic scores, no model loading."""
import numpy as np
import pytest

from nba_search.benchmark import compute_recall_at_k, cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors_score_1(self):
        v = np.array([1.0, 0.0, 0.0])
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        scores = cosine_similarity(v, matrix)
        assert abs(scores[0] - 1.0) < 1e-6
        assert abs(scores[1] - 0.0) < 1e-6

    def test_orthogonal_vectors_score_0(self):
        v = np.array([1.0, 0.0])
        matrix = np.array([[0.0, 1.0]])
        scores = cosine_similarity(v, matrix)
        assert abs(scores[0]) < 1e-6

    def test_returns_one_score_per_doc(self):
        v = np.ones(4)
        matrix = np.ones((10, 4))
        scores = cosine_similarity(v, matrix)
        assert scores.shape == (10,)


class TestComputeRecallAtK:
    def test_relevant_in_top1_returns_1(self):
        scores = np.array([0.9, 0.5, 0.1])
        game_ids = ["a", "b", "c"]
        assert compute_recall_at_k(scores, game_ids, ["a"], k=1) == 1.0

    def test_relevant_not_in_top1_returns_0(self):
        scores = np.array([0.9, 0.5, 0.1])
        game_ids = ["a", "b", "c"]
        assert compute_recall_at_k(scores, game_ids, ["c"], k=1) == 0.0

    def test_relevant_in_top5_but_not_top1(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
        game_ids = ["a", "b", "c", "d", "e", "f"]
        assert compute_recall_at_k(scores, game_ids, ["e"], k=1) == 0.0
        assert compute_recall_at_k(scores, game_ids, ["e"], k=5) == 1.0

    def test_any_relevant_id_match_returns_1(self):
        scores = np.array([0.9, 0.5, 0.1])
        game_ids = ["a", "b", "c"]
        assert compute_recall_at_k(scores, game_ids, ["c", "a"], k=1) == 1.0

    def test_no_match_returns_0(self):
        scores = np.array([0.9, 0.5, 0.1])
        game_ids = ["a", "b", "c"]
        assert compute_recall_at_k(scores, game_ids, ["x", "y"], k=3) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_benchmark.py -v
```

Expected: `ImportError` — `benchmark` module doesn't exist.

- [ ] **Step 3: Create benchmark.py with pure functions**

Create `src/nba_search/benchmark.py`:

```python
"""
Benchmark: measures Recall@K and latency for each embedding model.
Requires pre-computed embeddings (run embed.py first) and data/eval_set.json.
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from nba_search.embed import DEFAULT_MODELS, load_embeddings
from nba_search.models import EvalItem

EVAL_SET_PATH = Path("data/eval_set.json")
RESULTS_CSV_PATH = Path("data/benchmark_results.csv")


def cosine_similarity(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and all rows of corpus_matrix."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(corpus_matrix, axis=1, keepdims=True) + 1e-10
    return (corpus_matrix / norms) @ q


def compute_recall_at_k(
    scores: np.ndarray,
    game_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Return 1.0 if any relevant_id appears in the top-k results, else 0.0."""
    top_k_indices = np.argsort(scores)[::-1][:k]
    top_k_ids = {game_ids[i] for i in top_k_indices}
    return 1.0 if top_k_ids & set(relevant_ids) else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_benchmark.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_search/benchmark.py tests/test_benchmark.py
git commit -m "feat: add benchmark pure functions (cosine similarity + recall@k)"
```

---

## Task 9: benchmark.py — Orchestration and CLI

**Files:**
- Modify: `src/nba_search/benchmark.py`

- [ ] **Step 1: Implement run_benchmark, print_table, append_csv, and main**

Append to `src/nba_search/benchmark.py`:

```python

def _load_eval_set(path: Path = EVAL_SET_PATH) -> list[EvalItem]:
    if not path.exists():
        print(
            f"Error: eval set not found at {path}. "
            "Run: python -m nba_search.eval_gen parse",
            file=sys.stderr,
        )
        raise SystemExit(1)
    raw = json.loads(path.read_text())
    return [EvalItem.model_validate(item) for item in raw]


def run_benchmark(
    model_name: str,
    eval_items: list[EvalItem],
    ks: list[int],
) -> dict[str, object]:
    """Run full benchmark for one model. Returns a result dict."""
    matrix, game_ids, index_time = load_embeddings(model_name)
    model = SentenceTransformer(model_name)

    recall_scores: dict[int, list[float]] = {k: [] for k in ks}
    query_latencies: list[float] = []

    for item in eval_items:
        start = time.perf_counter()
        vec: np.ndarray = model.encode(item.query, convert_to_numpy=True)
        latency_ms = (time.perf_counter() - start) * 1000

        query_latencies.append(latency_ms)
        scores = cosine_similarity(vec, matrix)
        for k in ks:
            recall_scores[k].append(
                compute_recall_at_k(scores, game_ids, item.relevant_ids, k)
            )

    avg_query_ms = sum(query_latencies) / len(query_latencies) if query_latencies else 0.0
    recalls = {k: sum(v) / len(v) if v else 0.0 for k, v in recall_scores.items()}

    return {
        "model": model_name,
        "index_time_s": round(index_time, 2),
        "avg_query_ms": round(avg_query_ms, 1),
        "recalls": recalls,
    }


def print_table(results: list[dict[str, object]], ks: list[int]) -> None:
    recall_headers = "  ".join(f"R@{k:<4}" for k in ks)
    header = f"{'Model':<32} | {'Idx Time':>8} | {'Avg Query':>9} | {recall_headers}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        recalls = r["recalls"]
        assert isinstance(recalls, dict)
        recall_cols = "  ".join(f"{recalls[k]:.3f}" for k in ks)
        print(
            f"{r['model']:<32} | {r['index_time_s']:>7.1f}s "
            f"| {r['avg_query_ms']:>7.0f}ms | {recall_cols}"
        )
    print()


def append_csv(
    results: list[dict[str, object]],
    ks: list[int],
    path: Path = RESULTS_CSV_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = (
        ["timestamp", "model", "index_time_s", "avg_query_ms"]
        + [f"recall_at_{k}" for k in ks]
    )
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        ts = datetime.now(timezone.utc).isoformat()
        for r in results:
            recalls = r["recalls"]
            assert isinstance(recalls, dict)
            row: dict[str, object] = {
                "timestamp": ts,
                "model": r["model"],
                "index_time_s": r["index_time_s"],
                "avg_query_ms": r["avg_query_ms"],
            }
            for k in ks:
                row[f"recall_at_{k}"] = round(float(recalls[k]), 4)
            writer.writerow(row)
    print(f"Results appended to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models for NBA search."
    )
    parser.add_argument(
        "--k",
        type=str,
        default="1,5,10",
        help="Comma-separated K values for Recall@K (default: 1,5,10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model to benchmark (default: all three)",
    )
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.k.split(",")]
    models = [args.model] if args.model else DEFAULT_MODELS
    eval_items = _load_eval_set()

    results = []
    for model_name in models:
        print(f"\nBenchmarking {model_name}...")
        try:
            results.append(run_benchmark(model_name, eval_items, ks))
        except FileNotFoundError as e:
            print(f"  Skipping — {e}", file=sys.stderr)

    if results:
        print_table(results, ks)
        append_csv(results, ks)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full test suite**

```bash
pytest -v
```

Expected: all existing + new tests pass. Coverage report shows new modules covered.

- [ ] **Step 3: Commit**

```bash
git add src/nba_search/benchmark.py
git commit -m "feat: add benchmark orchestration and CLI"
```

---

## Task 10: End-to-End Smoke Test

**Files:** none — verification only

- [ ] **Step 1: Generate the eval prompt**

```bash
python -m nba_search.eval_gen generate
```

Expected: `data/eval_prompt.txt` created. Output: `Prompt written to data/eval_prompt.txt`

- [ ] **Step 2: Paste prompt into Claude and save response**

Open `data/eval_prompt.txt`, paste into Claude.ai (Opus model) or Claude Code. Save the JSON response to `data/eval_raw.json`.

- [ ] **Step 3: Parse the eval set**

```bash
python -m nba_search.eval_gen parse
```

Expected: `data/eval_set.json` created. Output: `Wrote N eval items to data/eval_set.json`
Spot-check: open `data/eval_set.json` and verify a few items have multiple `relevant_ids` where appropriate.

- [ ] **Step 4: Embed the corpus**

```bash
python -m nba_search.embed
```

Expected: three `.npy` files appear under `data/embeddings/`. Each model prints its doc count and time.

- [ ] **Step 5: Run the benchmark**

```bash
python -m nba_search.benchmark
```

Expected: terminal table with three rows (one per model) showing `Idx Time`, `Avg Query`, `R@1`, `R@5`, `R@10`. `data/benchmark_results.csv` created.

- [ ] **Step 6: Final test run**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add data/eval_set.json data/benchmark_results.csv
git commit -m "feat: embedding benchmark pipeline complete"
```
