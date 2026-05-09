# NBA Search

Search NBA game recaps and box scores using keyword or semantic queries.

Data is pulled from the ESPN public API and stored locally as a JSONL corpus.

---

## What's Working

### Data Ingestion (`src/nba_search/ingest.py`)
- Fetches completed games from ESPN's scoreboard and summary endpoints
- Extracts game metadata (teams, scores, date), top 3 performers (pts/reb/ast), and recap text
- Deduplicates against existing corpus; appends new games to `data/corpus.jsonl`
- Supports `--days N` or `--start-date / --end-date` range

```bash
uv run python -m nba_search.ingest --days 7
uv run python -m nba_search.ingest --start-date 2025-01-01 --end-date 2025-04-15
```

### BM25 Keyword Search (`src/nba_search/index.py`)
- Builds a BM25Okapi index over the corpus with custom tokenization
- Preserves proper nouns (player/team names) and hyphenated terms; stems everything else
- Index is serialized to `data/bm25_index.pkl`

```bash
uv run python -m nba_search.index                     # build index
uv run python -m nba_search.index search "Jayson Tatum triple double"
```

---

## In Progress

### Embedding Benchmark (`docs/superpowers/plans/2026-04-15-embedding-benchmark.md`)
Evaluating open-source sentence embedding models for dense retrieval as a complement or replacement for BM25.

Pipeline stages:
- **`eval_gen.py`** — generates a Claude-produced eval set (query → relevant game IDs)
- **`embed.py`** — encodes corpus once per model, caches to `.npy` files
- **`benchmark.py`** — runs cosine retrieval, computes Recall@K, outputs terminal table + CSV

---

## Setup

```bash
uv sync
uv run python -m nba_search.ingest   # populate corpus
uv run python -m nba_search.index    # build BM25 index
```

Requires Python 3.12+.
