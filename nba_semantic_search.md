# 🏀 NBA Semantic Search

A hybrid retrieval system for NBA game data combining **dense vector search**, **sparse BM25 retrieval**, and **cross-encoder re-ranking** — built as a portfolio project to demonstrate production-grade search engineering.

---

## What It Does

Ask natural language questions over a corpus of NBA game summaries and get semantically ranked results:

> *"Find games where a player went off in the 4th quarter despite a slow start"*
> *"Games decided by a buzzer beater after a blown double-digit lead"*
> *"Dominant defensive performances by a center against a top-10 offense"*

Traditional keyword search breaks on these queries. This system doesn't.

---

## Architecture

```
Data Ingestion
  nba_api ──────────────────────────────┐
  ESPN Unofficial API (game recaps) ────┤──▶ Document Builder ──▶ Corpus (JSONL)
                                        │
                                        ▼
                           ┌──────────────────────┐
                           │   Indexing Pipeline  │
                           │                      │
                           │  BM25 (rank_bm25)    │ ◀── Sparse
                           │  FAISS / Qdrant      │ ◀── Dense (nomic-embed-text)
                           └──────────┬───────────┘
                                      │
                           ┌──────────▼───────────┐
                           │    Query Pipeline    │
                           │                      │
                           │  Retrieve (top-k)    │
                           │  RRF Fusion          │
                           │  Cross-Encoder Rerank│
                           └──────────┬───────────┘
                                      │
                               Ranked Results
```

### Stack

| Component | Tool |
|-----------|------|
| Structured stats | nba_api |
| Game recap text | ESPN Unofficial API |
| Sparse retrieval | rank_bm25 |
| Dense retrieval | FAISS (local) or Qdrant |
| Embeddings | nomic-embed-text |
| Score fusion | Reciprocal Rank Fusion (RRF) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Interface | Streamlit or FastAPI |

### Document Schema

Each document in the corpus represents one NBA game:

```json
{
  "game_id": "401584793",
  "date": "2024-06-17",
  "home_team": "Boston Celtics",
  "away_team": "Dallas Mavericks",
  "home_score": 106,
  "away_score": 88,
  "top_performers": [
    { "player": "Jayson Tatum", "pts": 31, "reb": 11, "ast": 5 }
  ],
  "recap_text": "...",
  "stat_summary": "...",
  "full_text": "..."
}
```

`full_text` is the BM25 + embedding target — a templated fusion of recap prose and stat highlights.

---

## Retrieval Pipeline

### 1. Sparse Retrieval (BM25)

Token-level matching over `full_text`. Good for exact player names, team names, stat references.

### 2. Dense Retrieval (ANN)

Embedding similarity via `nomic-embed-text`. Good for semantic queries that don't mention specific terms.

### 3. Reciprocal Rank Fusion (RRF)

Merges ranked lists from both retrievers without requiring score normalization:

```
rrf_score(d) = Σ 1 / (k + rank_i(d))   # k=60 by default
```

### 4. Cross-Encoder Re-ranking

Top-N fused results passed through `ms-marco-MiniLM` for precise relevance scoring between query and document.

---

## Evaluation

Retrieval quality is measured across three configurations:

| Config | P@5 | MRR | NDCG@10 |
|--------|-----|-----|---------|
| BM25 only | — | — | — |
| Dense only | — | — | — |
| Hybrid + Rerank | — | — | — |

A hand-labeled eval set of 50 queries with expected game results is included in `eval/queries.jsonl`.

---

## Sample Results: BM25 Only (Baseline)

Corpus: 396 games (2026 season). Queries run against `full_text` using the custom tokenizer (proper noun preservation + Porter stemming).

---

**Query:** `"buzzer beater comeback"`

| Score | Date | Matchup | Result | Top Performers |
|-------|------|---------|--------|----------------|
| 6.417 | 2026-02-22 | Houston Rockets @ New York Knicks | 106-108 | Kevin Durant 30pts, Karl-Anthony Towns 25pts |
| 5.788 | 2026-02-22 | Cleveland Cavaliers @ Oklahoma City Thunder | 113-121 | Isaiah Joe 22pts, James Harden 20pts |
| 3.783 | 2026-03-20 | LA Clippers @ New Orleans Pelicans | 99-105 | Trey Murphy III 27pts, Derrick Jones Jr. 22pts |
| 3.692 | 2026-03-21 | Los Angeles Lakers @ Orlando Magic | 105-104 | Luka Doncic 33pts, Austin Reaves 26pts |
| 3.672 | 2026-04-02 | Denver Nuggets @ Utah Jazz | 130-117 | Jamal Murray 37pts, Brice Sensabaugh 28pts |

*Observation: top result (Knicks 108-106) and #4 (Lakers 105-104) are close games plausibly involving late-game drama. BM25 is matching on word overlap ("comeback", "buzzer") in recap text — scores drop sharply after the top 2, suggesting sparse term coverage.*

---

**Query:** `"dominant defensive performance"`

| Score | Date | Matchup | Result | Top Performers |
|-------|------|---------|--------|----------------|
| 6.258 | 2026-03-28 | San Antonio Spurs @ Milwaukee Bucks | 127-95 | Victor Wembanyama 23pts, Stephon Castle 22pts |
| 5.490 | 2026-03-27 | Miami Heat @ Cleveland Cavaliers | 128-149 | Max Strus 29pts, Evan Mobley 23pts |
| 5.475 | 2026-02-22 | Philadelphia 76ers @ New Orleans Pelicans | 111-126 | Tyrese Maxey 27pts, Kelly Oubre Jr. 25pts |
| 4.110 | 2026-04-11 | Phoenix Suns @ Los Angeles Lakers | 73-101 | LeBron James 28pts, Luke Kennard 19pts |
| 3.946 | 2026-03-04 | Washington Wizards @ Orlando Magic | 109-126 | Paolo Banchero 37pts, Desmond Bane 25pts |

*Observation: Wembanyama/Spurs blowout (127-95) is a plausible defensive result. The Lakers 101-73 win (#4) is a legitimate blowout where defense is the story. However, the Cavaliers 149-128 result (#2) is a high-scoring game — BM25 matched "dominant" or "defensive" from the recap without understanding these are contradictory signals for the query.*

---

**Query:** `"triple double"`

| Score | Date | Matchup | Result | Top Performers |
|-------|------|---------|--------|----------------|
| 2.937 | 2026-04-01 | New York Knicks @ Houston Rockets | 94-111 | Kevin Durant 27pts, Karl-Anthony Towns 22pts |
| 2.410 | 2026-04-09 | Indiana Pacers @ Brooklyn Nets | 123-94 | Obi Toppin 26pts, E.J. Liddell 26pts |
| 2.324 | 2026-03-31 | Phoenix Suns @ Memphis Grizzlies | 131-105 | Devin Booker 36pts, Jalen Green 21pts |
| 2.283 | 2026-03-21 | Los Angeles Lakers @ Orlando Magic | 105-104 | Luka Doncic 33pts, Austin Reaves 26pts |
| 2.265 | 2026-03-20 | Milwaukee Bucks @ Utah Jazz | 96-128 | Ace Bailey 33pts, Cody Williams 23pts |

*Observation: scores are uniformly low (all under 3.0) — "triple-double" is a hyphenated compound that BM25 treats as a single token. If a recap writes "triple double" (no hyphen) or "10 assists, 11 rebounds" instead, there is zero match. This is the clearest example of BM25's lexical brittleness and the strongest motivation for adding dense retrieval.*

---

## Project Structure

```
nba-semantic-search/
├── data/
│   └── corpus.jsonl              # Generated game documents
├── eval/
│   ├── queries.jsonl             # Labeled eval queries
│   └── evaluate.py               # NDCG / MRR / P@K scoring
├── notebooks/
│   └── retrieval_comparison.ipynb  # BM25 vs Dense vs Hybrid analysis
├── src/
│   ├── ingest.py                 # nba_api + ESPN ingestion
│   ├── index.py                  # Build BM25 + FAISS indexes
│   ├── retrieve.py               # Hybrid retrieval + RRF
│   ├── rerank.py                 # Cross-encoder re-ranking
│   └── app.py                    # Streamlit interface
├── requirements.txt
└── README.md
```

---

## Build Steps

### 1. Data Ingestion
- [x] Fetch game IDs from ESPN scoreboard API for a configurable number of past days
- [x] Fetch full game summary per game (scores, boxscore, recap article)
- [x] Extract top performers from boxscore
- [x] Extract and clean recap text (strip HTML)
- [x] Validate documents against Pydantic schema (`GameDocument`)
- [x] Write corpus to `data/corpus.jsonl`
- [x] CLI: `python -m nba_search.ingest --days N`

### 2. Indexing
- [x] Build BM25 index over `full_text` (`rank_bm25`)
- [x] Custom tokenizer: proper noun preservation + Porter stemming + hyphenated term handling
- [x] Persist BM25 index to disk (`data/bm25_index.pkl`)
- [x] BM25 search CLI (`python -m nba_search.index search <query>`)
- [x] Document baseline BM25 results (3 sample queries)
- [ ] Embedding model bakeoff (`experiments/embedding_bakeoff.ipynb`) — compare candidates (e.g. `nomic-embed-text`, `all-MiniLM-L6-v2`, `bge-small-en`) on retrieval quality before committing to one
- [ ] Generate embeddings with chosen model
- [ ] Store embeddings in a vector database (Qdrant local)

### 3. Retrieval
- [ ] BM25 sparse retrieval (top-k)
- [ ] FAISS dense retrieval (top-k)
- [ ] Reciprocal Rank Fusion to merge ranked lists

### 4. Re-ranking
- [ ] Cross-encoder re-ranking with `ms-marco-MiniLM-L-6-v2`

### 5. Evaluation
- [ ] Build hand-labeled query set (`eval/queries.jsonl`)
- [ ] Implement NDCG / MRR / P@K scoring (`eval/evaluate.py`)
- [ ] Compare BM25-only vs dense-only vs hybrid+rerank

### 6. Interface
- [ ] Streamlit search UI (`src/app.py`)

---

## Quickstart

```bash
# Install dependencies
uv sync

# Pull game data for the last N days
uv run python -m nba_search.ingest --days 7

# Build indexes
uv run python -m nba_search.index

# Launch search interface
uv run streamlit run src/nba_search/app.py
```

---

## Data Sources

- **ESPN Unofficial API** — game recap text and box scores via `site.api.espn.com` (no auth required)

> **Note:** The ESPN endpoints used here are unofficial and undocumented. Data is cached locally after ingestion. Do not redistribute raw recap text.

---

## Key Concepts Demonstrated

- **Hybrid search:** combining lexical and semantic retrieval
- **Reciprocal Rank Fusion:** score-free rank aggregation
- **Cross-encoder re-ranking:** late-stage precision improvement
- **Retrieval eval harness:** measuring P@K, MRR, NDCG
- **Document design:** structuring heterogeneous data for retrieval

---

## Roadmap

- [ ] Agentic query expansion (ReAct loop for multi-hop queries)
- [ ] Player career search (entity-level, not game-level)
- [ ] Qdrant cloud deployment
- [ ] Filtering by team, season, game type (playoffs vs regular)

---

## Author

Built by Frank — Senior Data Scientist with a focus on embedding search and entity resolution at scale.
