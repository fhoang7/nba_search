"""
Indexing pipeline: builds BM25 and (later) dense vector indexes over the corpus.
"""

import json
import pickle
import re
from pathlib import Path

from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from nba_search.models import GameDocument

CORPUS_PATH = Path("data/corpus.jsonl")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")

_stemmer = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.

    Strategy:
    - Tokens with special punctuation (hyphens, apostrophes, etc.) are kept as-is
      since they often carry specific meaning (e.g. "triple-double", "3-pointer").
    - Tokens that were capitalized in the original text are kept as-is since they
      are likely proper nouns (player names, team names) that stemming would corrupt.
    - All other tokens are lowercased and stemmed.
    """
    # Split on whitespace to preserve original casing for proper noun detection
    raw_tokens = text.split()

    result = []
    for raw in raw_tokens:
        # Strip leading/trailing punctuation for the clean token, but keep the
        # raw form for casing detection
        clean = re.sub(r"^[^\w]+|[^\w]+$", "", raw)
        if not clean:
            continue

        has_special_punctuation = bool(re.search(r"[-'.]", clean))
        is_capitalized = raw[0].isupper()

        if has_special_punctuation or is_capitalized:
            result.append(clean.lower())
        else:
            result.append(_stemmer.stem(clean.lower()))

    return result


def load_corpus(path: Path = CORPUS_PATH) -> list[GameDocument]:
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(GameDocument.model_validate_json(line.strip()))
    return docs


def build_bm25_index(docs: list[GameDocument]) -> BM25Okapi:
    if not docs:
        raise ValueError("Cannot build BM25 index from empty corpus")
    tokenized = [tokenize(doc.full_text) for doc in docs]
    return BM25Okapi(tokenized)


def save_index(index: BM25Okapi, path: Path = BM25_INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_index(path: Path = BM25_INDEX_PATH) -> BM25Okapi:
    with open(path, "rb") as f:
        return pickle.load(f)


def search(query: str, top_k: int = 5) -> None:
    docs = load_corpus()
    index = load_index()
    scores = index.get_scores(tokenize(query))

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:top_k]
    for score, doc in ranked:
        print(f"[{score:.3f}] {doc.date} | {doc.away_team} @ {doc.home_team} ({doc.away_score}-{doc.home_score})")
        if doc.top_performers:
            names = ", ".join(f"{p.player} {p.pts}pts" for p in doc.top_performers)
            print(f"         {names}")
        print()


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:])
        if not query:
            print("Usage: python -m nba_search.index search <query>")
            sys.exit(1)
        search(query)
        return

    print(f"Loading corpus from {CORPUS_PATH}...")
    docs = load_corpus()
    print(f"  {len(docs)} documents loaded")

    print("Building BM25 index...")
    index = build_bm25_index(docs)

    print(f"Saving index to {BM25_INDEX_PATH}...")
    save_index(index)
    print("Done.")


if __name__ == "__main__":
    main()
