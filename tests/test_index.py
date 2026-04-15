"""Tests for BM25 indexing pipeline."""

import json
import tempfile
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi

from nba_search.index import build_bm25_index, load_corpus, load_index, save_index, tokenize
from nba_search.models import GameDocument, Performer


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_stems_common_verbs(self):
        # Porter treats "scorer" as a noun — it stays as "scorer"
        # Only verb forms collapse to the stem
        tokens = tokenize("scored scoring")
        assert all(t == "score" for t in tokens)

    def test_noun_form_not_over_stemmed(self):
        tokens = tokenize("scorer")
        assert tokens == ["scorer"]  # noun, not stemmed to "score"

    def test_preserves_capitalized_proper_nouns(self):
        tokens = tokenize("Tatum dominated the Mavericks")
        assert "tatum" in tokens
        assert "maverick" not in tokens  # stemmer would corrupt this
        assert "mavericks" in tokens

    def test_preserves_hyphenated_terms(self):
        tokens = tokenize("triple-double 3-pointer pull-up")
        assert "triple-double" in tokens
        assert "3-pointer" in tokens
        assert "pull-up" in tokens

    def test_lowercases_everything(self):
        tokens = tokenize("Celtics BOSTON celtics")
        assert all(t == t.lower() for t in tokens)

    def test_strips_punctuation_around_words(self):
        tokens = tokenize("dominant. explosive, unstoppable!")
        assert "." not in " ".join(tokens)
        assert "," not in " ".join(tokens)

    def test_empty_string(self):
        assert tokenize("") == []

    def test_numbers_preserved(self):
        # Stat references like "31" and "40" should survive
        tokens = tokenize("scored 31 points in 40 minutes")
        assert "31" in tokens
        assert "40" in tokens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_game(game_id: str, full_text: str) -> GameDocument:
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
    make_game("001", "Jayson Tatum scored 31 points with 11 rebounds in a dominant Celtics win"),
    make_game("002", "Luka Doncic exploded for 45 points leading the Mavericks comeback victory"),
    make_game("003", "Defensive masterclass by Rudy Gobert held the opposing offense to 89 points"),
]


# ---------------------------------------------------------------------------
# Index build tests
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_returns_bm25_okapi(self):
        index = build_bm25_index(SAMPLE_DOCS)
        assert isinstance(index, BM25Okapi)

    def test_index_has_correct_doc_count(self):
        index = build_bm25_index(SAMPLE_DOCS)
        assert index.corpus_size == len(SAMPLE_DOCS)

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError, match="empty corpus"):
            build_bm25_index([])


# ---------------------------------------------------------------------------
# Retrieval smoke tests
# ---------------------------------------------------------------------------


class TestRetrieval:
    def setup_method(self):
        self.index = build_bm25_index(SAMPLE_DOCS)
        self.game_ids = [doc.game_id for doc in SAMPLE_DOCS]

    def test_returns_scores_for_all_docs(self):
        scores = self.index.get_scores(tokenize("Tatum points"))
        assert len(scores) == len(SAMPLE_DOCS)

    def test_tatum_query_ranks_celtics_game_first(self):
        scores = self.index.get_scores(tokenize("Tatum scored points"))
        top_idx = scores.argmax()
        assert self.game_ids[top_idx] == "001"

    def test_defense_query_ranks_gobert_game_first(self):
        scores = self.index.get_scores(tokenize("defensive performance held offense"))
        top_idx = scores.argmax()
        assert self.game_ids[top_idx] == "003"

    def test_get_top_n(self):
        results = self.index.get_top_n(tokenize("points"), SAMPLE_DOCS, n=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self):
        index = build_bm25_index(SAMPLE_DOCS)
        original_scores = index.get_scores(tokenize("Tatum"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index.pkl"
            save_index(index, path)
            loaded = load_index(path)

        loaded_scores = loaded.get_scores(tokenize("Tatum"))
        assert list(original_scores) == list(loaded_scores)


# ---------------------------------------------------------------------------
# Corpus loading tests
# ---------------------------------------------------------------------------


class TestLoadCorpus:
    def test_loads_valid_corpus(self, tmp_path):
        corpus_path = tmp_path / "corpus.jsonl"
        corpus_path.write_text(
            "\n".join(doc.model_dump_json() for doc in SAMPLE_DOCS)
        )
        docs = load_corpus(corpus_path)
        assert len(docs) == len(SAMPLE_DOCS)
        assert all(isinstance(d, GameDocument) for d in docs)
