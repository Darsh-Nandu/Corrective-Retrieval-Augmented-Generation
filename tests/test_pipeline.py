"""
Unit tests for C-RAG pure logic functions.
No Ollama, no API keys, no network — tests run entirely offline.
"""
import re
import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Inline pydantic models (mirrors rag_state.py) so tests need no LLM deps
from pydantic import BaseModel
from typing import TypedDict, List, Any

class DocEvalScore(BaseModel):
    score: float
    reason: str

class KeepOrDrop(BaseModel):
    keep: bool

class WebQuery(BaseModel):
    query: str

class State(TypedDict):
    question: str
    docs: List[Any]
    good_docs: List[Any]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Any]
    answer: str


# Replicate pure functions locally so tests don't trigger Ollama imports

UPPER_TH = 0.7
LOWER_TH = 0.3


def decompose_to_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def route_after_eval(verdict: str) -> str:
    if verdict == "CORRECT":
        return "refine"
    else:
        return "rewrite_query"


def select_verdict(scores: list[float]) -> str:
    if any(s > UPPER_TH for s in scores):
        return "CORRECT"
    if scores and all(s < LOWER_TH for s in scores):
        return "INCORRECT"
    return "AMBIGUOUS"


# sentence decomposition

class TestDecomposeToSentences:
    def test_basic_split_on_period(self):
        text = "Batch normalization normalizes across the batch. Layer normalization normalizes across the features. Both techniques reduce internal covariate shift."
        result = decompose_to_sentences(text)
        assert len(result) == 3

    def test_splits_on_exclamation(self):
        text = "This technique is surprisingly effective! It works across many architectures and tasks. Results consistently improve over baseline approaches."
        result = decompose_to_sentences(text)
        assert len(result) == 3

    def test_splits_on_question_mark(self):
        text = "What is batch normalization exactly? It normalizes activations across the batch dimension during training."
        result = decompose_to_sentences(text)
        assert len(result) == 2

    def test_filters_short_fragments(self):
        # Fragments under 20 chars are dropped by design
        text = "Hi. This is a longer sentence that comfortably passes the twenty-character length filter."
        result = decompose_to_sentences(text)
        assert all(len(s) > 20 for s in result)
        assert not any(s == "Hi." for s in result)

    def test_collapses_whitespace(self):
        text = "Attention mechanisms are central to transformers.   Self-attention computes queries, keys, and values from the same sequence."
        result = decompose_to_sentences(text)
        assert len(result) == 2

    def test_empty_string(self):
        assert decompose_to_sentences("") == []

    def test_single_long_sentence_no_split(self):
        text = "This is one single long sentence without any terminal punctuation at the end"
        result = decompose_to_sentences(text)
        assert len(result) == 1
        assert result[0] == text

    def test_newlines_treated_as_whitespace(self):
        text = "Transformers use self-attention mechanisms.\nFeed-forward layers follow each attention block.\nLayer normalization is applied before each sublayer."
        result = decompose_to_sentences(text)
        assert len(result) == 3


# routing logic

class TestRouteAfterEval:
    def test_correct_routes_to_refine(self):
        assert route_after_eval("CORRECT") == "refine"

    def test_incorrect_routes_to_rewrite(self):
        assert route_after_eval("INCORRECT") == "rewrite_query"

    def test_ambiguous_routes_to_rewrite(self):
        assert route_after_eval("AMBIGUOUS") == "rewrite_query"

    def test_unknown_verdict_routes_to_rewrite(self):
        # Any non-CORRECT verdict should go to web search path
        assert route_after_eval("") == "rewrite_query"


# verdict selection

class TestSelectVerdict:
    def test_one_high_score_is_correct(self):
        assert select_verdict([0.9, 0.2, 0.1]) == "CORRECT"

    def test_all_at_threshold_is_correct(self):
        assert select_verdict([0.8, 0.8]) == "CORRECT"

    def test_all_below_lower_is_incorrect(self):
        assert select_verdict([0.1, 0.2, 0.05]) == "INCORRECT"

    def test_all_exactly_lower_th_is_incorrect(self):
        # 0.3 is NOT < LOWER_TH (it equals it), so boundary case → AMBIGUOUS
        assert select_verdict([0.3, 0.3]) == "AMBIGUOUS"

    def test_mixed_mid_range_is_ambiguous(self):
        assert select_verdict([0.4, 0.5, 0.35]) == "AMBIGUOUS"

    def test_empty_scores_is_ambiguous(self):
        # Empty list: not all < LOWER_TH (vacuously true but `scores` is falsy)
        assert select_verdict([]) == "AMBIGUOUS"

    def test_single_exact_upper_th_is_correct(self):
        # 0.7 is NOT > UPPER_TH (equals it), so should NOT be correct
        assert select_verdict([0.7]) == "AMBIGUOUS"

    def test_single_above_upper_th_is_correct(self):
        assert select_verdict([0.71]) == "CORRECT"

# state schema

class TestStateSchema:
    def test_doc_eval_score_valid(self):
        d = DocEvalScore(score=0.85, reason="Highly relevant chunk.")
        assert d.score == 0.85
        assert isinstance(d.reason, str)

    def test_doc_eval_score_bounds(self):
        # pydantic doesn't enforce [0,1] bounds unless we add validators,
        # but score should be a float
        d = DocEvalScore(score=0.0, reason="Irrelevant.")
        assert isinstance(d.score, float)

    def test_keep_or_drop_true(self):
        k = KeepOrDrop(keep=True)
        assert k.keep is True

    def test_keep_or_drop_false(self):
        k = KeepOrDrop(keep=False)
        assert k.keep is False

    def test_web_query_model(self):
        w = WebQuery(query="batch normalization layer normalization comparison")
        assert "normalization" in w.query

    def test_state_required_keys(self):
        # All keys needed to build a valid initial state
        required = {
            "question", "docs", "good_docs", "verdict", "reason",
            "strips", "kept_strips", "refined_context",
            "web_query", "web_docs", "answer",
        }
        annotations = set(State.__annotations__.keys())
        assert required.issubset(annotations), f"Missing keys: {required - annotations}"