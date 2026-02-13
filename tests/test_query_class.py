"""Tests for query classification and adaptive budgets."""

from brr.core.query_class import QueryClass
from brr.core.query_class import adaptive_budget
from brr.core.query_class import classify_query


def test_empty_query():
    assert classify_query("") == QueryClass.EMPTY
    assert classify_query("   ") == QueryClass.EMPTY


def test_identifier_path():
    assert classify_query("src/main.rs") == QueryClass.IDENTIFIER


def test_identifier_ticket():
    assert classify_query("br-123") == QueryClass.IDENTIFIER
    assert classify_query("JIRA-456") == QueryClass.IDENTIFIER


def test_identifier_single_word():
    assert classify_query("FnvHashEmbedder") == QueryClass.IDENTIFIER


def test_short_keyword():
    assert classify_query("error handling") == QueryClass.SHORT_KEYWORD
    assert classify_query("search index build") == QueryClass.SHORT_KEYWORD


def test_natural_language():
    assert classify_query("how does the search pipeline work?") == QueryClass.NATURAL_LANGUAGE
    assert classify_query("what is reciprocal rank fusion") == QueryClass.NATURAL_LANGUAGE


def test_budget_empty():
    budget = adaptive_budget(QueryClass.EMPTY)
    assert budget.lexical_multiplier == 0
    assert budget.semantic_multiplier == 0


def test_budget_identifier_prefers_lexical():
    budget = adaptive_budget(QueryClass.IDENTIFIER, base_multiplier=3)
    assert budget.lexical_multiplier > budget.semantic_multiplier


def test_budget_natural_language_prefers_semantic():
    budget = adaptive_budget(QueryClass.NATURAL_LANGUAGE, base_multiplier=3)
    assert budget.semantic_multiplier > budget.lexical_multiplier


def test_budget_short_keyword_balanced():
    budget = adaptive_budget(QueryClass.SHORT_KEYWORD, base_multiplier=3)
    assert budget.lexical_multiplier == budget.semantic_multiplier
