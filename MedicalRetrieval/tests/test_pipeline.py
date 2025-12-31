"""
Tests for the MedicalRetrieval pipeline components.
"""
import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import Document, parse_keywords, create_document_index
from utils import OpenAIClient, cluster_documents_by_keywords, compute_llm_judge_metrics
from query_generator import Query, queries_to_list, queries_to_dict
from llm_judge import RelevanceJudgment


class TestDataLoader(unittest.TestCase):
    """Tests for data_loader module."""

    def test_parse_keywords_valid_list(self):
        """Test parsing valid Python list string."""
        keywords_str = "['华氏巨球蛋白血症', '蛋白电泳', 'MYD88L265P']"
        result = parse_keywords(keywords_str)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], '华氏巨球蛋白血症')

    def test_parse_keywords_empty(self):
        """Test parsing empty string returns empty list."""
        self.assertEqual(parse_keywords(""), [])
        self.assertEqual(parse_keywords("   "), [])

    def test_parse_keywords_fallback(self):
        """Test fallback to comma splitting for invalid list."""
        result = parse_keywords("keyword1, keyword2, keyword3")
        self.assertEqual(len(result), 3)

    def test_document_full_text(self):
        """Test Document.full_text concatenation."""
        doc = Document(
            doc_id=0,
            title="Test Title",
            abstract="Test Abstract",
            keywords=["kw1", "kw2"]
        )
        full_text = doc.full_text
        self.assertIn("Test Title", full_text)
        self.assertIn("Test Abstract", full_text)
        self.assertIn("kw1", full_text)

    def test_create_document_index(self):
        """Test document index creation."""
        docs = [
            Document(doc_id=0, title="Doc 0", abstract="", keywords=[]),
            Document(doc_id=1, title="Doc 1", abstract="", keywords=[]),
        ]
        index = create_document_index(docs)
        self.assertEqual(len(index), 2)
        self.assertEqual(index[0].title, "Doc 0")
        self.assertEqual(index[1].title, "Doc 1")


class TestUtils(unittest.TestCase):
    """Tests for utils module."""

    def test_cluster_documents_by_keywords(self):
        """Test keyword clustering."""
        docs = [
            Document(doc_id=0, title="", abstract="", keywords=["cancer", "treatment"]),
            Document(doc_id=1, title="", abstract="", keywords=["cancer", "diagnosis"]),
            Document(doc_id=2, title="", abstract="", keywords=["diabetes", "treatment"]),
            Document(doc_id=3, title="", abstract="", keywords=["cancer", "research"]),
            Document(doc_id=4, title="", abstract="", keywords=["cancer", "therapy"]),
            Document(doc_id=5, title="", abstract="", keywords=["cancer", "prevention"]),
        ]
        clusters = cluster_documents_by_keywords(docs, min_cluster_size=3)
        # "cancer" should have 5 documents, "treatment" only 2
        self.assertIn("cancer", clusters)
        self.assertEqual(len(clusters["cancer"]), 5)
        self.assertNotIn("treatment", clusters)  # Only 2 docs

    def test_compute_llm_judge_metrics(self):
        """Test metrics computation."""
        judgments = [
            {"query_id": "0", "doc_id": 1, "relevance_score": 2, "rank": 0},
            {"query_id": "0", "doc_id": 2, "relevance_score": 1, "rank": 1},
            {"query_id": "0", "doc_id": 3, "relevance_score": 0, "rank": 2},
            {"query_id": "1", "doc_id": 4, "relevance_score": 2, "rank": 0},
            {"query_id": "1", "doc_id": 5, "relevance_score": 2, "rank": 1},
        ]
        metrics = compute_llm_judge_metrics(judgments, k_values=[1, 5])
        self.assertIn("mean_relevance", metrics)
        self.assertIn("precision@1", metrics)
        self.assertEqual(metrics["zero_relevance_query_count"], 0)


class TestQueryGenerator(unittest.TestCase):
    """Tests for query_generator module."""

    def test_queries_to_list(self):
        """Test converting queries to list format."""
        queries = [
            Query(query_id=0, query_text="query 0", source_keyword="kw0", source_doc_ids=[0, 1]),
            Query(query_id=1, query_text="query 1", source_keyword="kw1", source_doc_ids=[2, 3]),
        ]
        result = queries_to_list(queries)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ("0", "query 0"))
        self.assertEqual(result[1], ("1", "query 1"))

    def test_queries_to_dict(self):
        """Test converting queries to dict format."""
        queries = [
            Query(query_id=0, query_text="query 0", source_keyword="kw0", source_doc_ids=[0, 1]),
        ]
        result = queries_to_dict(queries)
        self.assertIn("0", result)
        self.assertEqual(result["0"]["query_text"], "query 0")


class TestLLMJudge(unittest.TestCase):
    """Tests for llm_judge module."""

    def test_relevance_judgment_dataclass(self):
        """Test RelevanceJudgment creation."""
        judgment = RelevanceJudgment(
            query_id="0",
            query_text="test query",
            doc_id=123,
            doc_title="Test Doc",
            relevance_score=2,
            rank=0,
            retrieval_score=0.85,
            rerank_score=10.0,
            explanation="Highly relevant"
        )
        self.assertEqual(judgment.relevance_score, 2)
        self.assertEqual(judgment.query_id, "0")


class TestOpenAIClient(unittest.TestCase):
    """Tests for OpenAI client with retry logic."""

    @patch('utils.OpenAI')
    def test_client_initialization(self, mock_openai):
        """Test client initializes correctly."""
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-4o-mini",
            retry_attempts=3
        )
        self.assertEqual(client.model, "gpt-4o-mini")
        self.assertEqual(client.retry_attempts, 3)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for pipeline components."""

    def test_document_flow(self):
        """Test document flows correctly through the system."""
        # Create test documents
        docs = [
            Document(
                doc_id=i,
                title=f"Test Document {i}",
                abstract=f"Abstract for document {i}",
                keywords=[f"keyword_{i}", "common_keyword"]
            )
            for i in range(10)
        ]

        # Test index creation
        index = create_document_index(docs)
        self.assertEqual(len(index), 10)

        # Test clustering
        clusters = cluster_documents_by_keywords(docs, min_cluster_size=5)
        self.assertIn("common_keyword", clusters)
        self.assertEqual(len(clusters["common_keyword"]), 10)

    def test_judgment_to_metrics_flow(self):
        """Test judgment data flows to metrics correctly."""
        judgments = [
            {"query_id": "0", "doc_id": i, "relevance_score": i % 3, "rank": i}
            for i in range(10)
        ]
        metrics = compute_llm_judge_metrics(judgments, k_values=[1, 5, 10])

        self.assertIn("mean_relevance", metrics)
        self.assertIn("precision@5", metrics)
        self.assertIn("ndcg@10", metrics)


class TestRunPipeline(unittest.TestCase):
    """Tests for run_pipeline module."""

    def test_run_pipeline_module_imports(self):
        """Test that run_pipeline can be imported."""
        try:
            from run_pipeline import run_pipeline, main
            self.assertTrue(callable(run_pipeline))
            self.assertTrue(callable(main))
        except ImportError as e:
            self.fail(f"Failed to import run_pipeline: {e}")


if __name__ == "__main__":
    unittest.main()
