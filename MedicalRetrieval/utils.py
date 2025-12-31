"""
Utility functions for Medical Retrieval pipeline.
"""
import time
import json
from typing import List, Dict, Any, Callable, Optional
from collections import defaultdict
from openai import OpenAI


class OpenAIClient:
    """OpenAI API client with retry logic."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        retry_attempts: int = 3,
        retry_backoff_base: float = 1.0
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.retry_attempts = retry_attempts
        self.retry_backoff_base = retry_backoff_base

    def _retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                wait_time = self.retry_backoff_base * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise last_exception

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Get chat completion with retry."""
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        return self._retry_with_backoff(_call)

    def chat_completion_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict:
        """Get chat completion and parse as JSON."""
        response_text = self.chat_completion(messages, temperature, max_tokens)

        # Try to extract JSON from response
        try:
            # Check for JSON code block
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Failed to parse JSON from response: {response_text[:500]}")
            raise ValueError(f"Invalid JSON response: {e}")


def cluster_documents_by_keywords(
    documents: List,
    min_cluster_size: int = 5
) -> Dict[str, List[int]]:
    """
    Cluster documents by their keywords.

    Args:
        documents: List of Document objects
        min_cluster_size: Minimum number of docs per keyword to form a cluster

    Returns:
        Dict mapping keyword to list of document IDs
    """
    keyword_to_docs = defaultdict(list)

    for doc in documents:
        for keyword in doc.keywords:
            keyword_to_docs[keyword].append(doc.doc_id)

    # Filter to clusters meeting minimum size
    clusters = {
        kw: doc_ids
        for kw, doc_ids in keyword_to_docs.items()
        if len(doc_ids) >= min_cluster_size
    }

    return clusters


def select_diverse_clusters(
    clusters: Dict[str, List[int]],
    num_clusters: int,
    min_overlap: float = 0.3
) -> List[str]:
    """
    Select diverse keyword clusters with minimal overlap.

    Args:
        clusters: Dict mapping keyword to document IDs
        num_clusters: Number of clusters to select
        min_overlap: Maximum overlap ratio allowed between selected clusters

    Returns:
        List of selected keyword names
    """
    # Sort by cluster size (prefer larger clusters)
    sorted_keywords = sorted(clusters.keys(), key=lambda k: len(clusters[k]), reverse=True)

    selected = []
    selected_docs = set()

    for keyword in sorted_keywords:
        if len(selected) >= num_clusters:
            break

        doc_set = set(clusters[keyword])

        # Check overlap with already selected clusters
        if selected_docs:
            overlap = len(doc_set & selected_docs) / len(doc_set)
            if overlap > min_overlap:
                continue

        selected.append(keyword)
        selected_docs.update(doc_set)

    return selected


def compute_llm_judge_metrics(
    judgments: List[Dict],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute aggregate metrics from LLM judge relevance scores.

    Args:
        judgments: List of dicts with query_id, doc_id, relevance_score (0/1/2)
        k_values: K values for precision@K computation

    Returns:
        Dict of metric names to values
    """
    # Group by query
    query_judgments = defaultdict(list)
    for j in judgments:
        query_judgments[j["query_id"]].append(j)

    metrics = {}

    # Mean relevance score
    all_scores = [j["relevance_score"] for j in judgments]
    metrics["mean_relevance"] = sum(all_scores) / len(all_scores) if all_scores else 0

    # Per-query metrics
    precision_at_k = {k: [] for k in k_values}
    ndcg_at_k = {k: [] for k in k_values}

    for query_id, qj in query_judgments.items():
        # Sort by retrieval rank (assuming doc_rank field exists)
        qj_sorted = sorted(qj, key=lambda x: x.get("rank", 0))

        for k in k_values:
            top_k = qj_sorted[:k]
            if not top_k:
                continue

            # Precision@K: fraction of docs rated >= 1 (partially or highly relevant)
            relevant_count = sum(1 for j in top_k if j["relevance_score"] >= 1)
            precision_at_k[k].append(relevant_count / len(top_k))

            # NDCG@K with graded relevance
            dcg = sum(
                j["relevance_score"] / (i + 2)  # log2(i+2) approximated
                for i, j in enumerate(top_k)
            )
            # Ideal DCG: all 2s
            idcg = sum(2 / (i + 2) for i in range(len(top_k)))
            ndcg_at_k[k].append(dcg / idcg if idcg > 0 else 0)

    # Average across queries
    for k in k_values:
        if precision_at_k[k]:
            metrics[f"precision@{k}"] = sum(precision_at_k[k]) / len(precision_at_k[k])
        if ndcg_at_k[k]:
            metrics[f"ndcg@{k}"] = sum(ndcg_at_k[k]) / len(ndcg_at_k[k])

    # Count zero-relevance queries (all docs rated 0)
    zero_relevance_queries = [
        qid for qid, qj in query_judgments.items()
        if all(j["relevance_score"] == 0 for j in qj)
    ]
    metrics["zero_relevance_query_count"] = len(zero_relevance_queries)
    metrics["zero_relevance_query_ids"] = zero_relevance_queries

    return metrics


def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: str):
    """Save data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
