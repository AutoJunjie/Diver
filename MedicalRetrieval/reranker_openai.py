"""
Reranker using OpenAI GPT-4o-mini.
Implements listwise reranking of retrieved documents.
"""
import re
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

from utils import OpenAIClient
from data_loader import Document
from retriever import RetrievalResult


@dataclass
class RerankResult:
    """Represents reranked results for a query."""
    query_id: str
    query_text: str
    reranked_doc_scores: Dict[int, float]  # doc_id -> rerank score


RERANK_PROMPT = """以下是与查询相关的医学文献文档。请根据它们与查询的相关性对这些文档进行排序。

查询: {query}

文档:
{documents}

请首先分析查询的核心需求，然后逐一评估每个文档的相关性，最后给出排序结果。

请严格按照以下JSON格式输出排序结果（从最相关到最不相关的文档ID列表）:
```json
[1, 3, 2, ...]
```

注意：列表中的数字是上面文档的编号（1开始），按相关性从高到低排列。必须包含所有文档编号。"""


def format_documents_for_rerank(
    docs_with_scores: List[Tuple[Document, float]],
    max_tokens_per_doc: int = 300
) -> str:
    """Format documents for reranking prompt."""
    formatted = []
    for i, (doc, score) in enumerate(docs_with_scores):
        if doc is None:
            continue

        # Truncate text
        text = doc.full_text
        if len(text) > max_tokens_per_doc * 2:
            text = text[:max_tokens_per_doc * 2] + "..."

        formatted.append(f"[{i + 1}]. {text}")

    return "\n\n".join(formatted)


def parse_rerank_response(response: str, num_docs: int) -> List[int]:
    """
    Parse reranking response to extract document order.

    Args:
        response: LLM response text
        num_docs: Number of documents being reranked

    Returns:
        List of 0-indexed document positions in ranked order
    """
    # Try to extract JSON from response
    try:
        # Look for JSON code block
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            # Try to find array pattern
            match = re.search(r'\[[\d,\s]+\]', response)
            if match:
                json_str = match.group()
            else:
                json_str = response.strip()

        ranking = json.loads(json_str)

        # Convert to 0-indexed
        ranking = [int(x) - 1 for x in ranking]

        # Validate
        if len(ranking) != num_docs:
            print(f"Warning: Expected {num_docs} items, got {len(ranking)}")

        # Ensure all indices are valid
        ranking = [x for x in ranking if 0 <= x < num_docs]

        # Add any missing indices at the end
        missing = [i for i in range(num_docs) if i not in ranking]
        ranking.extend(missing)

        return ranking[:num_docs]

    except (json.JSONDecodeError, IndexError, ValueError) as e:
        print(f"Failed to parse rerank response: {e}")
        print(f"Response: {response[:500]}")
        # Return original order
        return list(range(num_docs))


def rerank_documents(
    openai_client: OpenAIClient,
    query_id: str,
    query_text: str,
    docs_with_scores: List[Tuple[Document, float]],
    top_k: int = 20
) -> RerankResult:
    """
    Rerank documents using LLM.

    Args:
        openai_client: OpenAI client
        query_id: Query identifier
        query_text: Query text
        docs_with_scores: List of (Document, score) tuples from retrieval
        top_k: Number of top documents to rerank

    Returns:
        RerankResult with new scores
    """
    # Limit to top_k documents
    docs_to_rerank = docs_with_scores[:top_k]

    # Format documents for prompt
    docs_text = format_documents_for_rerank(docs_to_rerank)

    prompt = RERANK_PROMPT.format(
        query=query_text,
        documents=docs_text
    )

    try:
        response = openai_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        # Parse ranking
        ranking = parse_rerank_response(response, len(docs_to_rerank))

        # Assign new scores based on ranking position
        reranked_scores = {}
        for rank_position, doc_idx in enumerate(ranking):
            doc, _ = docs_to_rerank[doc_idx]
            if doc:
                # Higher score for higher rank
                reranked_scores[doc.doc_id] = top_k - rank_position

        return RerankResult(
            query_id=query_id,
            query_text=query_text,
            reranked_doc_scores=reranked_scores
        )

    except Exception as e:
        print(f"Reranking failed for query {query_id}: {e}")
        # Return original scores
        return RerankResult(
            query_id=query_id,
            query_text=query_text,
            reranked_doc_scores={
                doc.doc_id: score
                for doc, score in docs_to_rerank
                if doc is not None
            }
        )


def rerank_all(
    retrieval_results: List[RetrievalResult],
    documents: List[Document],
    openai_client: OpenAIClient,
    top_k: int = 20
) -> List[RerankResult]:
    """
    Rerank all retrieval results.

    Args:
        retrieval_results: List of retrieval results
        documents: List of all documents
        openai_client: OpenAI client
        top_k: Number of documents to rerank per query

    Returns:
        List of RerankResult objects
    """
    # Create document index
    doc_index = {d.doc_id: d for d in documents}

    reranked_results = []

    for result in retrieval_results:
        print(f"Reranking query {result.query_id}: {result.query_text[:50]}...")

        # Get top documents with scores
        sorted_docs = sorted(result.doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        docs_with_scores = [
            (doc_index.get(doc_id), score)
            for doc_id, score in sorted_docs
        ]

        # Rerank
        reranked = rerank_documents(
            openai_client,
            result.query_id,
            result.query_text,
            docs_with_scores,
            top_k=top_k
        )

        reranked_results.append(reranked)

    return reranked_results


def rerank_results_to_dict(results: List[RerankResult]) -> Dict[str, Dict[str, float]]:
    """Convert rerank results to dict format."""
    return {
        r.query_id: {str(k): v for k, v in r.reranked_doc_scores.items()}
        for r in results
    }


if __name__ == "__main__":
    import json as json_module
    from data_loader import load_medical_documents
    from retriever import HybridRetriever

    # Load config
    with open("config.json") as f:
        config = json_module.load(f)

    # Load documents
    docs = load_medical_documents(
        config["s3_input_path"],
        local_cache_path="cache/medical_literature.csv",
        max_documents=500
    )

    # Initialize retriever
    retriever = HybridRetriever(
        documents=docs,
        model_path=config["retriever_model"],
        cache_dir="cache",
        bm25_alpha=config["bm25_alpha"]
    )

    # Initialize OpenAI client
    openai_client = OpenAIClient(
        api_key=config["openai_api_key"],
        model=config["openai_model"]
    )

    # Test retrieval + reranking
    test_queries = [("0", "华氏巨球蛋白血症 治疗")]
    results = retriever.retrieve(test_queries, top_k=20)

    # Rerank
    reranked = rerank_all(results, docs, openai_client, top_k=10)

    for r in reranked:
        print(f"\nQuery {r.query_id}: {r.query_text}")
        sorted_scores = sorted(r.reranked_doc_scores.items(), key=lambda x: x[1], reverse=True)
        for doc_id, score in sorted_scores[:5]:
            doc = docs[doc_id] if doc_id < len(docs) else None
            if doc:
                print(f"  [{score}] {doc.title[:50]}...")
