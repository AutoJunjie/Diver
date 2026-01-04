"""
LLM-as-a-Judge evaluation for relevance assessment.
Uses 3-point scale: Not relevant (0), Partially relevant (1), Highly relevant (2).
"""
import json
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

from utils import OpenAIClient, BedrockClient, compute_llm_judge_metrics
from data_loader import Document
from reranker_openai import RerankResult

# Type alias for LLM client
LLMClient = Union[OpenAIClient, BedrockClient]


@dataclass
class RelevanceJudgment:
    """Represents a single relevance judgment."""
    query_id: str
    query_text: str  # Original query before expansion
    expanded_query: str  # Query after expansion/rewrite
    doc_id: int
    doc_title: str
    relevance_score: int  # 0, 1, or 2
    rank: int  # Position in reranked results
    retrieval_score: float
    rerank_score: float
    explanation: Optional[str] = None


JUDGE_PROMPT = """你是一个医学文献检索评估专家。请评估以下文档与给定查询的相关性。

查询: {query}

文档:
标题: {title}
摘要: {abstract}
关键词: {keywords}

请根据以下标准评估相关性（仅考虑主题相关性）:

0 - 不相关: 文档与查询的主题无关或仅有很弱的关联
1 - 部分相关: 文档与查询有一定关联，但不能完全满足查询需求
2 - 高度相关: 文档直接回答或高度符合查询需求

请以JSON格式输出评估结果:
```json
{{"score": <0或1或2>, "explanation": "<简短解释>"}}
```"""


def parse_judgment_response(response: str) -> Tuple[int, str]:
    """
    Parse judgment response to extract score and explanation.

    Returns:
        Tuple of (score, explanation)
    """
    try:
        # Try to extract JSON
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            # Try to find JSON object
            import re
            match = re.search(r'\{[^}]+\}', response)
            if match:
                json_str = match.group()
            else:
                json_str = response.strip()

        result = json.loads(json_str)
        score = int(result.get("score", 0))
        explanation = result.get("explanation", "")

        # Validate score
        if score not in [0, 1, 2]:
            print(f"Invalid score {score}, defaulting to 0")
            score = 0

        return score, explanation

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Failed to parse judgment: {e}")
        print(f"Response: {response[:300]}")
        return 0, "Failed to parse"


def judge_single_document(
    llm_client: LLMClient,
    query_text: str,
    document: Document
) -> Tuple[int, str]:
    """
    Judge relevance of a single document to a query.

    Args:
        llm_client: LLM client (OpenAIClient or BedrockClient)
        query_text: Query text
        document: Document to judge

    Returns:
        Tuple of (relevance_score, explanation)
    """
    keywords_str = ", ".join(document.keywords) if document.keywords else "无"

    prompt = JUDGE_PROMPT.format(
        query=query_text,
        title=document.title,
        abstract=document.abstract[:500] + "..." if len(document.abstract) > 500 else document.abstract,
        keywords=keywords_str
    )

    response = llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
        n=1
    )

    # Handle both OpenAI (str) and Bedrock (List[str]) response formats
    if isinstance(response, list):
        response = response[0] if response else ""

    return parse_judgment_response(response)


def evaluate_reranked_results(
    reranked_results: List[RerankResult],
    retrieval_scores: Dict[str, Dict[int, float]],
    documents: List[Document],
    llm_client: LLMClient,
    top_k: int = 10,
    original_queries: Optional[Dict[str, str]] = None
) -> List[RelevanceJudgment]:
    """
    Evaluate reranked results using LLM judge.

    Args:
        reranked_results: List of reranked results
        retrieval_scores: Original retrieval scores {query_id: {doc_id: score}}
        documents: List of all documents
        llm_client: LLM client (OpenAIClient or BedrockClient)
        top_k: Number of top documents to evaluate per query
        original_queries: Optional dict mapping query_id to original query text

    Returns:
        List of RelevanceJudgment objects
    """
    # Create document index
    doc_index = {d.doc_id: d for d in documents}

    # Default to empty dict if not provided
    original_queries = original_queries or {}

    all_judgments = []

    for result in reranked_results:
        print(f"\nEvaluating query {result.query_id}: {result.query_text[:50]}...")

        # Get original query (before expansion)
        original_query = original_queries.get(result.query_id, result.query_text)

        # Get top-k documents by rerank score
        sorted_docs = sorted(
            result.reranked_doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Get retrieval scores for this query
        query_retrieval_scores = retrieval_scores.get(result.query_id, {})

        for rank, (doc_id, rerank_score) in enumerate(sorted_docs):
            doc = doc_index.get(doc_id)
            if doc is None:
                print(f"  Document {doc_id} not found, skipping")
                continue

            # Get retrieval score
            retrieval_score = query_retrieval_scores.get(str(doc_id), 0.0)

            # Judge relevance
            relevance_score, explanation = judge_single_document(
                llm_client,
                result.query_text,
                doc
            )

            judgment = RelevanceJudgment(
                query_id=result.query_id,
                query_text=original_query,
                expanded_query=result.query_text,
                doc_id=doc_id,
                doc_title=doc.title,
                relevance_score=relevance_score,
                rank=rank,
                retrieval_score=retrieval_score,
                rerank_score=rerank_score,
                explanation=explanation
            )

            all_judgments.append(judgment)
            print(f"  [{rank + 1}] Doc {doc_id}: score={relevance_score} ({doc.title[:30]}...)")

    return all_judgments


def judgments_to_list(judgments: List[RelevanceJudgment]) -> List[Dict]:
    """Convert judgments to list of dicts for metrics computation."""
    return [
        {
            "query_id": j.query_id,
            "doc_id": j.doc_id,
            "relevance_score": j.relevance_score,
            "rank": j.rank
        }
        for j in judgments
    ]


def compute_evaluation_metrics(judgments: List[RelevanceJudgment]) -> Dict:
    """Compute evaluation metrics from judgments."""
    judgment_dicts = judgments_to_list(judgments)
    return compute_llm_judge_metrics(judgment_dicts, k_values=[1, 5, 10])


def identify_zero_relevance_queries(judgments: List[RelevanceJudgment]) -> List[str]:
    """Identify queries where all retrieved documents were rated 0 (not relevant)."""
    from collections import defaultdict

    query_scores = defaultdict(list)
    for j in judgments:
        query_scores[j.query_id].append(j.relevance_score)

    zero_relevance = [
        qid for qid, scores in query_scores.items()
        if all(s == 0 for s in scores)
    ]

    return zero_relevance


if __name__ == "__main__":
    import json as json_module
    from data_loader import load_medical_documents
    from retriever import HybridRetriever, results_to_dict
    from reranker_openai import rerank_all

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

    # Test queries
    test_queries = [("0", "华氏巨球蛋白血症 治疗")]

    # Retrieve
    results = retriever.retrieve(test_queries, top_k=20)
    retrieval_scores = results_to_dict(results)

    # Rerank
    reranked = rerank_all(results, docs, openai_client, top_k=10)

    # Evaluate
    judgments = evaluate_reranked_results(
        reranked,
        retrieval_scores,
        docs,
        openai_client,
        top_k=5
    )

    # Compute metrics
    metrics = compute_evaluation_metrics(judgments)
    print(f"\nMetrics: {json_module.dumps(metrics, indent=2, ensure_ascii=False)}")

    # Check for zero-relevance queries
    zero_rel = identify_zero_relevance_queries(judgments)
    if zero_rel:
        print(f"\nWarning: Zero-relevance queries: {zero_rel}")
