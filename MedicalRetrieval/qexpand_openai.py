"""
Query expansion using LLM (OpenAI or Bedrock).
Implements iterative retrieval-augmented query expansion.
"""
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

from utils import OpenAIClient, BedrockClient
from data_loader import Document
from retriever import HybridRetriever, RetrievalResult

# Type alias for LLM client
LLMClient = Union[OpenAIClient, BedrockClient]


@dataclass
class ExpandedQuery:
    """Represents an expanded query."""
    query_id: str
    original_query: str
    expanded_query: str
    expansion_rounds: List[str]  # History of expansions


EXPANSION_PROMPT_INITIAL = """你是一个医学文献检索专家。给定一个医学查询和相关的检索文档，请扩展该查询以提高检索效果。

原始查询: {query}

检索到的相关文档:
{documents}

请基于这些文档中的信息，生成一个扩展后的查询。扩展应该：
1. 保留原始查询的核心意图
2. 添加相关的医学术语、同义词或相关概念
3. 使用中文
4. 保持简洁（不超过50个字）

只输出扩展后的查询，不要包含任何解释。"""


EXPANSION_PROMPT_ITERATIVE = """你是一个医学文献检索专家。给定一个医学查询、之前的扩展结果和新检索到的文档，请进一步优化查询。

原始查询: {query}

上一轮扩展结果: {previous_expansion}

新检索到的相关文档:
{documents}

请基于新文档中的信息，进一步优化查询。优化应该：
1. 保留原始查询的核心意图
2. 整合新发现的相关术语和概念
3. 移除可能导致检索偏差的词语
4. 使用中文
5. 保持简洁（不超过50个字）

只输出优化后的查询，不要包含任何解释。"""


def truncate_document(doc: Document, max_tokens: int = 200) -> str:
    """Truncate document text for prompt."""
    text = doc.full_text
    # Simple character-based truncation (rough approximation)
    if len(text) > max_tokens * 2:
        text = text[:max_tokens * 2] + "..."
    return text


def format_documents_for_prompt(
    docs_with_scores: List[Tuple[Document, float]],
    max_docs: int = 5,
    max_tokens_per_doc: int = 200
) -> str:
    """Format documents for inclusion in prompt."""
    formatted = []
    for i, (doc, score) in enumerate(docs_with_scores[:max_docs]):
        if doc is None:
            continue
        text = truncate_document(doc, max_tokens_per_doc)
        formatted.append(f"[{i + 1}]. {text}")
    return "\n\n".join(formatted)


def expand_query_single_round(
    llm_client: LLMClient,
    query_text: str,
    retrieved_docs: List[Tuple[Document, float]],
    previous_expansion: Optional[str] = None
) -> str:
    """
    Expand a query using retrieved documents.

    Args:
        llm_client: LLM client (OpenAIClient or BedrockClient)
        query_text: Original query text
        retrieved_docs: List of (Document, score) tuples
        previous_expansion: Previous expansion result (for iterative expansion)

    Returns:
        Expanded query text
    """
    docs_text = format_documents_for_prompt(retrieved_docs)

    if previous_expansion:
        prompt = EXPANSION_PROMPT_ITERATIVE.format(
            query=query_text,
            previous_expansion=previous_expansion,
            documents=docs_text
        )
    else:
        prompt = EXPANSION_PROMPT_INITIAL.format(
            query=query_text,
            documents=docs_text
        )

    response = llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=200,
        n=1
    )

    # Handle both OpenAI (str) and Bedrock (List[str]) response formats
    if isinstance(response, list):
        result = response[0] if response else ""
    else:
        result = response

    return result.strip().strip('"\'')


def expand_queries_iterative(
    queries: List[Tuple[str, str]],
    retriever: HybridRetriever,
    llm_client: LLMClient,
    num_rounds: int = 3,
    docs_per_round: int = 10
) -> List[ExpandedQuery]:
    """
    Expand queries using iterative retrieval-augmented expansion.

    Args:
        queries: List of (query_id, query_text) tuples
        retriever: Hybrid retriever for document retrieval
        llm_client: LLM client for expansion (OpenAIClient or BedrockClient)
        num_rounds: Number of expansion rounds
        docs_per_round: Number of documents to use per round

    Returns:
        List of ExpandedQuery objects
    """
    expanded_queries = []

    for qid, qtext in queries:
        print(f"\nExpanding query {qid}: {qtext}")

        current_query = qtext
        expansion_history = []

        for round_idx in range(num_rounds):
            print(f"  Round {round_idx + 1}/{num_rounds}")

            # Retrieve documents using current query
            results = retriever.retrieve([(qid, current_query)], top_k=docs_per_round)
            if not results:
                print(f"    No results, skipping round")
                continue

            # Get top documents
            top_docs = retriever.get_top_documents(results[0], top_k=docs_per_round)

            # Expand query
            previous = expansion_history[-1] if expansion_history else None
            expanded = expand_query_single_round(
                llm_client,
                qtext,  # Always use original query as reference
                top_docs,
                previous_expansion=previous
            )

            expansion_history.append(expanded)
            current_query = f"{qtext}\n\n{expanded}"  # Concatenate for retrieval

            print(f"    Expanded: {expanded}")

        # Create final expanded query
        final_expansion = expansion_history[-1] if expansion_history else qtext
        final_query = f"{qtext}\n\n{final_expansion}"

        expanded_queries.append(ExpandedQuery(
            query_id=qid,
            original_query=qtext,
            expanded_query=final_query,
            expansion_rounds=expansion_history
        ))

    return expanded_queries


def expanded_queries_to_list(expanded: List[ExpandedQuery]) -> List[Tuple[str, str]]:
    """Convert expanded queries to (query_id, query_text) tuples."""
    return [(eq.query_id, eq.expanded_query) for eq in expanded]


def expanded_queries_to_dict(expanded: List[ExpandedQuery]) -> Dict[str, Dict]:
    """Convert expanded queries to dict for serialization."""
    return {
        eq.query_id: {
            "query_id": eq.query_id,
            "original_query": eq.original_query,
            "expanded_query": eq.expanded_query,
            "expansion_rounds": eq.expansion_rounds
        }
        for eq in expanded
    }


if __name__ == "__main__":
    import json
    from data_loader import load_medical_documents

    # Load config
    with open("config.json") as f:
        config = json.load(f)

    # Load documents (small subset for testing)
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

    # Test expansion
    test_queries = [
        ("0", "华氏巨球蛋白血症 治疗")
    ]

    expanded = expand_queries_iterative(
        test_queries,
        retriever,
        openai_client,
        num_rounds=2
    )

    for eq in expanded:
        print(f"\nQuery {eq.query_id}")
        print(f"  Original: {eq.original_query}")
        print(f"  Expanded: {eq.expanded_query}")
        print(f"  Rounds: {eq.expansion_rounds}")
