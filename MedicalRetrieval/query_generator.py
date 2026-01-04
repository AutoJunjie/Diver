"""
Synthetic query generator for medical literature.
Clusters documents by keywords and generates Chinese keyword-style queries.
"""
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

from data_loader import Document
from utils import OpenAIClient, BedrockClient, cluster_documents_by_keywords, select_diverse_clusters

# Type alias for LLM client (can be OpenAIClient or BedrockClient)
LLMClient = OpenAIClient | BedrockClient


@dataclass
class Query:
    """Represents a generated query."""
    query_id: int
    query_text: str
    source_keyword: str
    source_doc_ids: List[int]


QUERY_GENERATION_PROMPT = """你是一个医学文献检索专家。基于以下医学文献的关键词和摘要，生成一个简短的中文关键词式检索查询。

关键词主题: {keyword}

相关文献样本:
{doc_samples}

请生成一个简短的关键词式查询（2-5个词），类似于用户在医学数据库中搜索时会输入的内容。
查询应该：
1. 使用中文
2. 简洁，像搜索关键词（例如："华氏巨球蛋白血症 治疗方案" 或 "糖尿病 并发症 预防"）
3. 与该主题的文献相关

只输出查询本身，不要包含任何解释或额外文字。"""


def generate_queries(
    documents: List[Document],
    llm_client: LLMClient,
    num_queries: int = 20,
    min_cluster_size: int = 10,
    docs_per_sample: int = 3
) -> List[Query]:
    """
    Generate synthetic queries by clustering documents and sampling from clusters.

    Args:
        documents: List of Document objects
        llm_client: LLM client for query generation (OpenAIClient or BedrockClient)
        num_queries: Number of queries to generate
        min_cluster_size: Minimum documents per keyword cluster
        docs_per_sample: Number of sample documents to show in prompt

    Returns:
        List of Query objects
    """
    print(f"Clustering {len(documents)} documents by keywords...")

    # Cluster documents by keywords
    clusters = cluster_documents_by_keywords(documents, min_cluster_size=min_cluster_size)
    print(f"Found {len(clusters)} keyword clusters with >= {min_cluster_size} documents")

    if len(clusters) < num_queries:
        print(f"Warning: Only {len(clusters)} clusters available, adjusting target to {len(clusters)}")
        num_queries = len(clusters)

    # Select diverse clusters
    selected_keywords = select_diverse_clusters(clusters, num_queries)
    print(f"Selected {len(selected_keywords)} diverse clusters")

    # Create document index for quick lookup
    doc_index = {d.doc_id: d for d in documents}

    queries = []

    for idx, keyword in enumerate(selected_keywords):
        print(f"Generating query {idx + 1}/{len(selected_keywords)} for keyword: {keyword}")

        doc_ids = clusters[keyword]

        # Sample documents for the prompt
        sample_ids = random.sample(doc_ids, min(docs_per_sample, len(doc_ids)))
        sample_docs = [doc_index[did] for did in sample_ids]

        # Format document samples
        doc_samples = "\n\n".join([
            f"标题: {doc.title}\n摘要: {doc.abstract[:300]}..."
            for doc in sample_docs
        ])

        # Generate query using LLM
        prompt = QUERY_GENERATION_PROMPT.format(
            keyword=keyword,
            doc_samples=doc_samples
        )

        try:
            response = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100,
                n=1  # Only need one response for query generation
            )

            # Handle both OpenAI (str) and Bedrock (List[str]) response formats
            if isinstance(response, list):
                query_text = response[0].strip() if response else ""
            else:
                query_text = response.strip()

            # Clean up query text (remove quotes if present)
            query_text = query_text.strip('"\'')

            query = Query(
                query_id=idx,
                query_text=query_text,
                source_keyword=keyword,
                source_doc_ids=doc_ids
            )
            queries.append(query)
            print(f"  Generated: {query_text}")

        except Exception as e:
            print(f"  Failed to generate query for {keyword}: {e}")
            # Fallback: use keyword as query
            query = Query(
                query_id=idx,
                query_text=keyword,
                source_keyword=keyword,
                source_doc_ids=doc_ids
            )
            queries.append(query)
            print(f"  Fallback: {keyword}")

    print(f"Generated {len(queries)} queries")
    return queries


def queries_to_list(queries: List[Query]) -> List[Tuple[str, str]]:
    """Convert Query objects to (query_id, query_text) tuples."""
    return [(str(q.query_id), q.query_text) for q in queries]


def queries_to_dict(queries: List[Query]) -> Dict[str, Dict]:
    """Convert Query objects to dict for serialization."""
    return {
        str(q.query_id): {
            "query_id": q.query_id,
            "query_text": q.query_text,
            "source_keyword": q.source_keyword,
            "source_doc_ids": q.source_doc_ids[:10]  # Limit for readability
        }
        for q in queries
    }


if __name__ == "__main__":
    import json
    from data_loader import load_medical_documents

    # Load config
    with open("config.json") as f:
        config = json.load(f)

    # Load documents
    docs = load_medical_documents(
        config["s3_input_path"],
        local_cache_path="cache/medical_literature.csv",
        max_documents=1000  # Use subset for testing
    )

    # Initialize OpenAI client
    client = OpenAIClient(
        api_key=config["openai_api_key"],
        model=config["openai_model"]
    )

    # Generate queries
    queries = generate_queries(docs, client, num_queries=5)

    # Print results
    for q in queries:
        print(f"Query {q.query_id}: {q.query_text} (from keyword: {q.source_keyword})")
