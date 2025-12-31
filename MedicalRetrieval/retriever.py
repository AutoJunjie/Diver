"""
Retriever module using DIVER-Retriever-4B and BM25 hybrid search.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm, trange
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
import jieba

from data_loader import Document


@dataclass
class RetrievalResult:
    """Represents retrieval results for a query."""
    query_id: str
    query_text: str
    doc_scores: Dict[int, float]  # doc_id -> score


class DiverEmbeddingModel:
    """DIVER-Retriever embedding model."""

    def __init__(self, model_path: str, device: str = "auto"):
        print(f"Loading DIVER model from {model_path}...")
        self.model = AutoModel.from_pretrained(
            model_path,
            attn_implementation="eager",  # Use eager attention (flash_attention_2 not installed)
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.task = "Given a web search query, retrieve relevant passages that answer the query"
        print("DIVER model loaded")

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    def encode(self, texts: List[str], max_length: int = 16384, batch_size: int = 1) -> np.ndarray:
        """Encode texts to embeddings."""
        all_embeddings = []

        for i in trange(0, len(texts), batch_size, desc="Encoding"):
            batch_texts = texts[i:i + batch_size]

            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}

            with torch.inference_mode():
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(
                    outputs.last_hidden_state,
                    batch_dict['attention_mask']
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

            torch.cuda.empty_cache()

        return np.vstack(all_embeddings)

    def encode_queries(self, queries: List[str], max_length: int = 8192) -> np.ndarray:
        """Encode queries with instruction prefix."""
        instructed_queries = [
            self.get_detailed_instruct(self.task, q) for q in queries
        ]
        return self.encode(instructed_queries, max_length=max_length)

    def encode_documents(self, documents: List[str], max_length: int = 16384, batch_size: int = 1) -> np.ndarray:
        """Encode documents."""
        return self.encode(documents, max_length=max_length, batch_size=batch_size)


class BM25Retriever:
    """BM25 retriever for Chinese text using jieba tokenization."""

    def __init__(self, documents: List[str]):
        print("Building BM25 index...")
        # Tokenize documents with jieba
        self.tokenized_docs = [list(jieba.cut(doc)) for doc in tqdm(documents, desc="Tokenizing")]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print("BM25 index built")

    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for all documents given a query."""
        tokenized_query = list(jieba.cut(query))
        return self.bm25.get_scores(tokenized_query)


class HybridRetriever:
    """Hybrid retriever combining DIVER dense retrieval and BM25 sparse retrieval."""

    def __init__(
        self,
        documents: List[Document],
        model_path: str,
        cache_dir: str = "cache",
        bm25_alpha: float = 0.3
    ):
        self.documents = documents
        self.doc_ids = [d.doc_id for d in documents]
        self.doc_texts = [d.full_text for d in documents]
        self.cache_dir = cache_dir
        self.bm25_alpha = bm25_alpha

        # Initialize models
        self.diver_model = DiverEmbeddingModel(model_path)
        self.bm25_retriever = BM25Retriever(self.doc_texts)

        # Cache paths
        os.makedirs(cache_dir, exist_ok=True)
        self.doc_emb_cache_path = os.path.join(cache_dir, "medical_doc_embeddings.npy")

        # Load or compute document embeddings
        self.doc_embeddings = self._get_doc_embeddings()

    def _get_doc_embeddings(self) -> np.ndarray:
        """Get document embeddings, using cache if available."""
        if os.path.exists(self.doc_emb_cache_path):
            print(f"Loading cached document embeddings from {self.doc_emb_cache_path}")
            return np.load(self.doc_emb_cache_path)

        print("Computing document embeddings...")
        embeddings = self.diver_model.encode_documents(self.doc_texts, batch_size=1)
        np.save(self.doc_emb_cache_path, embeddings)
        print(f"Document embeddings cached to {self.doc_emb_cache_path}")
        return embeddings

    def retrieve(
        self,
        queries: List[Tuple[str, str]],
        top_k: int = 100
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for queries using hybrid search.

        Args:
            queries: List of (query_id, query_text) tuples
            top_k: Number of top documents to return

        Returns:
            List of RetrievalResult objects
        """
        query_ids = [q[0] for q in queries]
        query_texts = [q[1] for q in queries]

        print(f"Encoding {len(queries)} queries...")
        query_embeddings = self.diver_model.encode_queries(query_texts)

        print("Computing dense similarity scores...")
        dense_scores = cosine_similarity(query_embeddings, self.doc_embeddings)

        results = []

        for idx, (qid, qtext) in enumerate(tqdm(queries, desc="Hybrid retrieval")):
            # Get dense scores
            dense_doc_scores = dense_scores[idx]

            # Get BM25 scores
            bm25_scores = np.array(self.bm25_retriever.get_scores(qtext))

            # Normalize scores to [0, 1]
            if dense_doc_scores.max() > dense_doc_scores.min():
                dense_norm = (dense_doc_scores - dense_doc_scores.min()) / (dense_doc_scores.max() - dense_doc_scores.min())
            else:
                dense_norm = dense_doc_scores

            if bm25_scores.max() > bm25_scores.min():
                bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
            else:
                bm25_norm = bm25_scores

            # Combine scores: hybrid = (1 - alpha) * dense + alpha * bm25
            hybrid_scores = (1 - self.bm25_alpha) * dense_norm + self.bm25_alpha * bm25_norm

            # Get top-k documents
            top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

            doc_scores = {
                self.doc_ids[i]: float(hybrid_scores[i])
                for i in top_indices
            }

            results.append(RetrievalResult(
                query_id=qid,
                query_text=qtext,
                doc_scores=doc_scores
            ))

        return results

    def get_document_by_id(self, doc_id: int) -> Optional[Document]:
        """Get document by ID."""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None

    def get_top_documents(
        self,
        result: RetrievalResult,
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """Get top-k documents with scores from a retrieval result."""
        sorted_docs = sorted(result.doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (self.get_document_by_id(doc_id), score)
            for doc_id, score in sorted_docs
        ]


def results_to_dict(results: List[RetrievalResult]) -> Dict[str, Dict[str, float]]:
    """Convert retrieval results to dict format for serialization."""
    return {
        r.query_id: {str(k): v for k, v in r.doc_scores.items()}
        for r in results
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
        max_documents=100
    )

    # Initialize retriever
    retriever = HybridRetriever(
        documents=docs,
        model_path=config["retriever_model"],
        cache_dir="cache",
        bm25_alpha=config["bm25_alpha"]
    )

    # Test retrieval
    test_queries = [
        ("0", "华氏巨球蛋白血症 治疗"),
        ("1", "糖尿病 并发症")
    ]

    results = retriever.retrieve(test_queries, top_k=10)

    for r in results:
        print(f"\nQuery {r.query_id}: {r.query_text}")
        top_docs = retriever.get_top_documents(r, top_k=3)
        for doc, score in top_docs:
            if doc:
                print(f"  [{score:.4f}] {doc.title[:50]}...")
