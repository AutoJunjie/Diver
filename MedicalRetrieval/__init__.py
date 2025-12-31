"""
MedicalRetrieval - Medical Literature Retrieval Pipeline

A production retrieval system for Chinese medical literature using:
- DIVER-Retriever-4B for dense embeddings
- BM25 for sparse retrieval
- GPT-4o-mini for query expansion, reranking, and LLM-as-judge evaluation
"""

__version__ = "1.0.0"
