# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DIVER is a multi-stage reasoning-intensive information retrieval system for complex queries requiring abstract or multi-step reasoning. It achieves state-of-the-art results (NDCG 45.8) on the BRIGHT benchmark.

## Architecture

The pipeline consists of four stages executed sequentially:

1. **Query Expansion (QExpand/)** - LLM-driven iterative query refinement using DeepSeek-R1-Distill-Qwen-14B
2. **Retrieval (Retriever/)** - Embedding-based retrieval (DIVER-Retriever) combined with BM25 hybrid search
3. **Reranking (Reranker/)** - Three approaches: pointwise, listwise (Gemini API), and groupwise (DIVER-GroupRank)
4. **Evaluation** - NDCG@10 metrics via pytrec_eval against BRIGHT benchmark

## Commands

```bash
# Install dependencies
pip install -r env_requirements/requirements.txt

# Run full pipeline (downloads data/models first)
sh run_all.sh

# Run individual stages
cd QExpand && bash run_qexpand.sh      # Query expansion
cd Retriever && bash retriever_script.sh  # Retrieval
cd Retriever && python merge_scores.py    # Merge BM25 + DIVER scores
cd Reranker && bash reranker_script.sh    # Reranking
```

## Key Files

- `QExpand/qexpand_main.py` - Query expansion entry point
- `Retriever/run.py` - Retrieval entry point
- `Retriever/retrievers.py` - Retriever implementations (DIVER, BM25, hybrid)
- `Reranker/rerank_pointwise.py` - Pointwise reranking with local LLM
- `Reranker/rerank_listwise.py` - Listwise reranking with Gemini API
- `Reranker/rerank_groupwise.py` - Groupwise reranking (DIVER-GroupRank)
- `utils/eval_util.py` - NDCG evaluation functions

## Models

Download from HuggingFace/ModelScope before running:
- `AQ-MedAI/Diver-Retriever-4B` (also 0.6B, 1.7B variants)
- `AQ-MedAI/Diver-GroupRank-7B` or `Diver-GroupRank-32B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (for QExpand)
- `Qwen/Qwen2.5-32B-Instruct` (for pointwise reranking)

## BRIGHT Datasets

Tasks: `biology`, `earth_science`, `economics`, `psychology`, `robotics`, `stackoverflow`, `sustainable_living`, `leetcode`, `pony`, `aops`, `theoremqa_theorems`, `theoremqa_questions`

Data location: `./data/BRIGHT/` (clone from `xlangai/BRIGHT`)

## Special Setup Requirements

**JDK 21** - Required by pyserini for BM25 (Lucene-based). Set `JAVA_HOME` appropriately.

**pytrec_eval** - Manual installation required; see `env_requirements/README.md` for instructions using the bundled source packages.

## Fine-tuning

Use the [SWIFT framework](https://github.com/modelscope/ms-swift) for fine-tuning retrievers with infonce loss. Training command example in README.md.

## External Dependencies

### System Dependencies
| Dependency | Purpose | Notes |
|------------|---------|-------|
| **JDK 21** | BM25 indexing (Lucene via pyserini) | Set `JAVA_HOME` |
| **CUDA Toolkit** | GPU acceleration | Required for vLLM, PyTorch |
| **Flash Attention 2** | Optimized attention computation | Optional but recommended |

### Python Package Dependencies

**Core ML/DL Stack:**
- `torch` - PyTorch deep learning framework
- `transformers` - HuggingFace model library
- `sentence-transformers` - Embedding models
- `vllm` - High-performance LLM inference engine
- `datasets` - HuggingFace dataset loading

**IR & NLP:**
- `pyserini` - BM25 search (Lucene-based)
- `gensim` - BM25 model implementation
- `tiktoken` - OpenAI tokenizer

**Scientific Computing:**
- `numpy`, `scipy` - Numerical computation
- `scikit-learn` - Cosine similarity, metrics
- `torchmetrics` - PyTorch metrics

**Evaluation:**
- `pytrec_eval` - IR evaluation metrics (manual install required)

**API Clients:**
- `openai` - OpenAI SDK (also for Gemini API via OpenAI-compatible endpoint)

**Optional Retriever APIs:**
- `cohere` - Cohere Embed API
- `voyageai` - Voyage AI API
- `vertexai` - Google Vertex AI
- `gritlm` - GritLM models

### External Services & APIs

| Service | Used In | Purpose |
|---------|---------|---------|
| **Google Gemini API** | Listwise Reranker | LLM-based document reranking |
| **HuggingFace Hub** | All stages | Model & dataset downloads |
| **ModelScope** | All stages | Model downloads (CN mirror) |
| **OpenAI API** | Retriever (optional) | text-embedding-3-large |
| **Cohere API** | Retriever (optional) | embed-english-v3.0 |
| **Voyage AI API** | Retriever (optional) | voyage-large-2-instruct |
| **Google Vertex AI** | Retriever (optional) | text-embedding-preview |

### External Models (Download Required)

| Model | Stage | Source |
|-------|-------|--------|
| `AQ-MedAI/Diver-Retriever-4B` | Retrieval | HuggingFace/ModelScope |
| `AQ-MedAI/Diver-GroupRank-7B/32B` | Groupwise Reranking | HuggingFace/ModelScope |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | Query Expansion | HuggingFace |
| `Qwen/Qwen2.5-32B-Instruct` | Pointwise Reranking | HuggingFace |

### External Data

| Dataset | Source | Purpose |
|---------|--------|---------|
| **BRIGHT Benchmark** | `xlangai/BRIGHT` | Evaluation queries & documents |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DIVER Pipeline Architecture                        │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │    Input     │
                              │    Query     │
                              └──────┬───────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Query Expansion (QExpand/)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DeepSeek-R1-Distill-Qwen-14B (via vLLM)                            │   │
│  │  • Iterative query refinement (3 rounds)                            │   │
│  │  • Retrieval-augmented expansion                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                              Expanded Query
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Retrieval (Retriever/)                                            │
│  ┌────────────────────────┐    ┌────────────────────────┐                  │
│  │   DIVER-Retriever-4B   │    │        BM25            │                  │
│  │   (Dense Embeddings)   │    │   (Sparse/Lexical)     │                  │
│  │                        │    │   via pyserini/gensim  │                  │
│  │   ┌──────────────┐     │    │                        │                  │
│  │   │    vLLM      │     │    │   ┌──────────────┐     │                  │
│  │   │  Embedding   │     │    │   │   Lucene     │     │                  │
│  │   └──────────────┘     │    │   │   (JDK 21)   │     │                  │
│  └───────────┬────────────┘    └─────────┬──────────────┘                  │
│              │                           │                                  │
│              └───────────┬───────────────┘                                  │
│                          ▼                                                  │
│                  ┌──────────────┐                                           │
│                  │ Hybrid Merge │  (merge_scores.py)                        │
│                  │   α=0.3      │                                           │
│                  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                            Top-K Candidates
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 3: Reranking (Reranker/)                                             │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │   Pointwise     │  │    Listwise     │  │        Groupwise            │ │
│  │                 │  │                 │  │    (DIVER-GroupRank)        │ │
│  │  Qwen2.5-32B    │  │   Gemini API    │  │                             │ │
│  │  (Local vLLM)   │  │   (External)    │  │   Diver-GroupRank-7B/32B    │ │
│  │                 │  │                 │  │       (Local vLLM)          │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘ │
│           │                    │                          │                 │
│           └────────────────────┼──────────────────────────┘                 │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │    Score Fusion       │                                │
│                    │  (Optional Merging)   │                                │
│                    └───────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                             Reranked Results
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 4: Evaluation                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  pytrec_eval                                                        │   │
│  │  • NDCG@10 metrics                                                  │   │
│  │  • Against BRIGHT benchmark ground truth                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         External Dependencies                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │ HuggingFace │  │  ModelScope │  │ Gemini API  │  │   JDK 21    │       │
│   │     Hub     │  │   (Models)  │  │  (Rerank)   │  │  (Lucene)   │       │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │    CUDA     │  │    vLLM     │  │   PyTorch   │  │ Transformers│       │
│   │   Toolkit   │  │   Engine    │  │   Backend   │  │   Library   │       │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                                             │
│   Optional APIs: OpenAI | Cohere | Voyage AI | Google Vertex AI             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
Query → [QExpand] → Expanded Query → [Retriever] → Candidates → [Reranker] → Final Ranking
         │                              │                          │
         │                              │                          │
         ▼                              ▼                          ▼
    DeepSeek-R1              DIVER-Retriever + BM25         GroupRank/Listwise
    (vLLM Local)             (vLLM + pyserini)              (vLLM/Gemini API)
```
