# Medical Literature Retrieval Pipeline - Technical Specification

## Overview

Transform the DIVER benchmark evaluation pipeline into a production retrieval system for Chinese medical literature, replacing gold-label-based evaluation with LLM-as-a-judge.

## Data Source

| Property | Value |
|----------|-------|
| Location | `s3://sagemaker-us-east-1-955643200499/diver/medical_literature_no_duplicates.csv` |
| Format | CSV with columns: `abstract`, `title`, `keywords` |
| Document Count | ~41,500 Chinese medical papers |
| Document ID | Row index (0, 1, 2, ...) |
| Language | Chinese |

### Document Structure
```csv
abstract,title,keywords
"华氏巨球蛋白血症(Waldenstrom...)...",以眼部胀痛为首要表现的华氏巨球蛋白血症1例报道,"['华氏巨球蛋白血症', '蛋白电泳', 'MYD88L265P']"
```

---

## Pipeline Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Query Generator │ ──▶ │     QExpand      │ ──▶ │    Retrieval     │
│  (GPT-4o-mini)   │     │  (GPT-4o-mini)   │     │ (DIVER-4B + BM25)│
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                           │
                                                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   CSV Report     │ ◀── │   LLM Judge      │ ◀── │    Reranking     │
│   (to S3)        │     │  (GPT-4o-mini)   │     │  (GPT-4o-mini)   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## Stage 1: Query Generation (NEW)

Generate 20 synthetic Chinese queries from the document corpus.

| Parameter | Value |
|-----------|-------|
| Query Count | 20 |
| Query Language | Chinese |
| Query Style | Keyword-style (e.g., "华氏巨球蛋白血症 治疗方案") |
| Clustering Method | Keyword-based grouping |
| Cluster Count | Auto-determined based on keyword distribution |
| LLM Model | GPT-4o-mini |

### Algorithm
1. Parse `keywords` column from all documents
2. Cluster documents by keyword overlap/similarity
3. Auto-determine optimal cluster count
4. Sample representative document(s) from each cluster
5. Generate keyword-style query per cluster using GPT-4o-mini

---

## Stage 2: Query Expansion (QExpand)

Iterative retrieval-augmented query expansion.

| Parameter | Value |
|-----------|-------|
| LLM Model | GPT-4o-mini (replacing DeepSeek-R1-14B) |
| Expansion Rounds | 3 (iterative) |
| Retrieval Feedback | Yes - requires full index built first |

### Process
1. Build full document index (41K docs) before expansion
2. For each query, iterate 3 rounds:
   - Expand query with LLM
   - Retrieve top documents
   - Use retrieved docs as context for next expansion

---

## Stage 3: Retrieval

Hybrid dense + sparse retrieval.

| Parameter | Value |
|-----------|-------|
| Dense Model | DIVER-Retriever-4B (local, via vLLM) |
| Sparse Model | BM25 (via pyserini/gensim) |
| Hybrid Weight | α = 0.3 (BM25 contribution) |
| Index | Full 41K documents |

### Document Representation for Embedding
- Concatenate: `title + abstract + keywords`
- No truncation needed (documents within token limits)

---

## Stage 4: Reranking

LLM-based reranking of retrieval candidates.

| Parameter | Value |
|-----------|-------|
| LLM Model | GPT-4o-mini |
| Candidates to Rerank | Top 20 from retrieval |
| Document Context | Title + Abstract + Keywords |

---

## Stage 5: LLM-as-Judge Evaluation

Replace NDCG/gold-label evaluation with LLM relevance judgments.

| Parameter | Value |
|-----------|-------|
| LLM Model | GPT-4o-mini |
| Documents Evaluated | Top 10 per query |
| Relevance Scale | 3-point: Not relevant (0) / Partially relevant (1) / Highly relevant (2) |
| Relevance Criteria | Topical relevance only |
| Total Judgments | 200 (20 queries × 10 docs) |

### Aggregate Metrics
- Mean relevance score (0-2 scale)
- Precision@K variants (% docs rated ≥ Partially relevant)
- NDCG-style with graded relevance (using 0/1/2 scores)

### Zero-Relevance Handling
- If all retrieved docs for a query are rated "Not relevant": Flag in report and continue (no regeneration)

---

## Output Specification

### Primary Output: CSV Report

| Column | Description |
|--------|-------------|
| `query_id` | Query identifier (0-19) |
| `query_text` | The Chinese query text |
| `doc_id` | Document row index |
| `doc_title` | Document title |
| `retrieval_score` | Score from retrieval stage |
| `rerank_score` | Score from reranking stage |
| `relevance_judgment` | LLM judge score (0, 1, or 2) |

### Output Location
- Primary: `s3://sagemaker-us-east-1-955643200499/diver/output/`
- Debug mode: Also saves intermediate files (embeddings, retrieval scores, etc.)

### Debug Mode Outputs
When enabled, save:
- Document embeddings (pickle)
- Retrieval scores per query
- Reranking scores per query
- Raw LLM judgments with explanations

---

## Configuration

### API Key Management
- Store in config file: `MedicalRetrieval/config.json`
- Copy `config.example.json` to `config.json` and add your OpenAI API key

### Error Handling
- OpenAI API calls: 3 retries with exponential backoff (1s, 2s, 4s)

### GPU Requirements
- DIVER-Retriever-4B requires ~16GB+ VRAM
- User confirmed sufficient GPU available

---

## Code Organization

Create new parallel module: `MedicalRetrieval/`

```
MedicalRetrieval/
├── config.json              # API keys, S3 paths
├── run_pipeline.py          # Single command entry point
├── query_generator.py       # Stage 1: Synthetic query generation
├── qexpand_openai.py        # Stage 2: Query expansion with GPT-4o-mini
├── retriever.py             # Stage 3: DIVER + BM25 hybrid (reuse existing)
├── reranker_openai.py       # Stage 4: GPT-4o-mini reranking
├── llm_judge.py             # Stage 5: LLM-as-judge evaluation
├── data_loader.py           # S3 CSV loading, document ID generation
├── output_writer.py         # CSV report generation, S3 upload
└── utils.py                 # Retry logic, clustering, metrics
```

### Execution
```bash
# Single command end-to-end
python MedicalRetrieval/run_pipeline.py

# With debug mode
python MedicalRetrieval/run_pipeline.py --debug
```

---

## Dependencies

### Existing (from DIVER)
- `vllm` - DIVER-Retriever-4B inference
- `pyserini` / `gensim` - BM25
- `torch`, `transformers`, `sentence-transformers`

### New
- `openai` - GPT-4o-mini API calls
- `boto3` - S3 read/write

---

## Cost Estimation

| Stage | Model | Est. Tokens | Est. Cost |
|-------|-------|-------------|-----------|
| Query Generation | GPT-4o-mini | ~50K | ~$0.01 |
| QExpand (20 queries × 3 rounds) | GPT-4o-mini | ~200K | ~$0.03 |
| Reranking (20 × 20 docs) | GPT-4o-mini | ~500K | ~$0.08 |
| LLM Judge (200 judgments) | GPT-4o-mini | ~300K | ~$0.05 |
| **Total** | | ~1M tokens | **~$0.17** |

---

## Summary of Key Decisions

| Aspect | Decision |
|--------|----------|
| Dataset | Medical literature CSV from S3 (no gold labels) |
| Embedding Model | DIVER-Retriever-4B (local) |
| LLM for all stages | GPT-4o-mini (OpenAI API) |
| Query Generation | 20 Chinese keyword-style queries, topic-clustered |
| Evaluation | LLM-as-judge, 3-point scale, topical relevance |
| Output | CSV per query-doc pair to S3 |
| Code Structure | New `MedicalRetrieval/` module |
| Orchestration | Single command `run_pipeline.py` |
