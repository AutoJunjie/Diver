#!/usr/bin/env python3
"""
Medical Literature Retrieval Pipeline - Main Entry Point

Single command to run the full pipeline:
1. Load documents from S3
2. Generate synthetic queries
3. Expand queries (iterative)
4. Retrieve documents (DIVER + BM25 hybrid)
5. Rerank candidates
6. Evaluate with LLM judge
7. Output CSV report to S3
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Ensure MedicalRetrieval is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_medical_documents, create_document_index
from utils import OpenAIClient, load_config, save_json
from query_generator import generate_queries, queries_to_list, queries_to_dict
from qexpand_openai import expand_queries_iterative, expanded_queries_to_list, expanded_queries_to_dict
from retriever import HybridRetriever, results_to_dict
from reranker_openai import rerank_all, rerank_results_to_dict
from llm_judge import evaluate_reranked_results, compute_evaluation_metrics, identify_zero_relevance_queries
from output_writer import OutputManager


def run_pipeline(config_path: str = "config.json", debug: bool = False, max_docs: int = None):
    """
    Run the complete medical literature retrieval pipeline.

    Args:
        config_path: Path to config.json
        debug: Enable debug mode to save intermediate outputs
        max_docs: Optional limit on documents to load (for testing)
    """
    print("=" * 60)
    print("Medical Literature Retrieval Pipeline")
    print("=" * 60)
    start_time = datetime.now()

    # Load configuration
    config_dir = Path(__file__).resolve().parent
    config_full_path = config_dir / config_path
    print(f"\n[1/8] Loading configuration from {config_full_path}")
    config = load_config(str(config_full_path))

    # Initialize OpenAI client
    print(f"\n[2/8] Initializing OpenAI client (model: {config['openai_model']})")
    openai_client = OpenAIClient(
        api_key=config["openai_api_key"],
        model=config["openai_model"],
        retry_attempts=config.get("retry_attempts", 3),
        retry_backoff_base=config.get("retry_backoff_base", 1.0)
    )

    # Load documents from S3
    print(f"\n[3/8] Loading documents from S3")
    cache_dir = config_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    local_csv_path = cache_dir / "medical_literature.csv"

    documents = load_medical_documents(
        s3_path=config["s3_input_path"],
        local_cache_path=str(local_csv_path),
        max_documents=max_docs
    )
    doc_index = create_document_index(documents)
    print(f"  Loaded {len(documents)} documents")

    # Initialize retriever (builds index)
    print(f"\n[4/8] Initializing retriever (DIVER-4B + BM25)")
    retriever = HybridRetriever(
        documents=documents,
        model_path=config["retriever_model"],
        cache_dir=str(cache_dir),
        bm25_alpha=config.get("bm25_alpha", 0.3)
    )

    # Generate synthetic queries
    print(f"\n[5/8] Generating {config['num_queries']} synthetic queries")
    queries = generate_queries(
        documents=documents,
        openai_client=openai_client,
        num_queries=config["num_queries"]
    )
    query_list = queries_to_list(queries)
    print(f"  Generated {len(queries)} queries")

    # Query expansion (iterative)
    print(f"\n[6/8] Expanding queries ({config['num_expansion_rounds']} rounds)")
    expanded_queries = expand_queries_iterative(
        queries=query_list,
        retriever=retriever,
        openai_client=openai_client,
        num_rounds=config["num_expansion_rounds"]
    )
    expanded_query_list = expanded_queries_to_list(expanded_queries)
    print(f"  Expanded {len(expanded_queries)} queries")

    # Retrieval
    print(f"\n[7/8] Retrieving documents")
    retrieval_results = retriever.retrieve(
        queries=expanded_query_list,
        top_k=config.get("rerank_top_k", 20) * 2  # Get more for reranking
    )
    retrieval_scores = results_to_dict(retrieval_results)
    print(f"  Retrieved candidates for {len(retrieval_results)} queries")

    # Reranking
    print(f"\n[8/8] Reranking top {config['rerank_top_k']} candidates per query")
    reranked_results = rerank_all(
        retrieval_results=retrieval_results,
        documents=documents,
        openai_client=openai_client,
        top_k=config["rerank_top_k"]
    )
    rerank_scores = rerank_results_to_dict(reranked_results)
    print(f"  Reranked {len(reranked_results)} queries")

    # LLM Judge evaluation
    print(f"\n[9/8] Evaluating with LLM judge (top {config['eval_top_k']} per query)")
    # Create mapping of query_id to original query text (use string key for consistency)
    original_queries_map = {str(q.query_id): q.query_text for q in queries}
    judgments = evaluate_reranked_results(
        reranked_results=reranked_results,
        retrieval_scores=retrieval_scores,
        documents=documents,
        openai_client=openai_client,
        top_k=config["eval_top_k"],
        original_queries=original_queries_map
    )
    print(f"  Made {len(judgments)} relevance judgments")

    # Compute metrics
    print("\n[10/8] Computing evaluation metrics")
    metrics = compute_evaluation_metrics(judgments)

    # Check for zero-relevance queries
    zero_rel_queries = identify_zero_relevance_queries(judgments)
    if zero_rel_queries:
        print(f"  Warning: {len(zero_rel_queries)} queries have zero-relevance results")
        metrics["zero_relevance_queries"] = zero_rel_queries

    # Output results
    print("\n[11/8] Writing output files")
    output_manager = OutputManager(
        local_output_dir=str(config_dir / "output"),
        s3_output_path=config.get("s3_output_path"),
        debug_mode=debug
    )

    # Prepare debug outputs
    debug_kwargs = {}
    if debug:
        debug_kwargs = {
            "queries": queries_to_dict(queries),
            "expanded_queries": expanded_queries_to_dict(expanded_queries),
            "retrieval_results": retrieval_scores,
            "rerank_results": rerank_scores
        }

    output_paths = output_manager.finalize(
        judgments=judgments,
        metrics=metrics,
        **debug_kwargs
    )

    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nDuration: {duration:.1f} seconds")
    print(f"\nMetrics:")
    for key, value in metrics.items():
        if key != "zero_relevance_query_ids":
            print(f"  {key}: {value}")

    print(f"\nOutputs:")
    for key, path in output_paths.items():
        if key != "s3_uploads":
            print(f"  {key}: {path}")

    if "s3_uploads" in output_paths and output_paths["s3_uploads"]:
        print(f"\nS3 uploads: {len(output_paths['s3_uploads'])} files")

    return {
        "metrics": metrics,
        "output_paths": output_paths,
        "duration_seconds": duration
    }


def main():
    parser = argparse.ArgumentParser(
        description="Medical Literature Retrieval Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run full pipeline
  python run_pipeline.py --debug            # Run with debug outputs
  python run_pipeline.py --max-docs 1000    # Run with document limit (testing)
"""
    )
    parser.add_argument(
        "--config", "-c",
        default="config.json",
        help="Path to config file (default: config.json)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode to save intermediate outputs"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to load (for testing)"
    )

    args = parser.parse_args()

    try:
        result = run_pipeline(
            config_path=args.config,
            debug=args.debug,
            max_docs=args.max_docs
        )
        return 0
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
