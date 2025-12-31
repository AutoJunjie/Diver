"""
Output writer for generating CSV reports and uploading to S3.
"""
import os
import csv
import json
import boto3
from datetime import datetime
from typing import List, Dict, Any, Optional

from llm_judge import RelevanceJudgment


def write_csv_report(
    judgments: List[RelevanceJudgment],
    output_path: str
) -> str:
    """
    Write judgments to CSV report.

    Args:
        judgments: List of RelevanceJudgment objects
        output_path: Path to output CSV file

    Returns:
        Path to written file
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            "query_id",
            "query_text",
            "expanded_query",
            "doc_id",
            "doc_title",
            "retrieval_score",
            "rerank_score",
            "relevance_judgment"
        ])

        # Write data
        for j in judgments:
            writer.writerow([
                j.query_id,
                j.query_text,
                j.expanded_query,
                j.doc_id,
                j.doc_title,
                f"{j.retrieval_score:.6f}",
                f"{j.rerank_score:.6f}",
                j.relevance_score
            ])

    print(f"CSV report written to {output_path}")
    return output_path


def write_metrics_report(
    metrics: Dict[str, Any],
    output_path: str
) -> str:
    """
    Write metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to output JSON file

    Returns:
        Path to written file
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Metrics report written to {output_path}")
    return output_path


def write_debug_outputs(
    output_dir: str,
    queries: Optional[List] = None,
    expanded_queries: Optional[List] = None,
    retrieval_results: Optional[Dict] = None,
    rerank_results: Optional[Dict] = None,
    judgments: Optional[List[RelevanceJudgment]] = None
):
    """
    Write intermediate outputs for debugging.

    Args:
        output_dir: Directory to write debug outputs
        queries: Generated queries
        expanded_queries: Expanded queries
        retrieval_results: Retrieval results dict
        rerank_results: Rerank results dict
        judgments: Relevance judgments
    """
    os.makedirs(output_dir, exist_ok=True)

    if queries:
        path = os.path.join(output_dir, "queries.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        print(f"Queries saved to {path}")

    if expanded_queries:
        path = os.path.join(output_dir, "expanded_queries.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(expanded_queries, f, ensure_ascii=False, indent=2)
        print(f"Expanded queries saved to {path}")

    if retrieval_results:
        path = os.path.join(output_dir, "retrieval_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(retrieval_results, f, ensure_ascii=False, indent=2)
        print(f"Retrieval results saved to {path}")

    if rerank_results:
        path = os.path.join(output_dir, "rerank_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rerank_results, f, ensure_ascii=False, indent=2)
        print(f"Rerank results saved to {path}")

    if judgments:
        path = os.path.join(output_dir, "judgments.json")
        judgment_dicts = [
            {
                "query_id": j.query_id,
                "query_text": j.query_text,
                "expanded_query": j.expanded_query,
                "doc_id": j.doc_id,
                "doc_title": j.doc_title,
                "relevance_score": j.relevance_score,
                "rank": j.rank,
                "retrieval_score": j.retrieval_score,
                "rerank_score": j.rerank_score,
                "explanation": j.explanation
            }
            for j in judgments
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(judgment_dicts, f, ensure_ascii=False, indent=2)
        print(f"Judgments saved to {path}")


def upload_to_s3(local_path: str, s3_path: str) -> str:
    """
    Upload a file to S3.

    Args:
        local_path: Path to local file
        s3_path: S3 URI (s3://bucket/key)

    Returns:
        S3 URI of uploaded file
    """
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")

    path_parts = s3_path[5:].split("/", 1)
    bucket = path_parts[0]
    key = path_parts[1] if len(path_parts) > 1 else ""

    print(f"Uploading {local_path} to s3://{bucket}/{key}")

    s3_client = boto3.client("s3")
    s3_client.upload_file(local_path, bucket, key)

    print(f"Upload complete: s3://{bucket}/{key}")
    return f"s3://{bucket}/{key}"


def upload_directory_to_s3(local_dir: str, s3_base_path: str) -> List[str]:
    """
    Upload all files in a directory to S3.

    Args:
        local_dir: Local directory path
        s3_base_path: S3 base URI

    Returns:
        List of S3 URIs of uploaded files
    """
    uploaded = []

    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_path = f"{s3_base_path.rstrip('/')}/{relative_path}"

            try:
                upload_to_s3(local_path, s3_path)
                uploaded.append(s3_path)
            except Exception as e:
                print(f"Failed to upload {local_path}: {e}")

    return uploaded


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class OutputManager:
    """Manages output file generation and S3 upload."""

    def __init__(
        self,
        local_output_dir: str = "output",
        s3_output_path: Optional[str] = None,
        debug_mode: bool = False
    ):
        self.run_id = generate_run_id()
        self.local_output_dir = os.path.join(local_output_dir, self.run_id)
        self.s3_output_path = s3_output_path
        self.debug_mode = debug_mode

        os.makedirs(self.local_output_dir, exist_ok=True)
        print(f"Output directory: {self.local_output_dir}")

    def save_csv_report(self, judgments: List[RelevanceJudgment]) -> str:
        """Save CSV report locally."""
        path = os.path.join(self.local_output_dir, "results.csv")
        return write_csv_report(judgments, path)

    def save_metrics(self, metrics: Dict[str, Any]) -> str:
        """Save metrics report locally."""
        path = os.path.join(self.local_output_dir, "metrics.json")
        return write_metrics_report(metrics, path)

    def save_debug_outputs(self, **kwargs):
        """Save debug outputs if in debug mode."""
        if self.debug_mode:
            debug_dir = os.path.join(self.local_output_dir, "debug")
            write_debug_outputs(debug_dir, **kwargs)

    def upload_all_to_s3(self) -> List[str]:
        """Upload all outputs to S3."""
        if not self.s3_output_path:
            print("No S3 output path configured, skipping upload")
            return []

        s3_run_path = f"{self.s3_output_path.rstrip('/')}/{self.run_id}"
        return upload_directory_to_s3(self.local_output_dir, s3_run_path)

    def finalize(
        self,
        judgments: List[RelevanceJudgment],
        metrics: Dict[str, Any],
        **debug_kwargs
    ) -> Dict[str, str]:
        """
        Finalize outputs: save locally and upload to S3.

        Returns:
            Dict with paths to outputs
        """
        paths = {}

        # Save CSV report
        csv_path = self.save_csv_report(judgments)
        paths["csv_report"] = csv_path

        # Save metrics
        metrics_path = self.save_metrics(metrics)
        paths["metrics"] = metrics_path

        # Save debug outputs
        if self.debug_mode:
            self.save_debug_outputs(**debug_kwargs)
            paths["debug_dir"] = os.path.join(self.local_output_dir, "debug")

        # Upload to S3
        if self.s3_output_path:
            uploaded = self.upload_all_to_s3()
            paths["s3_uploads"] = uploaded

        return paths


if __name__ == "__main__":
    # Test output writer
    from llm_judge import RelevanceJudgment

    test_judgments = [
        RelevanceJudgment(
            query_id="0",
            query_text="华氏巨球蛋白血症 治疗",
            expanded_query="华氏巨球蛋白血症 治疗 靶向药物 预后",
            doc_id=123,
            doc_title="华氏巨球蛋白血症的靶向治疗进展",
            relevance_score=2,
            rank=0,
            retrieval_score=0.85,
            rerank_score=10.0,
            explanation="高度相关"
        ),
        RelevanceJudgment(
            query_id="0",
            query_text="华氏巨球蛋白血症 治疗",
            expanded_query="华氏巨球蛋白血症 治疗 靶向药物 预后",
            doc_id=456,
            doc_title="糖尿病合并华氏巨球蛋白血症一例报告",
            relevance_score=1,
            rank=1,
            retrieval_score=0.72,
            rerank_score=9.0,
            explanation="部分相关"
        )
    ]

    test_metrics = {
        "mean_relevance": 1.5,
        "precision@1": 1.0,
        "precision@5": 0.6
    }

    # Test local output
    manager = OutputManager(
        local_output_dir="test_output",
        s3_output_path=None,
        debug_mode=True
    )

    paths = manager.finalize(
        judgments=test_judgments,
        metrics=test_metrics,
        queries=[{"query_id": "0", "query_text": "test"}]
    )

    print(f"\nOutput paths: {paths}")
