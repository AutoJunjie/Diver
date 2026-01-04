"""
Utility functions for Medical Retrieval pipeline.
"""
import time
import json
import logging
from typing import List, Dict, Any, Callable, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI API client with retry logic."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        retry_attempts: int = 3,
        retry_backoff_base: float = 1.0
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.retry_attempts = retry_attempts
        self.retry_backoff_base = retry_backoff_base

    def _retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                wait_time = self.retry_backoff_base * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise last_exception

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        n: int = 1
    ) -> str:
        """Get chat completion with retry.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Max output tokens
            n: Number of completions (ignored, kept for API compatibility with BedrockClient)

        Returns:
            Response string (single response, n parameter ignored)
        """
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        return self._retry_with_backoff(_call)

    def chat_completion_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict:
        """Get chat completion and parse as JSON."""
        response_text = self.chat_completion(messages, temperature, max_tokens)

        # Try to extract JSON from response
        try:
            # Check for JSON code block
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Failed to parse JSON from response: {response_text[:500]}")
            raise ValueError(f"Invalid JSON response: {e}")


class BedrockError(Exception):
    """Bedrock API call exception."""

    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error

    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {super().__str__()}"
        return super().__str__()


# Sentinel to indicate tokenizer loading hasn't been attempted yet
_TOKENIZER_NOT_LOADED = object()


class BedrockClient:
    """AWS Bedrock API client with retry logic and thinking mode support."""

    def __init__(
        self,
        model_id: str,
        region: str = "us-west-2",
        retry_attempts: int = 3,
        retry_backoff_base: float = 1.0,
        control_thinking: bool = False,
        max_tokens: int = 32768
    ):
        """
        Initialize Bedrock client.

        Args:
            model_id: Bedrock model ID (e.g., 'deepseek.r1-distill-qwen-14b')
            region: AWS region
            retry_attempts: Number of retry attempts
            retry_backoff_base: Base time for exponential backoff
            control_thinking: If True, skip thinking process by appending </think>
            max_tokens: Default max output tokens
        """
        import boto3

        self.model_id = model_id
        self.region = region
        self.retry_attempts = retry_attempts
        self.retry_backoff_base = retry_backoff_base
        self.control_thinking = control_thinking
        self.max_tokens = max_tokens

        # Initialize boto3 client (uses env vars or IAM role for credentials)
        self.client = boto3.client('bedrock-runtime', region_name=region)

        # Initialize tokenizer for chat template (optional, for thinking mode)
        self._tokenizer = _TOKENIZER_NOT_LOADED

        logger.info(f"BedrockClient initialized: model={model_id}, region={region}, control_thinking={control_thinking}")

    @property
    def tokenizer(self):
        """Lazy load tokenizer for chat template."""
        if self._tokenizer is _TOKENIZER_NOT_LOADED:
            try:
                from transformers import AutoTokenizer
                # Use a compatible tokenizer for DeepSeek R1 Distill Qwen
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                    trust_remote_code=True
                )
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}. Using simple prompt format.")
                self._tokenizer = None
        return self._tokenizer

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Build prompt from messages with optional thinking skip.

        Args:
            messages: OpenAI-style message list [{"role": "user", "content": "..."}]

        Returns:
            Formatted prompt string
        """
        # Try to use tokenizer's chat template if available
        if self.tokenizer is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed: {e}. Using simple format.")
                prompt = self._simple_prompt_format(messages)
        else:
            prompt = self._simple_prompt_format(messages)

        # Add thinking skip if control_thinking is enabled
        if self.control_thinking:
            prompt += 'Okay, I think I have finished thinking.\n</think>\n'

        return prompt

    def _simple_prompt_format(self, messages: List[Dict[str, str]]) -> str:
        """Simple prompt format when tokenizer is unavailable."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _extract_answer(self, response: str) -> str:
        """
        Extract final answer from response (after </think> tag).

        Args:
            response: Raw model response

        Returns:
            Extracted answer (content after </think> or full response)
        """
        if "</think>" in response:
            return response.split("</think>")[-1].strip()
        return response.strip()

    def _invoke_bedrock(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Invoke Bedrock Converse API.

        Args:
            prompt: Formatted prompt
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            Model response text
        """
        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                inferenceConfig={
                    "temperature": temperature,
                    "maxTokens": max_tokens
                }
            )

            # Extract text from response
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            if content and isinstance(content, list):
                return content[0].get("text", "")
            return ""

        except self.client.exceptions.ThrottlingException as e:
            raise BedrockError("Rate limited by Bedrock", "ThrottlingException", e)
        except self.client.exceptions.ValidationException as e:
            raise BedrockError("Invalid request to Bedrock", "ValidationException", e)
        except self.client.exceptions.ModelTimeoutException as e:
            raise BedrockError("Model timeout", "ModelTimeoutException", e)
        except self.client.exceptions.ModelErrorException as e:
            raise BedrockError("Model error", "ModelErrorException", e)
        except Exception as e:
            raise BedrockError(f"Bedrock call failed: {e}", original_error=e)

    def _single_call(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Make a single Bedrock API call with retry.

        Args:
            messages: OpenAI-style message list
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            Extracted answer string
        """
        prompt = self._build_prompt(messages)
        logger.info(f"INPUT TO the model ({self.model_id}): {prompt[:500]}...")

        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                response = self._invoke_bedrock(prompt, temperature, max_tokens)
                answer = self._extract_answer(response)
                logger.info(f"OUTPUT from model: {answer[:500]}...")
                return answer

            except BedrockError as e:
                last_exception = e
                wait_time = self.retry_backoff_base * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

            except Exception as e:
                last_exception = e
                wait_time = self.retry_backoff_base * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        # All retries failed
        raise BedrockError(
            f"Failed after {self.retry_attempts} attempts: {last_exception}",
            error_code=getattr(last_exception, 'error_code', None),
            original_error=last_exception
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = None,
        n: int = 2
    ) -> List[str]:
        """
        Get chat completion(s) from Bedrock.

        Args:
            messages: OpenAI-style message list [{"role": "user", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Max output tokens (uses instance default if None)
            n: Number of responses to generate (parallel calls)

        Returns:
            List of response strings (length = n)
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        if n == 1:
            return [self._single_call(messages, temperature, max_tokens)]

        # Parallel calls for n > 1
        logger.info(f"Making {n} parallel calls to Bedrock...")

        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [
                executor.submit(self._single_call, messages, temperature, max_tokens)
                for _ in range(n)
            ]
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Parallel call failed: {e}")
                    results.append("")  # Append empty string on failure

        return results


class StrandsError(Exception):
    """Strands Agent call exception."""

    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error

    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {super().__str__()}"
        return super().__str__()


class StrandsClient:
    """Strands Agents SDK client wrapper with consistent interface."""

    def __init__(
        self,
        model_id: str,
        provider: str = "bedrock",
        region: str = "us-west-2",
        temperature: float = 0.7,
        max_tokens: int = 32768,
        control_thinking: bool = False,
        streaming: bool = False
    ):
        """
        Initialize Strands client.

        Args:
            model_id: Model identifier (e.g., 'us.deepseek.r1-distill-qwen-14b')
            provider: Model provider ("bedrock", "openai")
            region: AWS region (for Bedrock)
            temperature: Default sampling temperature
            max_tokens: Default max output tokens
            control_thinking: If True, skip thinking process by appending </think>
            streaming: Enable streaming (default False for batch processing)
        """
        self.model_id = model_id
        self.provider = provider
        self.region = region
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.control_thinking = control_thinking
        self.streaming = streaming

        # Initialize model based on provider
        self._init_model()

        logger.info(
            f"StrandsClient initialized: provider={provider}, model={model_id}, "
            f"region={region}, control_thinking={control_thinking}"
        )

    def _init_model(self):
        """Initialize Strands model and agent."""
        try:
            from strands import Agent

            if self.provider == "bedrock":
                from strands.models import BedrockModel
                self.model = BedrockModel(
                    model_id=self.model_id,
                    region_name=self.region,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    streaming=self.streaming
                )
            elif self.provider == "openai":
                from strands.models import OpenAIModel
                self.model = OpenAIModel(
                    model_id=self.model_id,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            # Create agent without tools (pure LLM completion)
            self.agent = Agent(model=self.model, tools=[])

        except ImportError as e:
            raise StrandsError(
                f"Failed to import strands-agents. Install with: pip install strands-agents",
                error_code="ImportError",
                original_error=e
            )
        except Exception as e:
            raise StrandsError(
                f"Failed to initialize Strands model: {e}",
                original_error=e
            )

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Build prompt from messages with optional thinking skip.

        Args:
            messages: OpenAI-style message list [{"role": "user", "content": "..."}]

        Returns:
            Formatted prompt string
        """
        system_prompt = None
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Set system prompt on agent if provided
        if system_prompt:
            self.agent.system_prompt = system_prompt

        prompt = "\n\n".join(prompt_parts)

        # Add thinking skip if control_thinking is enabled
        if self.control_thinking:
            prompt += "\n\nOkay, I think I have finished thinking.\n</think>\n"

        return prompt

    def _extract_answer(self, response: str) -> str:
        """
        Extract final answer from response (after </think> tag).

        Args:
            response: Raw model response

        Returns:
            Extracted answer (content after </think> or full response)
        """
        if "</think>" in response:
            return response.split("</think>")[-1].strip()
        return response.strip()

    def _single_call(self, prompt: str) -> str:
        """
        Make a single call to Strands Agent.

        Args:
            prompt: Formatted prompt string

        Returns:
            Extracted answer string
        """
        try:
            logger.info(f"INPUT TO Strands ({self.model_id}): {prompt[:500]}...")

            result = self.agent(prompt)

            # Extract message from AgentResult
            if hasattr(result, 'message'):
                response = str(result.message) if result.message else ""
            else:
                response = str(result)

            answer = self._extract_answer(response)
            logger.info(f"OUTPUT from Strands: {answer[:500]}...")

            return answer

        except Exception as e:
            raise StrandsError(f"Strands call failed: {e}", original_error=e)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        n: int = 1
    ) -> List[str]:
        """
        Get chat completion(s) using Strands Agent.

        Args:
            messages: OpenAI-style message list [{"role": "user", "content": "..."}]
            temperature: Sampling temperature (uses instance default if None)
            max_tokens: Max output tokens (uses instance default if None)
            n: Number of responses to generate (parallel calls)

        Returns:
            List of response strings (length = n)
        """
        # Note: temperature and max_tokens are set at model init time in Strands
        # These parameters are kept for API compatibility but not used per-call
        if temperature is not None and temperature != self.temperature:
            logger.warning(
                f"Per-call temperature ({temperature}) differs from init ({self.temperature}). "
                f"Strands uses init-time temperature."
            )

        prompt = self._build_prompt(messages)

        if n == 1:
            return [self._single_call(prompt)]

        # Parallel calls for n > 1
        logger.info(f"Making {n} parallel calls to Strands...")

        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [executor.submit(self._single_call, prompt) for _ in range(n)]
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Parallel call failed: {e}")
                    results.append("")  # Append empty string on failure

        return results


# Type alias for all LLM clients
LLMClient = 'OpenAIClient | BedrockClient | StrandsClient'


def cluster_documents_by_keywords(
    documents: List,
    min_cluster_size: int = 5
) -> Dict[str, List[int]]:
    """
    Cluster documents by their keywords.

    Args:
        documents: List of Document objects
        min_cluster_size: Minimum number of docs per keyword to form a cluster

    Returns:
        Dict mapping keyword to list of document IDs
    """
    keyword_to_docs = defaultdict(list)

    for doc in documents:
        for keyword in doc.keywords:
            keyword_to_docs[keyword].append(doc.doc_id)

    # Filter to clusters meeting minimum size
    clusters = {
        kw: doc_ids
        for kw, doc_ids in keyword_to_docs.items()
        if len(doc_ids) >= min_cluster_size
    }

    return clusters


def select_diverse_clusters(
    clusters: Dict[str, List[int]],
    num_clusters: int,
    min_overlap: float = 0.3
) -> List[str]:
    """
    Select diverse keyword clusters with minimal overlap.

    Args:
        clusters: Dict mapping keyword to document IDs
        num_clusters: Number of clusters to select
        min_overlap: Maximum overlap ratio allowed between selected clusters

    Returns:
        List of selected keyword names
    """
    # Sort by cluster size (prefer larger clusters)
    sorted_keywords = sorted(clusters.keys(), key=lambda k: len(clusters[k]), reverse=True)

    selected = []
    selected_docs = set()

    for keyword in sorted_keywords:
        if len(selected) >= num_clusters:
            break

        doc_set = set(clusters[keyword])

        # Check overlap with already selected clusters
        if selected_docs:
            overlap = len(doc_set & selected_docs) / len(doc_set)
            if overlap > min_overlap:
                continue

        selected.append(keyword)
        selected_docs.update(doc_set)

    return selected


def compute_llm_judge_metrics(
    judgments: List[Dict],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute aggregate metrics from LLM judge relevance scores.

    Args:
        judgments: List of dicts with query_id, doc_id, relevance_score (0/1/2)
        k_values: K values for precision@K computation

    Returns:
        Dict of metric names to values
    """
    # Group by query
    query_judgments = defaultdict(list)
    for j in judgments:
        query_judgments[j["query_id"]].append(j)

    metrics = {}

    # Mean relevance score
    all_scores = [j["relevance_score"] for j in judgments]
    metrics["mean_relevance"] = sum(all_scores) / len(all_scores) if all_scores else 0

    # Per-query metrics
    precision_at_k = {k: [] for k in k_values}
    ndcg_at_k = {k: [] for k in k_values}

    for query_id, qj in query_judgments.items():
        # Sort by retrieval rank (assuming doc_rank field exists)
        qj_sorted = sorted(qj, key=lambda x: x.get("rank", 0))

        for k in k_values:
            top_k = qj_sorted[:k]
            if not top_k:
                continue

            # Precision@K: fraction of docs rated >= 1 (partially or highly relevant)
            relevant_count = sum(1 for j in top_k if j["relevance_score"] >= 1)
            precision_at_k[k].append(relevant_count / len(top_k))

            # NDCG@K with graded relevance
            dcg = sum(
                j["relevance_score"] / (i + 2)  # log2(i+2) approximated
                for i, j in enumerate(top_k)
            )
            # Ideal DCG: all 2s
            idcg = sum(2 / (i + 2) for i in range(len(top_k)))
            ndcg_at_k[k].append(dcg / idcg if idcg > 0 else 0)

    # Average across queries
    for k in k_values:
        if precision_at_k[k]:
            metrics[f"precision@{k}"] = sum(precision_at_k[k]) / len(precision_at_k[k])
        if ndcg_at_k[k]:
            metrics[f"ndcg@{k}"] = sum(ndcg_at_k[k]) / len(ndcg_at_k[k])

    # Count zero-relevance queries (all docs rated 0)
    zero_relevance_queries = [
        qid for qid, qj in query_judgments.items()
        if all(j["relevance_score"] == 0 for j in qj)
    ]
    metrics["zero_relevance_query_count"] = len(zero_relevance_queries)
    metrics["zero_relevance_query_ids"] = zero_relevance_queries

    return metrics


def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: str):
    """Save data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
