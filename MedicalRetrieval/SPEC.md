# MedicalRetrieval Bedrock Integration Specification

## Overview

为 MedicalRetrieval pipeline 新增 AWS Bedrock 平台支持，使用 DeepSeek R1 Distill Qwen 模型替代 OpenAI API。

### 目标模型
| 模型 | Bedrock Model ID | 用途 |
|------|------------------|------|
| DeepSeek-R1-Distill-Qwen-14B | `deepseek.r1-distill-qwen-14b` | Query Expansion |
| DeepSeek-R1-Distill-Qwen-32B | `deepseek.r1-distill-qwen-32b` | Reranking, LLM Judge |

> **Note**: Model ID 格式需根据 Bedrock 控制台实际确认，上述为占位符。

---

## Architecture

### 设计原则
- **独立实现**: `BedrockClient` 与 `OpenAIClient` 并列，通过 config 选择使用哪个
- **不抽象**: 不创建公共基类，保持实现简单
- **单 Provider**: 整个 pipeline 使用同一个 LLM provider，不支持混合模式

### 文件结构
```
MedicalRetrieval/
├── utils.py                  # 新增 BedrockClient, BedrockError
├── qexpand_openai.py         # 修改：支持 Bedrock
├── reranker_openai.py        # 修改：支持 Bedrock
├── llm_judge.py              # 修改：支持 Bedrock
├── query_generator.py        # 修改：支持 Bedrock
├── run_pipeline.py           # 修改：新增 provider 选择
├── config.json               # 新增 Bedrock 配置项
├── tests/
│   └── test_bedrock_client.py  # 新增：单元测试
└── SPEC.md                   # 本文档
```

---

## BedrockClient Implementation

### 类定义 (`utils.py`)

```python
class BedrockError(Exception):
    """Bedrock API 调用异常"""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error


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
        ...
```

### 核心方法

#### `chat_completion`
```python
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
```

### 关键实现细节

#### 1. Authentication
- 使用环境变量 AWS credentials
- 支持: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- 支持: IAM Role (EC2/ECS)
- 支持: `AWS_PROFILE` 环境变量

```python
import boto3

# boto3 自动从环境变量/IAM Role 获取 credentials
self.client = boto3.client(
    'bedrock-runtime',
    region_name=region
)
```

#### 2. API 选择
- 使用 **Converse API** (统一消息格式)

```python
response = self.client.converse(
    modelId=self.model_id,
    messages=[
        {"role": "user", "content": [{"text": prompt}]}
    ],
    inferenceConfig={
        "temperature": temperature,
        "maxTokens": max_tokens
    }
)
```

#### 3. Thinking Mode (control_thinking)

```python
def _build_prompt(self, messages: List[Dict]) -> str:
    """Build prompt with optional thinking skip."""
    # 使用 transformers tokenizer 构建 chat template
    prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt = self.tokenizer.decode(prompt)

    if self.control_thinking:
        # 跳过思考过程，直接输出答案
        prompt += 'Okay, I think I have finished thinking.' + "\n</think>\n"

    return prompt

def _extract_answer(self, response: str) -> str:
    """Extract final answer from response (after </think> tag)."""
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response
```

#### 4. Multi-Response (n > 1)

使用 `concurrent.futures.ThreadPoolExecutor` 并发调用:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def chat_completion(self, messages, temperature=0.7, max_tokens=None, n=2):
    if n == 1:
        return [self._single_call(messages, temperature, max_tokens)]

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [
            executor.submit(self._single_call, messages, temperature, max_tokens)
            for _ in range(n)
        ]
        results = [f.result() for f in as_completed(futures)]

    return results
```

#### 5. Retry Logic

与 OpenAIClient 保持一致:

```python
def _retry_with_backoff(self, func, *args, **kwargs):
    last_exception = None

    for attempt in range(self.retry_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            wait_time = self.retry_backoff_base * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    # 包装为 BedrockError
    raise BedrockError(
        f"Failed after {self.retry_attempts} attempts",
        error_code=getattr(last_exception, 'response', {}).get('Error', {}).get('Code'),
        original_error=last_exception
    )
```

#### 6. Error Handling

统一包装为 `BedrockError`:

```python
try:
    response = self.client.converse(...)
except self.client.exceptions.ThrottlingException as e:
    raise BedrockError("Rate limited", "ThrottlingException", e)
except self.client.exceptions.ValidationException as e:
    raise BedrockError("Invalid request", "ValidationException", e)
except Exception as e:
    raise BedrockError(f"Bedrock call failed: {e}", original_error=e)
```

#### 7. Logging

详细日志输出:

```python
import logging

logger = logging.getLogger(__name__)

def _single_call(self, messages, temperature, max_tokens):
    prompt = self._build_prompt(messages)
    logger.info(f"INPUT TO the model ({self.model_id}): {prompt[:500]}...")

    response = self._invoke_bedrock(prompt, temperature, max_tokens)
    logger.info(f"OUTPUT from model: {response[:500]}...")

    return self._extract_answer(response)
```

#### 8. Input Truncation
- **不实现截断**，依赖 Bedrock API 自行处理过长输入
- Bedrock 会返回 `ValidationException` 如果超出限制

---

## Configuration

### config.json 结构 (扁平)

```json
{
    "llm_provider": "bedrock",

    "openai_api_key": "sk-xxx",
    "openai_model": "gpt-4o-mini",

    "bedrock_model_14b": "deepseek.r1-distill-qwen-14b",
    "bedrock_model_32b": "deepseek.r1-distill-qwen-32b",
    "bedrock_region": "us-west-2",
    "bedrock_max_tokens": 32768,
    "bedrock_control_thinking": false,

    "retry_attempts": 3,
    "retry_backoff_base": 1.0,

    "s3_input_path": "s3://bucket/path/to/data.csv",
    "retriever_model": "AQ-MedAI/Diver-Retriever-4B",
    "bm25_alpha": 0.3,
    "num_queries": 20,
    "num_expansion_rounds": 3,
    "rerank_top_k": 20,
    "eval_top_k": 10
}
```

### 配置项说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm_provider` | string | `"openai"` | LLM provider 选择: `"openai"` 或 `"bedrock"` |
| `bedrock_model_14b` | string | - | 14B 模型的 Bedrock Model ID |
| `bedrock_model_32b` | string | - | 32B 模型的 Bedrock Model ID |
| `bedrock_region` | string | `"us-west-2"` | AWS Region |
| `bedrock_max_tokens` | int | `32768` | 最大输出 token 数 |
| `bedrock_control_thinking` | bool | `false` | 是否跳过 thinking 过程 |

---

## Module Integration

### 模型分配策略 (参考原 DIVER)

| 模块 | 模型 | 说明 |
|------|------|------|
| `query_generator.py` | 14B | 合成查询生成 |
| `qexpand_openai.py` | 14B | Query Expansion |
| `reranker_openai.py` | 32B | Listwise Reranking |
| `llm_judge.py` | 32B | 相关性评估 |

### 修改示例: qexpand_openai.py

```python
# Before
from utils import OpenAIClient

def expand_queries_iterative(
    queries: List[Tuple[str, str]],
    retriever: HybridRetriever,
    openai_client: OpenAIClient,
    ...
):
    ...

# After
from utils import OpenAIClient, BedrockClient

def expand_queries_iterative(
    queries: List[Tuple[str, str]],
    retriever: HybridRetriever,
    llm_client,  # 可以是 OpenAIClient 或 BedrockClient
    ...
):
    ...
    # 调用方式保持一致
    response = llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=200
    )
    # Bedrock 返回 List[str]，取第一个
    expanded = response[0] if isinstance(response, list) else response
```

### 修改示例: run_pipeline.py

```python
def run_pipeline(config_path: str = "config.json", ...):
    config = load_config(config_path)

    # 根据 provider 初始化 LLM client
    if config.get("llm_provider") == "bedrock":
        # QExpand/QueryGen 用 14B
        llm_client_14b = BedrockClient(
            model_id=config["bedrock_model_14b"],
            region=config.get("bedrock_region", "us-west-2"),
            retry_attempts=config.get("retry_attempts", 3),
            control_thinking=config.get("bedrock_control_thinking", False),
            max_tokens=config.get("bedrock_max_tokens", 32768)
        )
        # Rerank/Judge 用 32B
        llm_client_32b = BedrockClient(
            model_id=config["bedrock_model_32b"],
            region=config.get("bedrock_region", "us-west-2"),
            retry_attempts=config.get("retry_attempts", 3),
            control_thinking=config.get("bedrock_control_thinking", False),
            max_tokens=config.get("bedrock_max_tokens", 32768)
        )
    else:
        # OpenAI 使用同一个 client
        openai_client = OpenAIClient(
            api_key=config["openai_api_key"],
            model=config["openai_model"]
        )
        llm_client_14b = openai_client
        llm_client_32b = openai_client

    # 传递给各模块
    queries = generate_queries(documents, llm_client_14b, ...)
    expanded_queries = expand_queries_iterative(queries, retriever, llm_client_14b, ...)
    reranked_results = rerank_all(retrieval_results, documents, llm_client_32b, ...)
    judgments = evaluate_reranked_results(reranked_results, ..., llm_client_32b, ...)
```

---

## API Interface Compatibility

### Response Format Difference

| Method | OpenAIClient | BedrockClient |
|--------|--------------|---------------|
| `chat_completion()` | Returns `str` | Returns `List[str]` (length=n) |
| `chat_completion_json()` | Returns `Dict` | **Not implemented** |

### 调用方兼容处理

```python
# 统一处理方式
response = llm_client.chat_completion(messages, temperature=0.6)
if isinstance(response, list):
    text = response[0]  # Bedrock
else:
    text = response      # OpenAI
```

---

## Testing

### 单元测试 (`tests/test_bedrock_client.py`)

```python
import unittest
from unittest.mock import Mock, patch
from utils import BedrockClient, BedrockError


class TestBedrockClient(unittest.TestCase):

    @patch('boto3.client')
    def test_initialization(self, mock_boto):
        client = BedrockClient(
            model_id="deepseek.r1-distill-qwen-14b",
            region="us-west-2"
        )
        mock_boto.assert_called_with('bedrock-runtime', region_name='us-west-2')

    @patch('boto3.client')
    def test_chat_completion_single(self, mock_boto):
        mock_bedrock = Mock()
        mock_bedrock.converse.return_value = {
            'output': {'message': {'content': [{'text': '<think>thinking</think>\nAnswer'}]}}
        }
        mock_boto.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model")
        result = client.chat_completion(
            [{"role": "user", "content": "test"}],
            n=1
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Answer")

    @patch('boto3.client')
    def test_chat_completion_multi(self, mock_boto):
        mock_bedrock = Mock()
        mock_bedrock.converse.return_value = {
            'output': {'message': {'content': [{'text': 'Response'}]}}
        }
        mock_boto.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model")
        result = client.chat_completion(
            [{"role": "user", "content": "test"}],
            n=2
        )

        self.assertEqual(len(result), 2)

    @patch('boto3.client')
    def test_control_thinking(self, mock_boto):
        client = BedrockClient(
            model_id="test-model",
            control_thinking=True
        )
        # Verify prompt modification
        prompt = client._build_prompt([{"role": "user", "content": "test"}])
        self.assertIn("</think>", prompt)

    @patch('boto3.client')
    def test_retry_on_failure(self, mock_boto):
        mock_bedrock = Mock()
        mock_bedrock.converse.side_effect = [
            Exception("Temporary error"),
            {'output': {'message': {'content': [{'text': 'Success'}]}}}
        ]
        mock_boto.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model", retry_attempts=3)
        result = client.chat_completion([{"role": "user", "content": "test"}], n=1)

        self.assertEqual(result[0], "Success")

    @patch('boto3.client')
    def test_error_wrapping(self, mock_boto):
        mock_bedrock = Mock()
        mock_bedrock.converse.side_effect = Exception("API Error")
        mock_boto.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model", retry_attempts=1)

        with self.assertRaises(BedrockError) as ctx:
            client.chat_completion([{"role": "user", "content": "test"}], n=1)

        self.assertIn("Failed after", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
```

---

## Dependencies

### 新增 Python 依赖

```txt
boto3>=1.34.0
transformers>=4.40.0  # 用于 chat template (如果需要)
```

### AWS 权限要求

IAM Policy 最小权限:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:Converse"
            ],
            "Resource": [
                "arn:aws:bedrock:us-west-2::foundation-model/deepseek.r1-distill-qwen-14b",
                "arn:aws:bedrock:us-west-2::foundation-model/deepseek.r1-distill-qwen-32b"
            ]
        }
    ]
}
```

---

## Documentation Updates

### README.md 新增内容

```markdown
## LLM Provider Configuration

MedicalRetrieval 支持两种 LLM provider:

### OpenAI (默认)
```json
{
    "llm_provider": "openai",
    "openai_api_key": "sk-xxx",
    "openai_model": "gpt-4o-mini"
}
```

### AWS Bedrock (DeepSeek R1)
```json
{
    "llm_provider": "bedrock",
    "bedrock_model_14b": "deepseek.r1-distill-qwen-14b",
    "bedrock_model_32b": "deepseek.r1-distill-qwen-32b",
    "bedrock_region": "us-west-2",
    "bedrock_max_tokens": 32768,
    "bedrock_control_thinking": false
}
```

**AWS 认证**: 通过环境变量配置
```bash
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
export AWS_DEFAULT_REGION=us-west-2
```

或使用 IAM Role (推荐在 EC2/ECS 上使用)。
```

---

## Implementation Checklist

- [ ] `utils.py`: 新增 `BedrockError` 异常类
- [ ] `utils.py`: 新增 `BedrockClient` 类
  - [ ] `__init__` 初始化 boto3 client
  - [ ] `chat_completion` 主方法
  - [ ] `_single_call` 单次调用
  - [ ] `_build_prompt` 构建 prompt (含 thinking mode)
  - [ ] `_extract_answer` 提取答案
  - [ ] `_retry_with_backoff` 重试逻辑
- [ ] `query_generator.py`: 支持 BedrockClient
- [ ] `qexpand_openai.py`: 支持 BedrockClient
- [ ] `reranker_openai.py`: 支持 BedrockClient
- [ ] `llm_judge.py`: 支持 BedrockClient
- [ ] `run_pipeline.py`: 新增 provider 选择逻辑
- [ ] `config.json`: 新增 Bedrock 配置项示例
- [ ] `tests/test_bedrock_client.py`: 单元测试
- [ ] `README.md`: 更新文档

---

## Appendix: Bedrock Converse API Reference

```python
response = client.converse(
    modelId='deepseek.r1-distill-qwen-14b',
    messages=[
        {
            'role': 'user',
            'content': [
                {'text': 'Your prompt here'}
            ]
        }
    ],
    inferenceConfig={
        'temperature': 0.7,
        'topP': 0.9,
        'maxTokens': 32768
    }
)

# Response structure
{
    'output': {
        'message': {
            'role': 'assistant',
            'content': [
                {'text': 'Model response here'}
            ]
        }
    },
    'usage': {
        'inputTokens': 100,
        'outputTokens': 500,
        'totalTokens': 600
    },
    'stopReason': 'end_turn'
}
```
