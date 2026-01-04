# MedicalRetrieval Strands Agents Integration Specification

## Overview

重构 MedicalRetrieval pipeline 的 LLM 调用，使用 Strands Agents SDK 替代直接的 boto3/OpenAI 调用。

### 目标
- 统一使用 Strands Agents SDK 进行 LLM 调用
- 简化代码，利用 SDK 内置的重试、错误处理等功能
- 支持多种 Model Provider (Bedrock, OpenAI, etc.)

### 目标模型
| 模型 | Provider | Model ID | 用途 |
|------|----------|----------|------|
| DeepSeek-R1-Distill-Qwen-14B | Bedrock | `us.deepseek.r1-distill-qwen-14b` | Query Expansion, Query Generation |
| DeepSeek-R1-Distill-Qwen-32B | Bedrock | `us.deepseek.r1-distill-qwen-32b` | Reranking, LLM Judge |

---

## Architecture

### 设计原则
- **统一抽象**: 创建 `StrandsClient` 包装 Strands Agent
- **兼容接口**: 保持与现有 `OpenAIClient`/`BedrockClient` 相同的 `chat_completion` 接口
- **简化实现**: 利用 Strands SDK 内置功能，减少自定义代码

### 文件结构
```
MedicalRetrieval/
├── utils.py                  # 新增 StrandsClient, StrandsError
├── qexpand_openai.py         # 修改：支持 Strands
├── reranker_openai.py        # 修改：支持 Strands
├── llm_judge.py              # 修改：支持 Strands
├── query_generator.py        # 修改：支持 Strands
├── run_pipeline.py           # 修改：新增 strands provider 选择
├── config.example.json       # 新增 Strands 配置项
├── tests/
│   ├── test_bedrock_client.py  # 现有测试
│   └── test_strands_client.py  # 新增：Strands 单元测试
└── SPEC_STRANDS.md           # 本文档
```

---

## StrandsClient Implementation

### 类定义 (`utils.py`)

```python
class StrandsError(Exception):
    """Strands Agent 调用异常"""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error


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
            model_id: Model identifier
            provider: Model provider ("bedrock", "openai", etc.)
            region: AWS region (for Bedrock)
            temperature: Default sampling temperature
            max_tokens: Default max output tokens
            control_thinking: If True, skip thinking process
            streaming: Enable streaming (default False for batch processing)
        """
        ...
```

### 核心方法

#### `chat_completion`
```python
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
```

### 关键实现细节

#### 1. Model Provider 初始化

```python
from strands import Agent
from strands.models import BedrockModel

def __init__(self, model_id, provider="bedrock", region="us-west-2", ...):
    self.model_id = model_id
    self.provider = provider
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.control_thinking = control_thinking

    # Initialize model based on provider
    if provider == "bedrock":
        self.model = BedrockModel(
            model_id=model_id,
            region_name=region,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )
    elif provider == "openai":
        from strands.models import OpenAIModel
        self.model = OpenAIModel(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Create agent without tools (pure LLM completion)
    self.agent = Agent(model=self.model, tools=[])
```

#### 2. Chat Completion 实现

```python
def chat_completion(
    self,
    messages: List[Dict[str, str]],
    temperature: float = None,
    max_tokens: int = None,
    n: int = 1
) -> List[str]:
    # Build prompt from messages
    prompt = self._build_prompt(messages)

    if n == 1:
        return [self._single_call(prompt)]

    # Parallel calls for n > 1
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(self._single_call, prompt) for _ in range(n)]
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Parallel call failed: {e}")
                results.append("")
    return results

def _single_call(self, prompt: str) -> str:
    """Make a single call to Strands Agent."""
    try:
        result = self.agent(prompt)
        response = result.message if hasattr(result, 'message') else str(result)
        return self._extract_answer(response)
    except Exception as e:
        raise StrandsError(f"Strands call failed: {e}", original_error=e)
```

#### 3. Thinking Mode (control_thinking)

```python
def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
    """Build prompt from messages with optional thinking skip."""
    # Convert messages to single prompt
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

    prompt = "\n\n".join(parts)

    # Add thinking skip if enabled
    if self.control_thinking:
        prompt += "\n\nOkay, I think I have finished thinking.\n</think>\n"

    return prompt

def _extract_answer(self, response: str) -> str:
    """Extract final answer from response (after </think> tag)."""
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response.strip()
```

#### 4. System Prompt Support

```python
def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
    """Build prompt from messages."""
    system_prompt = None
    user_messages = []

    for msg in messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
        else:
            user_messages.append(msg)

    # Set system prompt on agent if provided
    if system_prompt:
        self.agent.system_prompt = system_prompt

    # Build user prompt
    prompt_parts = []
    for msg in user_messages:
        content = msg.get("content", "")
        prompt_parts.append(content)

    prompt = "\n\n".join(prompt_parts)

    if self.control_thinking:
        prompt += "\n\nOkay, I think I have finished thinking.\n</think>\n"

    return prompt
```

---

## Configuration

### config.example.json 结构

```json
{
    "llm_provider": "strands",

    "openai_api_key": "sk-xxx",
    "openai_model": "gpt-4o-mini",

    "bedrock_model_14b": "deepseek.r1-distill-qwen-14b",
    "bedrock_model_32b": "deepseek.r1-distill-qwen-32b",
    "bedrock_region": "us-west-2",
    "bedrock_max_tokens": 32768,
    "bedrock_control_thinking": false,

    "strands_provider": "bedrock",
    "strands_model_14b": "us.deepseek.r1-distill-qwen-14b",
    "strands_model_32b": "us.deepseek.r1-distill-qwen-32b",
    "strands_region": "us-west-2",
    "strands_max_tokens": 32768,
    "strands_temperature": 0.7,
    "strands_control_thinking": false,
    "strands_streaming": false,

    "retry_attempts": 3,
    "retry_backoff_base": 1.0
}
```

### 配置项说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm_provider` | string | `"openai"` | LLM provider: `"openai"`, `"bedrock"`, `"strands"` |
| `strands_provider` | string | `"bedrock"` | Strands 底层 provider: `"bedrock"`, `"openai"` |
| `strands_model_14b` | string | - | 14B 模型 ID |
| `strands_model_32b` | string | - | 32B 模型 ID |
| `strands_region` | string | `"us-west-2"` | AWS Region (Bedrock) |
| `strands_max_tokens` | int | `32768` | 最大输出 token 数 |
| `strands_temperature` | float | `0.7` | 采样温度 |
| `strands_control_thinking` | bool | `false` | 是否跳过 thinking |
| `strands_streaming` | bool | `false` | 是否启用流式输出 |

---

## Module Integration

### 修改示例: run_pipeline.py

```python
def create_llm_clients(config: dict):
    """Create LLM clients based on configuration."""
    provider = config.get("llm_provider", "openai")

    if provider == "strands":
        from utils import StrandsClient

        strands_provider = config.get("strands_provider", "bedrock")
        region = config.get("strands_region", "us-west-2")
        max_tokens = config.get("strands_max_tokens", 32768)
        temperature = config.get("strands_temperature", 0.7)
        control_thinking = config.get("strands_control_thinking", False)
        streaming = config.get("strands_streaming", False)

        # 14B for QExpand/QueryGen
        llm_client_14b = StrandsClient(
            model_id=config["strands_model_14b"],
            provider=strands_provider,
            region=region,
            temperature=temperature,
            max_tokens=max_tokens,
            control_thinking=control_thinking,
            streaming=streaming
        )

        # 32B for Rerank/Judge
        llm_client_32b = StrandsClient(
            model_id=config["strands_model_32b"],
            provider=strands_provider,
            region=region,
            temperature=temperature,
            max_tokens=max_tokens,
            control_thinking=control_thinking,
            streaming=streaming
        )

        return llm_client_14b, llm_client_32b

    elif provider == "bedrock":
        # ... existing BedrockClient code ...

    else:  # openai
        # ... existing OpenAIClient code ...
```

---

## API Interface

### Type Alias

```python
from typing import Union

# Type alias for all LLM clients
LLMClient = Union[OpenAIClient, BedrockClient, StrandsClient]
```

### Response Format

| Method | OpenAIClient | BedrockClient | StrandsClient |
|--------|--------------|---------------|---------------|
| `chat_completion()` | Returns `str` | Returns `List[str]` | Returns `List[str]` |

所有使用 `BedrockClient` 或 `StrandsClient` 的地方都返回 `List[str]`，保持一致性。

---

## Dependencies

### 新增 Python 依赖

```txt
strands-agents>=0.1.0
strands-agents-tools>=0.1.0  # Optional, for tool support
```

### 安装命令

```bash
pip install strands-agents strands-agents-tools
```

---

## Testing

### 单元测试 (`tests/test_strands_client.py`)

```python
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Mock strands before import
if 'strands' not in sys.modules:
    mock_strands = MagicMock()
    sys.modules['strands'] = mock_strands
    sys.modules['strands.models'] = MagicMock()

from utils import StrandsClient, StrandsError


class TestStrandsClientInit(unittest.TestCase):
    """Test StrandsClient initialization."""

    @patch('strands.Agent')
    @patch('strands.models.BedrockModel')
    def test_bedrock_initialization(self, mock_bedrock_model, mock_agent):
        client = StrandsClient(
            model_id="test-model",
            provider="bedrock",
            region="us-west-2"
        )
        mock_bedrock_model.assert_called_once()
        mock_agent.assert_called_once()


class TestStrandsClientChatCompletion(unittest.TestCase):
    """Test chat completion methods."""

    @patch('strands.Agent')
    @patch('strands.models.BedrockModel')
    def test_single_completion(self, mock_bedrock_model, mock_agent):
        mock_result = Mock()
        mock_result.message = "Test response"
        mock_agent.return_value.return_value = mock_result

        client = StrandsClient(model_id="test-model")
        result = client.chat_completion(
            [{"role": "user", "content": "test"}],
            n=1
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Test response")


class TestStrandsClientThinking(unittest.TestCase):
    """Test thinking mode handling."""

    def test_extract_answer_with_think_tag(self):
        client = StrandsClient.__new__(StrandsClient)
        client.control_thinking = False

        response = "<think>thinking process</think>\nFinal answer"
        result = client._extract_answer(response)

        self.assertEqual(result, "Final answer")


if __name__ == '__main__':
    unittest.main()
```

---

## Implementation Checklist

- [ ] `utils.py`: 新增 `StrandsError` 异常类
- [ ] `utils.py`: 新增 `StrandsClient` 类
  - [ ] `__init__` 初始化 Strands Agent
  - [ ] `chat_completion` 主方法
  - [ ] `_single_call` 单次调用
  - [ ] `_build_prompt` 构建 prompt
  - [ ] `_extract_answer` 提取答案
- [ ] `run_pipeline.py`: 新增 strands provider 选择
- [ ] `config.example.json`: 新增 Strands 配置项
- [ ] `tests/test_strands_client.py`: 单元测试
- [ ] 更新 `LLMClient` type alias

---

## Migration Guide

### 从 BedrockClient 迁移到 StrandsClient

1. 修改 `config.json`:
```json
{
    "llm_provider": "strands",
    "strands_provider": "bedrock",
    "strands_model_14b": "us.deepseek.r1-distill-qwen-14b",
    "strands_model_32b": "us.deepseek.r1-distill-qwen-32b"
}
```

2. 代码兼容性: 无需修改调用代码，`StrandsClient` 提供相同的 `chat_completion` 接口。

### 优势

1. **简化代码**: 利用 Strands SDK 内置功能
2. **多 Provider 支持**: 轻松切换 Bedrock/OpenAI/其他
3. **未来扩展**: 可添加 Tools/MCP 支持
4. **社区维护**: AWS 官方维护的 SDK
