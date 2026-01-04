# Medical Literature Retrieval Pipeline

Medical Literature Retrieval Pipeline 是 DIVER 项目的医学文献检索子系统，支持 OpenAI 和 AWS Bedrock 两种 LLM provider。

## Quick Start

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置
cp config.example.json config.json
# 编辑 config.json 填写你的配置

# 3. 运行
python run_pipeline.py
```

## LLM Provider 配置

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

### Model 分配策略

| 模块 | 模型 | 说明 |
|------|------|------|
| Query Generator | 14B | 合成查询生成 |
| Query Expansion | 14B | 迭代查询扩展 |
| Reranker | 32B | Listwise 重排序 |
| LLM Judge | 32B | 相关性评估 |

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm_provider` | string | `"openai"` | LLM provider: `"openai"` 或 `"bedrock"` |
| `bedrock_model_14b` | string | - | 14B 模型的 Bedrock Model ID |
| `bedrock_model_32b` | string | - | 32B 模型的 Bedrock Model ID |
| `bedrock_region` | string | `"us-west-2"` | AWS Region |
| `bedrock_max_tokens` | int | `32768` | 最大输出 token 数 |
| `bedrock_control_thinking` | bool | `false` | 是否跳过 thinking 过程 |
| `num_queries` | int | `20` | 生成的合成查询数 |
| `num_expansion_rounds` | int | `3` | 查询扩展轮数 |
| `rerank_top_k` | int | `20` | 重排序的文档数 |
| `eval_top_k` | int | `10` | 评估的 top-k 文档 |
| `bm25_alpha` | float | `0.3` | BM25 混合权重 |

## Thinking Mode

DeepSeek R1 模型会输出 `<think>...</think>` 思考过程。通过 `bedrock_control_thinking` 参数控制:

- `false` (默认): 让模型自然输出，自动提取 `</think>` 后的答案
- `true`: 在 prompt 末尾添加 `</think>` 跳过思考，直接输出答案

## Pipeline 流程

```
1. Load documents from S3
2. Generate synthetic queries (14B model)
3. Expand queries iteratively (14B model)
4. Retrieve documents (DIVER-4B + BM25)
5. Rerank candidates (32B model)
6. Evaluate with LLM judge (32B model)
7. Output CSV report to S3
```

## 运行测试

```bash
# 运行单元测试
python -m pytest tests/

# 运行 BedrockClient 测试
python -m pytest tests/test_bedrock_client.py -v
```

## 文件结构

```
MedicalRetrieval/
├── run_pipeline.py          # 主入口
├── utils.py                 # OpenAIClient, BedrockClient
├── query_generator.py       # 合成查询生成
├── qexpand_openai.py        # 查询扩展
├── retriever.py             # DIVER + BM25 混合检索
├── reranker_openai.py       # Listwise 重排序
├── llm_judge.py             # LLM Judge 评估
├── data_loader.py           # S3 数据加载
├── output_writer.py         # 输出管理
├── config.json              # 配置文件 (需自行创建)
├── config.example.json      # 配置示例
├── SPEC.md                  # Bedrock 集成技术规格
└── tests/
    └── test_bedrock_client.py  # BedrockClient 单元测试
```

## 依赖

### 必需

- `boto3>=1.34.0` - AWS SDK (Bedrock)
- `openai>=1.0.0` - OpenAI SDK
- `transformers>=4.40.0` - Chat template 支持

### 可选

- `sentence-transformers` - DIVER-Retriever
- `rank_bm25` - BM25 检索
- `jieba` - 中文分词

## 问题排查

### Bedrock 认证失败

确保环境变量正确设置:

```bash
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
```

或检查 IAM Role 权限是否包含:
- `bedrock:InvokeModel`
- `bedrock:Converse`

### Model ID 错误

检查 Bedrock 控制台中的实际 Model ID，格式可能是:
- `deepseek.r1-distill-qwen-14b`
- `us.deepseek.r1-distill-qwen-14b-v1:0`
- 或 Cross-Region Inference Profile ARN

### Token 限制

如果遇到 `ValidationException`，尝试减小 `bedrock_max_tokens`。
