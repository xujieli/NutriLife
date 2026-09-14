# NutriLife — 个人健康与饮食管理 Agent

> 基于 LangGraph + LlamaIndex + FastAPI 构建的 AI 健康顾问，接入本地 LM Studio (gemma-4-12b-qat)，具备全链路 LangFuse 可观测性。

---

## 🏗 项目架构

```
NutriLife/
├── app/
│   ├── main.py                  # FastAPI 应用入口 & 工厂函数
│   ├── api/
│   │   └── v1/
│   │       ├── chat.py          # 对话路由 (POST /chat)
│   │       ├── health.py        # 健康数据路由
│   │       └── nutrition.py     # 营养分析路由
│   ├── core/
│   │   ├── config.py            # Pydantic-Settings 全局配置
│   │   └── llm.py               # ChatOpenAI 初始化 & LangFuse Callback
│   ├── agents/
│   │   ├── graph.py             # LangGraph 状态机 (StateGraph)
│   │   ├── nodes.py             # 各 Agent 节点实现
│   │   └── state.py             # AgentState TypedDict 定义
│   ├── rag/
│   │   ├── pipeline.py          # LlamaIndex RAG 主管道
│   │   ├── indexer.py           # 文档索引 & 向量化
│   │   ├── retriever.py         # 混合检索器封装
│   │   ├── langchain_bridge.py  # LlamaIndex → LangChain 检索适配桥
│   │   └── query_engine.py      # RAG 数据组件 & 兼容查询引擎
│   ├── tools/
│   │   ├── nutrition_tools.py   # 食物营养查询工具
│   │   ├── calendar_tools.py    # 饮食日历工具
│   │   └── web_search_tools.py  # 网络搜索工具
│   ├── schemas/
│   │   ├── chat.py              # 对话 Pydantic 模型
│   │   ├── nutrition.py         # 营养数据模型
│   │   └── user.py              # 用户数据模型
│   └── db/
│       ├── session.py           # SQLAlchemy 异步 Session 工厂
│       └── models.py            # ORM 模型定义
├── tests/
│   ├── unit/                    # 单元测试
│   └── integration/             # 集成测试
├── data/
│   ├── knowledge_base/          # 营养学知识文档（PDF、MD 等）
│   └── nodes.json               # 切分节点缓存（供 BM25 检索）
├── pyproject.toml               # Poetry 依赖管理
├── .env.example                 # 环境变量模板
└── README.md
```

### 核心数据流

```
用户消息
  └─► FastAPI Router (api/v1/chat.py)
        └─► LangGraph StateGraph (agents/graph.py)
              ├─► 路由节点：意图识别
              ├─► RAG 节点：LlamaIndex 检索营养知识库
              ├─► Tool 节点：调用 LangChain Tools
              └─► 生成节点：ChatOpenAI (LM Studio) 生成回答
                    └─► LangFuse Callback → 追踪 Dashboard
```

---

## ⚙️ 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI 0.115 + Uvicorn |
| Agent 编排 | LangGraph 0.2 |
| RAG 管道 | LlamaIndex 0.11 + Qdrant |
| LLM | LM Studio (gemma-4-12b-qat)，OpenAI 兼容 API |
| 可观测性 | LangFuse 2.x |
| 结构化输出 | Instructor |
| 数据验证 | Pydantic v2 + Pydantic-Settings |
| 数据库 | SQLAlchemy 2 (AsyncIO) + SQLite / PostgreSQL |
| 前端 | React 18 + Vite + Tailwind CSS + shadcn/ui |

---

## 🚀 快速启动

### 前置条件

1. 安装 [LM Studio](https://lmstudio.ai/) 并加载 `gemma-4-12b-qat` 模型
2. 在 LM Studio 中启动本地服务器（默认端口 `1234`）
3. 安装 Python 3.12+ 和 [Poetry](https://python-poetry.org/)

### 后端启动

```bash
# 1. 克隆项目
git clone <repo-url> && cd NutriLife

# 2. 安装依赖
poetry install

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LangFuse Key（可选）

# 4. 初始化向量数据库（首次启动前执行一次）
poetry run python scripts/ingest.py

# 5. 启动开发服务器
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API 文档地址：http://localhost:8000/docs

### 前端启动

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

---

## 📦 部署指南

### 1. 导出 / 加载 LM Studio 模型

1. 打开 [LM Studio](https://lmstudio.ai/) → 搜索并下载 `gemma-4-12b-qat`（或任意 GGUF 模型）。
2. 在「My Models」中加载该模型。
3. 进入 **Developer** 页 → 启动 Local Server，端口保持 `1234`。
4. **重要**：勾选「Serve on Local Network」，否则局域网内其它机器无法访问。
5. 配置后端读取的服务地址：

```bash
LM_STUDIO_BASE_URL=http://[Server_IP]:1234/v1
LM_STUDIO_MODEL=gemma-4-12b-qat
```

验证服务可用：`curl http://[Server_IP]:1234/v1/models`

### 2. 初始化向量数据库

首次启动（或知识库更新后）需摄取文档并构建 Qdrant 向量索引：

```bash
# 首次构建
poetry run python scripts/ingest.py

# 强制重建（清空旧索引）
poetry run python scripts/ingest.py --rebuild

# 指定知识库目录与切分参数
poetry run python scripts/ingest.py \
  --kb-dir ./data/knowledge_base --chunk-size 512 --chunk-overlap 50
```

说明：
- `ingest.py` 读取 `data/knowledge_base/` 下的 PDF/TXT/MD 文档，用 `SentenceSplitter`（chunk_size=512、overlap=50）切分后写入 Qdrant（本地 Docker，默认 `http://localhost:6333`），同时把节点缓存到 `data/nodes.json` 供 BM25 检索。
- 检索采用 BM25 + 向量混合检索；若向量库未构建，RAG 节点会降级返回提示语而非崩溃。

### 3. 配置 LangFuse 环境变量

在 `.env` 中配置以下变量（缺省时 LangFuse 自动禁用，不影响功能）：

```bash
LANGFUSE_SECRET_KEY=sk-lf-...              # 必填，从 LangFuse 项目设置获取
LANGFUSE_PUBLIC_KEY=pk-lf-...              # 必填
LANGFUSE_HOST=http://localhost:3000             # 本地自托管 Docker 实例
```

配置后，Router / RAG / Workflow 每个节点、每次 LLM 调用都会自动上报为 LangFuse trace 树（可在 Dashboard 中查看具体 prompt 与 completion）。

### 4. Docker 部署

项目提供 `Dockerfile`（后端）、`frontend/Dockerfile`（前端）、`frontend/nginx.conf` 与 `docker-compose.yml`：

```bash
# 一键构建并启动前后端
docker compose up -d --build

# 查看日志
docker compose logs -f backend
```

- 前端：`http://localhost:5173`（nginx 托管静态资源，反向代理 `/api` 到后端）
- 后端：`http://localhost:8000`（含 SSE 流式）
- 向量库通过卷挂载到宿主机 `./data/`，容器重建不丢索引

注意：
- 后端容器启动时会自动执行 `ingest.py --rebuild` 初始化向量库（幂等）。
- `docker-compose.yml` 的 `LM_STUDIO_BASE_URL` 指向宿主机局域网 IP；若 LM Studio 与容器同机，改为 `http://host.docker.internal:1234/v1`。
- 首次构建前建议先执行 `poetry lock` 生成锁文件，以加速构建并固定依赖版本。

---

## 🔍 可观测性

- 在 [LangFuse Cloud](https://cloud.langfuse.com) 或自托管实例中配置 `LANGFUSE_SECRET_KEY` 和 `LANGFUSE_PUBLIC_KEY`
- 所有 LLM 调用、RAG 检索、Tool 调用均会自动上报 Trace
- 若未配置 Key，LangFuse 会自动禁用，不影响正常功能

---

## 🧪 测试

```bash
# 运行所有测试
poetry run pytest

# 带覆盖率报告
poetry run pytest --cov=app --cov-report=html

# 仅运行单元测试
poetry run pytest tests/unit/
```

---

## 📐 代码规范

```bash
# Lint & 格式化
poetry run ruff check . --fix
poetry run ruff format .

# 类型检查
poetry run mypy app/
```

---

## 📄 License

MIT
