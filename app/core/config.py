"""
NutriLife 全局配置模块。

使用 Pydantic-Settings 从环境变量或 .env 文件中读取所有配置项，
提供类型安全的配置访问，并在应用启动时进行验证。
"""

from functools import cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 将 .env 加载到 os.environ：pydantic-settings 的嵌套模型（AppSettings / LangFuseSettings
# 等）只从 os.environ 读取，顶层 env_file 不会自动注入；LangFuse SDK 也只读环境变量。
load_dotenv()


class LMStudioSettings(BaseSettings):
    """LM Studio 本地 LLM 服务配置。"""

    model_config = SettingsConfigDict(env_prefix="LM_STUDIO_")

    base_url: str = Field(
        default="http://192.168.31.48:1234/v1",
        description="LM Studio OpenAI 兼容 API 的 base URL",
    )
    api_key: str = Field(
        default="lm-studio",
        description="LM Studio 不需要真实 API Key，填任意字符串即可",
    )
    model: str = Field(
        default="google/gemma-4-12b-qat",
        description="在 LM Studio 中加载的对话模型名称（需与 /v1/models 的 id 完全一致）",
    )
    embed_model: str = Field(
        default="text-embedding-baai-bge-m3-568m",
        description="在 LM Studio 中加载的 Embedding 模型名称（供 RAG 向量化使用）",
    )


class LLMSettings(BaseSettings):
    """LLM 推理超参数配置。"""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="采样温度，值越低输出越确定性。小模型推荐 0.1",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=32768,
        description="单次生成的最大 Token 数",
    )
    streaming: bool = Field(
        default=True,
        description="是否启用流式输出",
    )
    request_timeout: int = Field(
        default=120,
        description="LLM 请求超时时间（秒）",
    )


class LangFuseSettings(BaseSettings):
    """LangFuse 可观测性平台配置。"""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_")

    secret_key: str = Field(
        default="",
        description="LangFuse Secret Key (sk-lf-...)",
    )
    public_key: str = Field(
        default="",
        description="LangFuse Public Key (pk-lf-...)",
    )
    host: str = Field(
        default="http://localhost:3000",
        description="LangFuse 服务地址（本地自托管 Docker）",
    )
    enabled: bool = Field(
        default=True,
        description="是否启用 LangFuse 追踪，关闭后不发送任何遥测数据",
    )

    @field_validator("enabled", mode="before")
    @classmethod
    def disable_if_no_keys(cls, v: bool, info: object) -> bool:
        """若未提供 Secret Key 则自动禁用 LangFuse，防止启动报错。"""
        return v


class DatabaseSettings(BaseSettings):
    """数据库连接配置。"""

    model_config = SettingsConfigDict(env_prefix="")

    database_url: str = Field(
        default="postgresql+psycopg://jerry:jerry@localhost:5432/nutrilife",
        description="SQLAlchemy 异步数据库连接字符串",
    )


class RAGSettings(BaseSettings):
    """RAG 检索增强生成配置。"""

    model_config = SettingsConfigDict(env_prefix="")

    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant 向量数据库服务地址（本地 Docker）",
    )
    qdrant_api_key: str = Field(
        default="",
        description="Qdrant API Key（本地部署通常留空）",
    )
    qdrant_collection_name: str = Field(
        default="nutrilife_kb",
        description="Qdrant Collection 名称",
    )
    vector_top_k: int = Field(
        default=10,
        description="向量检索召回的候选数量",
    )
    bm25_top_k: int = Field(
        default=10,
        description="BM25 关键词检索召回的候选数量",
    )
    fusion_top_k: int = Field(
        default=8,
        description="RRF 融合后保留的候选数量（进入 Reranker 之前）",
    )
    enable_rerank: bool = Field(
        default=False,
        description="是否启用 Reranker",
    )
    rerank_top_n: int = Field(
        default=5,
        description="Reranker 最终输出的节点数量",
    )
    similarity_cutoff: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="向量相似度门控阈值，低于此值时判定知识库中无相关内容",
    )
    num_queries: int = Field(
        default=1,
        description="QueryFusionRetriever 生成的查询变体数量",
    )
    use_async: bool = Field(
        default=False,
        description="是否使用异步模式进行并行检索",
    )
    node_cache_path: str = Field(
        default="./data/nodes.json",
        description="切分后节点缓存文件（供 BM25 检索器重建语料）",
    )
    knowledge_base_dir: str = Field(
        default="./data/knowledge_base",
        description="知识库文档原始文件目录",
    )
    rag_top_k: int = Field(
        default=5,
        description="RAG 检索返回的最大文档片段数",
    )
    chunk_size: int = Field(
        default=512,
        description="文档分块大小（Token 数）",
    )
    chunk_overlap: int = Field(
        default=50,
        description="相邻文档块的重叠 Token 数",
    )


class OpenFoodFactsMCPSettings(BaseSettings):
    """Open Food Facts MCP 服务配置。"""

    model_config = SettingsConfigDict(env_prefix="OPENFOODFACTS_MCP_")

    url: str = Field(
        default="http://localhost:28375",
        description="Open Food Facts MCP 服务地址（不含 /mcp 路径）",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=0.1,
        le=120.0,
        description="调用 Open Food Facts MCP 工具的超时时间（秒）",
    )


class AppSettings(BaseSettings):
    """应用级别配置。"""

    model_config = SettingsConfigDict(env_prefix="APP_")

    env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="运行环境标识",
    )
    debug: bool = Field(
        default=False,
        description="是否开启 Debug 模式（生产环境务必关闭）",
    )
    secret_key: str = Field(
        default="change-me-in-production",
        description="应用密钥，用于签发 JWT 等敏感操作",
    )
    title: str = Field(default="NutriLife API", description="API 文档标题")
    version: str = Field(default="0.1.0", description="API 版本号")
    api_prefix: str = Field(default="/api/v1", description="全局路由前缀")
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="允许的 CORS 来源列表",
    )


class Settings(BaseSettings):
    """
    NutriLife 顶层配置聚合类。

    所有子配置通过组合方式聚合到此类，
    外部代码统一通过 `get_settings()` 获取实例。

    示例::

        from app.core.config import get_settings

        settings = get_settings()
        print(settings.llm.temperature)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    lm_studio: LMStudioSettings = Field(default_factory=LMStudioSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    langfuse: LangFuseSettings = Field(default_factory=LangFuseSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    openfoodfacts_mcp: OpenFoodFactsMCPSettings = Field(
        default_factory=OpenFoodFactsMCPSettings
    )


@cache
def get_settings() -> Settings:
    """
    获取全局配置单例（通过 lru_cache 缓存，进程内只实例化一次）。

    Returns:
        Settings: 已从环境变量 / .env 文件加载的配置实例。
    """
    return Settings()
