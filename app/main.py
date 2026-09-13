"""
NutriLife FastAPI 应用入口。

职责：
- 创建并配置 FastAPI 实例
- 注册全局中间件（CORS、日志等）
- 挂载 API 路由
- 定义生命周期钩子（startup / shutdown）
"""
from llama_index.core import Settings
# from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.agents.graph import close_checkpointer
from app.core.config import get_settings
from app.core.database import close_db, init_db
from app.core.llm import NormalizedBGEEmbedding

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理：启动时预热资源，关闭时释放连接。"""
    settings = get_settings()
    logger.info(
        "NutriLife API 启动中 [env={}, debug={}]",
        settings.app.env,
        settings.app.debug,
    )

    lm = settings.lm_studio
    llm_cfg = settings.llm

    _temperature = llm_cfg.temperature
    _max_tokens = llm_cfg.max_tokens
    _streaming = llm_cfg.streaming

    Settings.embed_model = NormalizedBGEEmbedding(
        model_name=settings.lm_studio.embed_model,
        api_base="http://192.168.31.48:1234/v1",
        api_key=lm.api_key
    )
    # Settings.llm = OpenAI(
    #     model=lm.model,
    #     base_url=lm.base_url,
    #     api_key=lm.api_key,
    #     temperature=_temperature,
    #     max_tokens=_max_tokens,
    #     streaming=_streaming,
    #     request_timeout=llm_cfg.request_timeout
    # )
    Settings.llm = OpenAILike(
        api_base=lm.base_url,
        api_key=lm.api_key,
        model=lm.base_url,
        temperature=_temperature,
        max_tokens=_max_tokens,
        is_chat_model=True,
        context_window=32768,
        timeout=llm_cfg.request_timeout,  # 本地小模型推理慢，timeout设 10 分钟保底
    )
    await init_db()
    yield
    await close_checkpointer()
    await close_db()
    logger.info("NutriLife API 关闭，释放资源...")


def create_app() -> FastAPI:
    """FastAPI 应用工厂函数。"""
    settings = get_settings()

    app = FastAPI(
        title=settings.app.title,
        version=settings.app.version,
        debug=settings.app.debug,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 路由挂载
    from app.api.v1 import chat

    app.include_router(chat.router, prefix=settings.app.api_prefix)

    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        """健康检查根端点。"""
        return {"service": "NutriLife API", "status": "healthy"}

    return app


app = create_app()
