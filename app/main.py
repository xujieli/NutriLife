"""
NutriLife FastAPI 应用入口。

职责：
- 创建并配置 FastAPI 实例
- 注册全局中间件（CORS、日志等）
- 挂载 API 路由
- 定义生命周期钩子（startup / shutdown）
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings
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

    Settings.embed_model = NormalizedBGEEmbedding(
        model_name=settings.lm_studio.embed_model,
        api_base=lm.base_url,
        api_key=lm.api_key,
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
    from app.api.v1 import auth, chat, sessions

    app.include_router(chat.router, prefix=settings.app.api_prefix)
    app.include_router(auth.router, prefix=settings.app.api_prefix)
    app.include_router(sessions.router, prefix=settings.app.api_prefix)

    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        """健康检查根端点。"""
        return {"service": "NutriLife API", "status": "healthy"}

    return app


app = create_app()
