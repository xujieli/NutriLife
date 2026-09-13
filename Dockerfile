# NutriLife 后端生产镜像
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 先安装依赖（利用 Docker 层缓存；无 poetry.lock 时 poetry 会自动解析生成）
RUN pip install --upgrade pip && pip install poetry
COPY pyproject.toml ./
COPY poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --without dev,test --no-interaction --no-ansi

# 拷贝源码、知识库与脚本
COPY app ./app
COPY data ./data
COPY scripts ./scripts

EXPOSE 8000

# 启动前先初始化向量库（幂等，已存在则跳过），再启动 API
CMD ["sh", "-c", "python scripts/ingest.py --rebuild || true; uvicorn app.main:app --host 0.0.0.0 --port 8000"]
