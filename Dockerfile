# 使用官方 Python 3.11 slim 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY config/ ./config/

# 环境变量默认配置
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 暴露 FastAPI 和 Streamlit 端口
EXPOSE 8000
EXPOSE 8501

# 默认命令：启动 FastAPI 服务
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]