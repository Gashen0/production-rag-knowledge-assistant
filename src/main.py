"""FastAPI 应用入口：提供问答 API 与文档管理接口。"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.retriever import HybridRetriever
from rag.memory import ConversationMemory
from evaluation.evaluator import RAGEvaluator
from utils.document_loader import DocumentLoader


class QueryRequest(BaseModel):
    """查询请求体。"""
    query: str
    session_id: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """查询响应体。"""
    answer: str
    sources: list[dict]
    citations: list[str]


# 全局依赖实例（生产环境应使用依赖注入框架）
retriever: HybridRetriever | None = None
memory: ConversationMemory | None = None
evaluator: RAGEvaluator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """应用生命周期管理。"""
    global retriever, memory, evaluator
    
    # 启动时初始化
    retriever = HybridRetriever()
    memory = ConversationMemory()
    evaluator = RAGEvaluator()
    
    yield
    
    # 关闭时清理
    # TODO: 清理持久化连接


app = FastAPI(title="Production RAG Knowledge Assistant", lifespan=lifespan)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """执行问答查询。"""
    if retriever is None or memory is None:
        raise HTTPException(status_code=500, detail="服务未初始化")
    
    # TODO: 获取对话历史
    history = memory.get_history(request.session_id)
    
    # TODO: 执行检索
    # results = await retriever.retrieve(request.query, top_k=request.top_k)
    
    # TODO: 调用 LLM 生成答案
    answer = "占位：生成的答案"
    sources: list[dict] = []
    citations: list[str] = []
    
    # TODO: 更新对话历史
    # memory.add_turn(request.session_id, request.query, answer)
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        citations=citations,
    )


@app.post("/documents/upload")
async def upload_document(file_path: str) -> dict:
    """上传并处理文档。"""
    loader = DocumentLoader()
    # TODO: 解析文档、分块、嵌入并入库
    return {"status": "processing", "file": file_path}


@app.get("/health")
async def health_check() -> dict:
    """健康检查。"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)