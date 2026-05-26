"""文档加载与分块：支持多格式 + 智能语义分块。"""

from typing import List, Dict, Any, Optional, Protocol
from pathlib import Path
import re

# TODO: 接入实际的文档解析库
# from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
# import tiktoken


class DocumentParser(Protocol):
    """文档解析器协议。"""
    def parse(self, file_path: str) -> str: ...


class DocumentChunker(Protocol):
    """文档分块器协议。"""
    def split(self, text: str) -> List[str]: ...


class DocumentLoader:
    """
    文档加载器：支持多格式解析与智能分块。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_semantic_boundary: bool = True,
    ) -> None:
        """
        Args:
            chunk_size: 目标分块大小（字符数）
            chunk_overlap: 相邻块之间的重叠字符数
            use_semantic_boundary: 是否优先使用语义边界（段落、章节）进行分块
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_boundary = use_semantic_boundary

        # 支持的文件类型映射
        self._parsers: Dict[str, Any] = {}
        self._chunkers: Dict[str, Any] = {}

    def infer_file_type(self, file_path: str) -> str:
        """
        根据文件扩展名推断文件类型。

        Returns:
            文件类型标识（如 "pdf", "md", "docx" 等）
        """
        suffix = Path(file_path).suffix.lower()
        type_map = {
            ".pdf": "pdf",
            ".md": "markdown",
            ".txt": "text",
            ".docx": "docx",
            ".html": "html",
            ".json": "json",
        }
        return type_map.get(suffix, "unknown")

    def _parse_pdf(self, file_path: str) -> str:
        """解析 PDF 文档。"""
        # TODO: 接入 PyPDF 或 Unstructured
        # loader = PyPDFLoader(file_path)
        # pages = loader.load()
        # return "\n".join(page.page_content for page in pages)
        return "占位：PDF 内容"

    def _parse_markdown(self, file_path: str) -> str:
        """解析 Markdown 文档。"""
        # TODO: 接入 Markdown 解析器
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_text(self, file_path: str) -> str:
        """解析纯文本文档。"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def parse(self, file_path: str) -> str:
        """
        解析文件内容。

        根据文件类型选择对应的解析器。
        """
        file_type = self.infer_file_type(file_path)
        
        if file_type == "pdf":
            return self._parse_pdf(file_path)
        elif file_type == "markdown":
            return self._parse_markdown(file_path)
        elif file_type in ("text", "json"):
            return self._parse_text(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")

    def _split_by_semantic_boundary(self, text: str) -> List[str]:
        """
        基于语义边界（段落、章节标题）进行分块。

        优先保持段落完整性，只在段落过长时按 chunk_size 切分。
        """
        # 按段落切割
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks: List[str] = []
        current_chunk = ""
        
        for para in paragraphs:
            # 如果加入当前段落后不超过 chunk_size，直接加入
            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                # 当前段落会导致溢出，保存当前 chunk 并开始新的
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
                
                # 如果单个段落超过 chunk_size，需要进一步切分
                while len(current_chunk) > self.chunk_size:
                    chunks.append(current_chunk[:self.chunk_size])
                    current_chunk = current_chunk[self.chunk_size - self.chunk_overlap:]
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _split_by_fixed_size(self, text: str) -> List[str]:
        """
        基于固定大小的滑动窗口分块。

        保证每个 chunk 的长度尽量接近 chunk_size，
        相邻 chunk 之间有 chunk_overlap 的重叠。
        """
        if not text:
            return []

        chunks: List[str] = []
        start = 0

        while start < len(text):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        将文本切分为 Chunk 列表。

        Args:
            text: 输入文本

        Returns:
            Chunk 列表，每项包含 text, index, metadata
        """
        if self.use_semantic_boundary:
            raw_chunks = self._split_by_semantic_boundary(text)
        else:
            raw_chunks = self._split_by_fixed_size(text)

        return [
            {
                "text": chunk,
                "chunk_idx": idx,
                "metadata": {
                    "source": "loaded_document",
                    "chunk_size": len(chunk),
                    "split_method": "semantic" if self.use_semantic_boundary else "fixed",
                },
            }
            for idx, chunk in enumerate(raw_chunks)
        ]

    def load_and_chunk(self, file_path: str) -> List[Dict[str, Any]]:
        """
        一站式加载并分块。

        Args:
            file_path: 文件路径

        Returns:
            处理后的 Chunk 列表，可直接用于向量入库
        """
        # 1. 解析文档
        raw_text = self.parse(file_path)

        # 2. 智能分块
        chunks = self.chunk(raw_text)

        # 3. 添加文件级别的元数据
        for c in chunks:
            c["metadata"]["file_path"] = file_path
            c["metadata"]["file_type"] = self.infer_file_type(file_path)

        return chunks