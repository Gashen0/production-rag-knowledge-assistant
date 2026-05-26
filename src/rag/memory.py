"""对话记忆模块：滑动窗口 + 摘要 + 实体记忆。"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import time

# TODO: 接入实际 LLM 进行摘要生成
# from langchain.llms import BaseLLM


@dataclass
class ConversationTurn:
    """对话轮次。"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    entities: List[str] = field(default_factory=list)


@dataclass
class ConversationSummary:
    """对话摘要。"""
    summary_text: str
    key_constraints: List[str]
    last_updated: float


class ConversationMemory:
    """
    对话记忆管理器。

    采用混合策略：
    - 滑动窗口保留最近 K 轮完整对话
    - 窗口外的历史生成压缩摘要
    - 维护关键实体独立记忆池
    """

    def __init__(
        self,
        window_size: int = 3,
        max_history_turns: int = 20,
        enable_summarization: bool = True,
    ) -> None:
        """
        Args:
            window_size: 保留完整对话的轮次数
            max_history_turns: 触发摘要生成前最大保留轮数
            enable_summarization: 是否启用摘要压缩
        """
        self.window_size = window_size
        self.max_history_turns = max_history_turns
        self.enable_summarization = enable_summarization

        # 存储结构
        # key: session_id, value: { "turns": [], "summary": None, "entities": set() }
        self._sessions: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"turns": [], "summary": None, "entities": set()}
        )

    def _generate_summary(
        self, session_id: str, turns: List[ConversationTurn]
    ) -> Optional[ConversationSummary]:
        """
        生成对话摘要。

        利用 LLM 对历史对话进行压缩，保留关键约束和主题。
        """
        # TODO: 接入 LLM 生成摘要
        # prompt = f"将以下对话历史总结为关键点与用户约束：\n{dialogue_text}"
        # summary = llm.generate(prompt)

        # 占位实现
        summary_text = "用户关注技术文档与代码实现"
        key_constraints = ["需要源代码示例", "关注性能指标"]

        return ConversationSummary(
            summary_text=summary_text,
            key_constraints=key_constraints,
            last_updated=time.time(),
        )

    def _extract_entities(self, text: str) -> List[str]:
        """
        提取文本中的关键实体。

        使用规则 + NER 混合方式。
        """
        # TODO: 接入 NER 模型进行实体识别
        # 当前使用简单规则占位
        entities: List[str] = []
        # 检测技术术语/专有名词模式
        # entities.append(...)
        return entities

    def add_turn(
        self, session_id: str, role: str, content: str
    ) -> None:
        """
        添加对话轮次。

        如果超过 max_history_turns，触发摘要生成并压缩历史。
        """
        session = self._sessions[session_id]

        # 提取实体
        entities = self._extract_entities(content)

        turn = ConversationTurn(
            role=role,
            content=content,
            entities=entities,
        )

        # 更新实体记忆池
        session["entities"].update(entities)

        # 添加轮次
        session["turns"].append(turn)

        # 检查是否需要摘要压缩
        if len(session["turns"]) > self.max_history_turns and self.enable_summarization:
            # 保留最近 window_size 轮，其余压缩为摘要
            turns_to_summarize = session["turns"][: -self.window_size]
            new_summary = self._generate_summary(session_id, turns_to_summarize)

            if new_summary:
                # 合并摘要（如果已有历史摘要）
                if session["summary"]:
                    existing = session["summary"]
                    new_summary.summary_text = (
                        f"{existing.summary_text}\n{new_summary.summary_text}"
                    )

                session["summary"] = new_summary
                # 只保留最近 window_size 轮
                session["turns"] = session["turns"][-self.window_size :]

    def get_history(self, session_id: str) -> Dict[str, Any]:
        """
        获取格式化后的对话历史。

        返回包含摘要、近期轮次、实体记忆的完整上下文。
        """
        session = self._sessions.get(session_id)
        if not session:
            return {"summary": None, "recent_turns": [], "entities": []}

        return {
            "summary": session["summary"].summary_text if session["summary"] else None,
            "recent_turns": [
                {"role": t.role, "content": t.content}
                for t in session["turns"]
            ],
            "entities": list(session["entities"]),
        }

    def clear_session(self, session_id: str) -> None:
        """清空指定会话。"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_relevant_context(self, session_id: str, query: str) -> str:
        """
        根据查询动态组装最相关的上下文。

        将摘要、近期对话、相关实体组装为提示上下文。
        """
        history = self.get_history(session_id)

        parts: List[str] = []

        # 1. 摘要上下文
        if history["summary"]:
            parts.append(f"[对话摘要]\n{history['summary']}\n")

        # 2. 近期轮次
        if history["recent_turns"]:
            turns_text = "\n".join(
                f"{t['role']}: {t['content']}" for t in history["recent_turns"]
            )
            parts.append(f"[近期对话]\n{turns_text}\n")

        # 3. 相关实体（如果查询中匹配到历史实体）
        query_entities = self._extract_entities(query)
        relevant_entities = set(query_entities) & set(history["entities"])
        if relevant_entities:
            parts.append(f"[相关实体]\n{', '.join(relevant_entities)}\n")

        return "\n".join(parts) if parts else ""