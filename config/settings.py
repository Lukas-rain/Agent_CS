"""集中配置管理 - API keys, 模型名, 路径, 参数"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── 智谱AI ──
ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_VISION_MODEL: str = os.getenv("ZHIPU_VISION_MODEL", "glm-4v-flash")
ZHIPU_TEXT_MODEL: str = os.getenv("ZHIPU_TEXT_MODEL", "glm-4-flash")

# ── Chroma向量库 ──
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME: str = "knowledge_base"

# ── 知识库 ──
KNOWLEDGE_BASE_DIR: str = os.getenv("KNOWLEDGE_BASE_DIR", "./data/knowledge_base")

# ── Agent参数 ──
HALLUCINATION_THRESHOLD: float = float(os.getenv("HALLUCINATION_THRESHOLD", "0.6"))
MAX_RETRY_COUNT: int = int(os.getenv("MAX_RETRY_COUNT", "2"))
MEMORY_SUMMARY_THRESHOLD: int = int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "20"))
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RECENT_MEMORY_WINDOW: int = 10  # 最近N条消息

# ── 文档分块 ──
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
