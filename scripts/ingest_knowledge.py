"""构建Chroma索引脚本"""

import logging
import sys
import os

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.chroma_store import get_chroma_store
from src.rag.document_loader import load_all_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("开始构建知识库索引...")

    # 加载文档
    chunks = load_all_documents()
    if not chunks:
        logger.warning("未找到任何知识库文档，请检查 data/knowledge_base/ 目录")
        return

    # 准备数据
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]

    # 写入Chroma
    store = get_chroma_store()
    store.reset()  # 清除旧数据
    store.add_documents(documents=texts, metadatas=metadatas, ids=ids)

    count = store.count()
    logger.info(f"知识库索引构建完成! 共 {count} 条文档")


if __name__ == "__main__":
    main()
