"""多模态客服智能体 - 启动入口"""

import logging
import os
import sys

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.ui.gradio_app import launch_ui

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 50)
    logger.info("多模态客服智能体 启动中...")
    logger.info("=" * 50)

    # 检查API Key
    from config.settings import ZHIPU_API_KEY
    if not ZHIPU_API_KEY or ZHIPU_API_KEY == "your_zhipu_api_key_here":
        logger.error("请先在 .env 文件中设置 ZHIPU_API_KEY")
        logger.error("参考 .env.example 文件创建 .env 并填入你的API Key")
        sys.exit(1)

    # 检查Chroma索引
    from config.settings import CHROMA_PERSIST_DIR
    if not os.path.exists(CHROMA_PERSIST_DIR):
        logger.warning(f"Chroma索引目录不存在: {CHROMA_PERSIST_DIR}")
        logger.warning("请先运行: python scripts/ingest_knowledge.py 构建知识库索引")

    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    logger.info(f"启动Gradio界面 (端口:{port}, 分享:{share})")
    launch_ui(share=share, server_port=port)


if __name__ == "__main__":
    main()
