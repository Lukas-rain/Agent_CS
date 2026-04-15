"""Gradio多模态聊天界面 - 柔紫薰衣草主题 (Gradio 6.x)"""

import asyncio
import logging
import uuid
from typing import Any

import gradio as gr

from src.graph.workflow import compile_agent, get_initial_state

logger = logging.getLogger(__name__)

# ── 会话管理 ──
_sessions: dict[str, dict[str, Any]] = {}
_compiled_graph = None

CSS = """
/* ═══════════════════════════════════════════
   柔紫薰衣草客服 UI - Soft Lavender Theme
   ═══════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

:root {
  --lavender-100: #F0EBFF;
  --lavender-200: #E0D6FF;
  --lavender-300: #C6B9FF;
  --lavender-400: #A78BFA;
  --lavender-500: #8C7CF0;
  --lavender-600: #7C5CFC;
  --lavender-700: #6D28D9;
  --soft-pink: #FBBFD0;
  --soft-green: #A8E6CF;
  --soft-yellow: #FFE5A0;
  --soft-orange: #FFD4B8;
  --bg-main: #FAFAFE;
  --bg-card: #FFFFFF;
  --text-primary: #2D2640;
  --text-secondary: #6B6380;
  --text-muted: #9B93AD;
  --shadow-soft: 0 2px 16px rgba(140, 124, 240, 0.08);
  --shadow-hover: 0 4px 24px rgba(140, 124, 240, 0.15);
  --radius: 16px;
  --radius-sm: 10px;
  --radius-pill: 999px;
}

/* ── 全局 ── */
* { font-family: 'Inter', 'Noto Sans SC', -apple-system, sans-serif !important; }
body, .gradio-container {
  background: var(--bg-main) !important;
  max-width: 880px !important;
  margin: 0 auto !important;
}

/* ── 顶部横幅 ── */
.hero-banner {
  background: linear-gradient(135deg, var(--lavender-400) 0%, var(--lavender-500) 40%, #C084FC 100%);
  border-radius: 0 0 32px 32px !important;
  padding: 36px 40px 28px !important;
  margin: -16px -16px 0 -16px !important;
  text-align: center;
  position: relative;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(140, 124, 240, 0.25);
}
.hero-banner::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at 30% 40%, rgba(255,255,255,0.12) 0%, transparent 50%),
              radial-gradient(circle at 70% 60%, rgba(255,255,255,0.08) 0%, transparent 40%);
  pointer-events: none;
}
.hero-banner h1 {
  color: white !important;
  font-size: 26px !important;
  font-weight: 700 !important;
  margin: 0 0 6px !important;
  position: relative;
  letter-spacing: 0.5px;
}
.hero-banner p {
  color: rgba(255,255,255,0.85) !important;
  font-size: 14px !important;
  margin: 0 !important;
  font-weight: 400 !important;
  position: relative;
}
.hero-icon {
  font-size: 40px;
  margin-bottom: 8px;
  display: block;
}

/* ── 状态栏 ── */
.status-bar {
  background: var(--bg-card);
  border-radius: var(--radius);
  padding: 12px 20px;
  margin: 16px 0 0;
  box-shadow: var(--shadow-soft);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--soft-green);
  display: inline-block;
  margin-right: 6px;
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.6; transform: scale(0.85); }
}

/* ── 聊天区域 ── */
.chat-area {
  background: var(--bg-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow-soft);
  border: 1px solid var(--lavender-100);
  margin-top: 16px;
  overflow: hidden;
}
.chatbot-row {
  min-height: 420px;
  max-height: 520px;
}

/* 用户气泡 */
.msg-user .message-bubble {
  background: linear-gradient(135deg, var(--lavender-400), var(--lavender-500)) !important;
  color: white !important;
  border-radius: 18px 18px 4px 18px !important;
  box-shadow: 0 2px 12px rgba(140, 124, 240, 0.2) !important;
  padding: 12px 18px !important;
  font-size: 14.5px !important;
  line-height: 1.6 !important;
}

/* 助手气泡 */
.msg-bot .message-bubble {
  background: var(--bg-main) !important;
  color: var(--text-primary) !important;
  border-radius: 18px 18px 18px 4px !important;
  border: 1px solid var(--lavender-100) !important;
  box-shadow: 0 1px 8px rgba(140, 124, 240, 0.06) !important;
  padding: 12px 18px !important;
  font-size: 14.5px !important;
  line-height: 1.7 !important;
}

/* ── 输入区域 ── */
.input-area {
  background: var(--bg-card);
  border-radius: var(--radius);
  box-shadow: var(--shadow-soft);
  border: 1px solid var(--lavender-100);
  margin-top: 12px;
  padding: 8px !important;
}

/* ── 功能按钮行 ── */
.action-bar {
  display: flex;
  gap: 10px;
  margin-top: 12px;
  justify-content: center;
  flex-wrap: wrap;
}
.action-btn {
  background: var(--bg-card) !important;
  border: 1.5px solid var(--lavender-200) !important;
  border-radius: var(--radius-pill) !important;
  color: var(--lavender-500) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 8px 20px !important;
  cursor: pointer;
  transition: all 0.2s ease !important;
  box-shadow: 0 1px 6px rgba(140,124,240,0.06) !important;
}
.action-btn:hover {
  background: var(--lavender-100) !important;
  border-color: var(--lavender-400) !important;
  box-shadow: var(--shadow-hover) !important;
  transform: translateY(-1px);
}

/* 快捷问题标签 */
.quick-tags {
  display: flex;
  gap: 8px;
  margin-top: 12px;
  flex-wrap: wrap;
  justify-content: center;
}
.quick-tag {
  background: linear-gradient(135deg, var(--lavender-100), rgba(251,191,208,0.2)) !important;
  border: 1px solid var(--lavender-200) !important;
  border-radius: var(--radius-pill) !important;
  color: var(--lavender-600) !important;
  font-size: 12.5px !important;
  padding: 6px 14px !important;
  cursor: pointer;
  transition: all 0.2s ease !important;
}
.quick-tag:hover {
  background: var(--lavender-200) !important;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(140,124,240,0.15) !important;
}

/* ── 隐藏默认footer ── */
footer { display: none !important; }

/* ── 会话ID区域 ── */
.session-row {
  display: flex;
  gap: 8px;
  align-items: center;
  justify-content: center;
  margin-top: 8px;
}
.session-row input {
  background: var(--lavender-100) !important;
  border: 1px solid var(--lavender-200) !important;
  border-radius: var(--radius-pill) !important;
  color: var(--text-secondary) !important;
  font-size: 12px !important;
  text-align: center;
}

/* ── 打字中动画 ── */
.typing-indicator span {
  display: inline-block;
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--lavender-300);
  margin: 0 2px;
  animation: typing 1.4s infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30% { transform: translateY(-6px); opacity: 1; }
}
"""


def _get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = compile_agent()
    return _compiled_graph


def _make_msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def get_session_state(session_id: str) -> dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = get_initial_state(
            user_input="", images=[], session_id=session_id
        )
    return _sessions[session_id]


def run_agent_sync(message: dict, history: list, session_id: str) -> str:
    user_text = message.get("text", "").strip()
    files = message.get("files", [])
    if not user_text and not files:
        return "请输入您的问题或上传图片。"

    state = get_session_state(session_id)
    state["user_input"] = user_text
    state["images"] = [f for f in files if f] if files else []

    compiled = _get_compiled_graph()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(
                lambda: asyncio.run(compiled.ainvoke(state))
            ).result()
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(compiled.ainvoke(state))

    if result:
        state["conversation_history"] = result.get(
            "conversation_history", state.get("conversation_history", [])
        )
        state["conversation_summary"] = result.get(
            "conversation_summary", state.get("conversation_summary", "")
        )
        state["message_count"] = result.get(
            "message_count", state.get("message_count", 0)
        )

    return result.get("final_response", "抱歉，处理您的请求时出现了问题。")


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="多模态客服智能体") as demo:

        # ── 顶部横幅 ──
        gr.HTML("""
        <div class="hero-banner">
            <span class="hero-icon">&#x1F96B;</span>
            <h1>智能客服助手</h1>
            <p>基于知识库的 AI 客服 · 支持图文问答 · 7x24 在线</p>
        </div>
        """)

        # ── 状态栏 ──
        gr.HTML("""
        <div class="status-bar">
            <div style="display:flex;align-items:center;gap:6px;">
                <span class="status-dot"></span>
                <span style="color:var(--text-secondary);font-size:13px;font-weight:500;">在线服务中</span>
            </div>
            <div style="color:var(--text-muted);font-size:12px;">powered by GLM-4V + RAG</div>
        </div>
        """)

        # ── 聊天区域 ──
        with gr.Column(elem_classes=["chat-area"]):
            chatbot = gr.Chatbot(
                label=None,
                show_label=False,
                value=[_make_msg("assistant", "你好！我是智能客服助手，有什么可以帮你的吗？")],
                elem_classes=["chatbot-row"],
            )

        # ── 快捷问题 ──
        with gr.Row():
            quick_q1 = gr.Button("如何重置密码？", elem_classes=["quick-tag"])
            quick_q2 = gr.Button("产品有哪些功能？", elem_classes=["quick-tag"])
            quick_q3 = gr.Button("如何申请退款？", elem_classes=["quick-tag"])
            quick_q4 = gr.Button("支持哪些付款方式？", elem_classes=["quick-tag"])

        # ── 输入区域 ──
        with gr.Column(elem_classes=["input-area"]):
            msg_input = gr.MultimodalTextbox(
                placeholder="输入您的问题，或拖拽上传图片...",
                sources=["upload"],
                file_types=["image"],
                show_label=False,
                submit_btn=True,
                stop_btn=True,
            )

        # ── 操作按钮 ──
        with gr.Row(elem_classes=["action-bar"]):
            session_id_input = gr.Textbox(
                value="default",
                visible=False,
            )
            clear_btn = gr.Button("清除对话", elem_classes=["action-btn"])
            new_btn = gr.Button("新会话", elem_classes=["action-btn"])

        # ── 事件处理 ──
        def respond(message: dict, history: list, session_id: str):
            reply = run_agent_sync(message, history, session_id)
            history = history or []
            user_text = message.get("text", "")
            files = message.get("files", [])
            content_blocks = []
            for f in files:
                content_blocks.append({"type": "file", "file": {"path": f}})
            display = user_text
            if files:
                display += f"\n（已上传 {len(files)} 张图片）"
            content_blocks.append({"type": "text", "text": display})
            history.append({"role": "user", "content": content_blocks})
            history.append(_make_msg("assistant", reply))
            return history, gr.MultimodalTextbox(
                value=None, placeholder="输入您的问题，或拖拽上传图片..."
            )

        def quick_ask(question: str, history: list, session_id: str):
            return respond({"text": question, "files": []}, history, session_id)

        def clear_chat(session_id: str):
            if session_id in _sessions:
                del _sessions[session_id]
            return (
                [_make_msg("assistant", "你好！我是智能客服助手，有什么可以帮你的吗？")],
                gr.MultimodalTextbox(
                    value=None, placeholder="输入您的问题，或拖拽上传图片..."
                ),
            )

        def new_session():
            new_id = uuid.uuid4().hex[:8]
            return (
                new_id,
                [_make_msg("assistant", "你好！新会话已开始，请问有什么可以帮你的？")],
                gr.MultimodalTextbox(
                    value=None, placeholder="输入您的问题，或拖拽上传图片..."
                ),
            )

        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot, session_id_input],
            outputs=[chatbot, msg_input],
        )

        for btn, q in [
            (quick_q1, "如何重置密码？"),
            (quick_q2, "产品有哪些功能？"),
            (quick_q3, "如何申请退款？"),
            (quick_q4, "支持哪些付款方式？"),
        ]:
            btn.click(
                lambda q=q: quick_ask(q, [], session_id_input.value),
                outputs=[chatbot, msg_input],
            )

        clear_btn.click(
            clear_chat,
            inputs=[session_id_input],
            outputs=[chatbot, msg_input],
        )

        new_btn.click(
            new_session,
            outputs=[session_id_input, chatbot, msg_input],
        )

    return demo


def launch_ui(share: bool = False, server_port: int = 7860):
    demo = create_ui()
    demo.launch(
        share=share,
        server_port=server_port,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.purple,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )
