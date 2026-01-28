"""Interactive demo for testing mem0 memory architecture with LangGraph multi-agent system.

What you get:
- Colored logs
- Timing logs for each pipeline step (search, generation, extraction, update, etc.)
- Interactive loop: type messages continuously; type `q` / `quit` to exit

Usage:
    python demo.py
"""

import uuid

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.agents.graph import create_multi_agent_graph
from src.memory.memory_manager import MemoryManager
from src.memory.storage import FileStorage
from src.memory.vector_store import FileVectorStore
from src.utils.observability import logger


def _build_llms():
    """
    Hard-coded Ollama models (no .env needed).

    - Normal chat/extraction: hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest
    - Tool/function calling: functiongemma:270m
    - Embeddings: jina/jina-embeddings-v2-small-en:latest (handled by MemoryManager)

    Returns:
        Tuple of (llm_chat, llm_tool)
    """
    llm_chat = ChatOllama(
        model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
        temperature=0,
    )
    llm_tool = ChatOllama(
        model="functiongemma:270m",
        temperature=0,
    )
    return llm_chat, llm_tool


def main():
    """Run interactive demo."""
    logger.info("🚀 Initializing components...")
    
    # Create storage instances (file-based for demo)
    # Note: embedding_dim will be auto-detected from jina model
    vector_store = FileVectorStore(storage_path="./data/demo/vector_store")
    storage = FileStorage(storage_path="./data/demo/storage")
    
    # 🧹 OPTION: Clear all memory to start fresh (uncomment to reset)
    # vector_store.clear_all()
    # storage.clear_all()
    
    # Create LLM instances (hard-coded Ollama models)
    llm_chat, llm_tool = _build_llms()
    logger.info("✅ LLMs initialized:")
    logger.info(f"   - Chat: hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest")
    logger.info(f"   - Tool: functiongemma:270m")
    logger.info(f"   - Embeddings: jina/jina-embeddings-v2-small-en:latest")
    
    # Create memory manager with proper models
    # ⚡ OPTIMIZATION: batch_memory_updates=True reduces N LLM calls to 1 call
    memory_manager = MemoryManager(
        vector_store=vector_store,
        storage=storage,
        llm_chat=llm_chat,  # For generation/extraction
        llm_tool=llm_tool,  # For function calling (memory operations)
        batch_memory_updates=True  # ⚡ Enable batch mode (1 LLM call instead of N)
    )
    
    # Create multi-agent graph (uses llm_chat for supervisor and agents)
    logger.info("📊 Creating multi-agent graph...")
    graph = create_multi_agent_graph(memory_manager, llm_chat)
    
    # User + session for long-term testing across turns
    user_id = "demo_user_123"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"👤 User ID: {user_id}")
    logger.info(f"📝 Session ID: {session_id}")
    
    # Show memory status
    memory_count = len(vector_store.memories)
    context_count = len(storage.contexts)
    if memory_count > 0 or context_count > 0:
        logger.info(f"📊 Existing data: {memory_count} memories, {context_count} contexts")
        logger.info("   💡 To reset: uncomment clear_all() calls in demo.py (line ~55)")
    else:
        logger.info("📊 Starting with empty memory (fresh start)")
    
    logger.info("Type your message. Type 'q' or 'quit' to exit.")

    turn = 0
    while True:
        user_text = input("\nYou> ").strip()
        if user_text.lower() in {"q", "quit"}:
            logger.info("Exiting demo.")
            break
        if not user_text:
            continue

        turn += 1
        logger.info(f"[TURN] {turn}")

        initial_state = {
            "messages": [HumanMessage(content=user_text)],
            "next_agent": "general",
            "user_id": user_id,
            "session_id": session_id,
            "agent_id": "general",
            "memory_context": {},
        }

        try:
            result = graph.invoke(initial_state)
            agent_id = result.get("agent_id", "unknown")
            if result.get("messages"):
                response = result["messages"][-1].content
                print(f"\nAssistant({agent_id})> {response}")
            else:
                print(f"\nAssistant({agent_id})> (no response)")

            # Quick memory visibility hint
            mem_ctx = result.get("memory_context", {}) or {}
            retrieved = mem_ctx.get("retrieved_memories", []) or []
            if retrieved:
                logger.info(f"[MEMORY_RETRIEVED] {len(retrieved)} ids={retrieved}")

            # Show rolling window size to help you test turn 1..10 behavior
            context = storage.get_context(session_id)
            if context:
                logger.info(f"[CONTEXT] recent_messages={len(context.recent_messages)} (rolling window max=10)")

        except Exception as e:
            logger.error(f"Run failed: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup: stop background queue
    try:
        from src.memory.background_tasks import get_memory_update_queue
        queue = get_memory_update_queue()
        queue.stop()
        logger.info("✅ Background queue stopped")
    except Exception as e:
        logger.warning(f"Could not stop background queue: {e}")
    
    logger.info("📝 Data stored in: ./data/demo/")
    logger.info("   - vector_store/: Vector database files")
    logger.info("   - storage/: Conversation context files")


if __name__ == "__main__":
    main()
