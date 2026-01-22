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


def _build_llm():
    """
    Hard-coded Ollama models (no .env needed).

    - Normal responses: hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest
    - Tool/function calling: functiongemma:270m

    The graph currently uses one LLM instance; we default to the normal model.
    If you want to experiment with tool-calling, switch `llm_normal` below.
    """
    llm_normal = ChatOllama(
        model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
        temperature=0,
    )
    # Optional: tool-calling model
    llm_tool = ChatOllama(
        model="functiongemma:270m",
        temperature=0,
    )
    # Default: use normal model; swap to llm_tool if you need tool calls
    return llm_normal


def main():
    """Run interactive demo."""
    logger.info("🚀 Initializing components...")
    
    # Create storage instances (file-based for demo)
    vector_store = FileVectorStore(storage_path="./data/demo/vector_store")
    storage = FileStorage(storage_path="./data/demo/storage")
    
    # Create LLM instance (hard-coded Ollama models)
    llm = _build_llm()
    
    # Create memory manager with batch optimization enabled
    # ⚡ OPTIMIZATION: batch_memory_updates=True reduces N LLM calls to 1 call
    # This dramatically improves performance (300s → ~60s for 5 facts)
    memory_manager = MemoryManager(
        vector_store=vector_store,
        storage=storage,
        llm=llm,
        batch_memory_updates=True  # ⚡ Enable batch mode (1 LLM call instead of N)
    )
    
    # Create multi-agent graph
    logger.info("📊 Creating multi-agent graph...")
    graph = create_multi_agent_graph(memory_manager, llm)
    
    # User + session for long-term testing across turns
    user_id = "demo_user_123"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"👤 User ID: {user_id}")
    logger.info(f"📝 Session ID: {session_id}")
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

    logger.info("📝 Data stored in: ./data/demo/")
    logger.info("   - vector_store/: Vector database files")
    logger.info("   - storage/: Conversation context files")


if __name__ == "__main__":
    main()
