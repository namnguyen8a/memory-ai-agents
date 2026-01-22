"""Multi-agent graph implementation with mem0 memory integration.

This implements a multi-agent supervisor system with three specialized agents:
- Sales Agent: Handles product inquiries and recommendations
- Support Agent: Handles technical support and troubleshooting
- General Agent: Handles general questions

All agents use mem0 memory architecture for context-aware responses.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .state import AgentState
from ..memory.memory_manager import MemoryManager
from ..utils.observability import logger, measure_time


def create_multi_agent_graph(
    memory_manager: MemoryManager,
    llm=None
) -> StateGraph:
    """Create multi-agent graph with mem0 memory integration.
    
    Architecture:
        START → Supervisor → [Sales/Support/General Agent] → Memory Update → END
        
    Flow:
        1. User sends message
        2. Supervisor routes to appropriate agent
        3. Agent retrieves relevant memories
        4. Agent generates response using memories
        5. Memory manager extracts facts and updates memory
        6. Response returned to user
    
    Args:
        memory_manager: MemoryManager instance for memory operations
        llm: Optional LLM instance (creates default if None)
        
    Returns:
        Compiled StateGraph
    """
    
    if llm is None:
        raise ValueError("llm must be provided (e.g., ChatOllama).")
    
    # ==================== NODES ====================
    
    @measure_time
    def supervisor_node(state: AgentState) -> AgentState:
        """Supervisor node that routes to appropriate agent.
        
        This node decides which specialized agent should handle the request.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with next_agent set
        """
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        # Supervisor prompt
        supervisor_prompt = f"""You are a supervisor managing multiple specialized agents.

Available agents:
- sales: Product inquiries, recommendations, pricing
- support: Technical support, troubleshooting, help
- general: General questions, casual conversation

User message: "{last_message}"

Determine which agent should handle this. Respond with ONLY one word: "sales", "support", or "general"."""
        
        response = llm.invoke([HumanMessage(content=supervisor_prompt)])
        decision = response.content.strip().lower()
        
        # Validate decision
        if decision in ["sales", "support", "general"]:
            next_agent = decision
        else:
            next_agent = "general"  # Default fallback
        
        logger.info(f"[ROUTE] supervisor -> {next_agent}")
        return {
            "next_agent": next_agent,
            "agent_id": next_agent
        }
    
    @measure_time
    def sales_agent_node(state: AgentState) -> AgentState:
        """Sales agent node - handles product inquiries.
        
        This agent:
        1. Retrieves relevant memories about user preferences
        2. Generates response using memories
        3. Returns response
        
        Args:
            state: Current state
            
        Returns:
            Updated state with agent response
        """
        user_id = state["user_id"]
        session_id = state["session_id"]
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        # Retrieve relevant memories
        memories = memory_manager.retrieve_memories(
            query=last_message,
            user_id=user_id,
            top_k=10,
            agent_id="sales",
            scope="shared"
        )
        
        # Generate response with memories
        response_text = memory_manager.generate_response_with_memories(
            query=last_message,
            retrieved_memories=[mem for mem, _ in memories],
            session_id=session_id,
            user_id=user_id
        )
        
        # Get or create conversation context
        context = memory_manager.storage.get_context(session_id)
        if not context:
            context = memory_manager.storage.create_context(
                session_id=session_id,
                user_id=user_id
            )
        
        # Add messages to context (for memory update later)
        memory_manager.storage.add_message(session_id, "user", last_message)
        memory_manager.storage.add_message(session_id, "assistant", response_text)
        
        # Store memory context for update phase
        memory_context = {
            "retrieved_memories": [mem.id for mem, _ in memories],
            "last_user_message": last_message,
            "last_assistant_message": response_text
        }
        
        return {
            "messages": [AIMessage(content=response_text)],
            "memory_context": memory_context
        }
    
    @measure_time
    def support_agent_node(state: AgentState) -> AgentState:
        """Support agent node - handles technical support.
        
        Similar to sales agent but with support-focused context.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with agent response
        """
        user_id = state["user_id"]
        session_id = state["session_id"]
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        # Retrieve relevant memories
        memories = memory_manager.retrieve_memories(
            query=last_message,
            user_id=user_id,
            top_k=10,
            agent_id="support",
            scope="shared"
        )
        
        # Generate response with memories
        response_text = memory_manager.generate_response_with_memories(
            query=last_message,
            retrieved_memories=[mem for mem, _ in memories],
            session_id=session_id,
            user_id=user_id
        )
        
        # Get or create conversation context
        context = memory_manager.storage.get_context(session_id)
        if not context:
            context = memory_manager.storage.create_context(
                session_id=session_id,
                user_id=user_id
            )
        
        # Add messages to context
        memory_manager.storage.add_message(session_id, "user", last_message)
        memory_manager.storage.add_message(session_id, "assistant", response_text)
        
        # Store memory context
        memory_context = {
            "retrieved_memories": [mem.id for mem, _ in memories],
            "last_user_message": last_message,
            "last_assistant_message": response_text
        }
        
        return {
            "messages": [AIMessage(content=response_text)],
            "memory_context": memory_context
        }
    
    @measure_time
    def general_agent_node(state: AgentState) -> AgentState:
        """General agent node - handles general questions.
        
        Similar to other agents but for general conversation.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with agent response
        """
        user_id = state["user_id"]
        session_id = state["session_id"]
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        # Retrieve relevant memories
        memories = memory_manager.retrieve_memories(
            query=last_message,
            user_id=user_id,
            top_k=10,
            agent_id="general",
            scope="shared"
        )
        
        # Generate response with memories
        response_text = memory_manager.generate_response_with_memories(
            query=last_message,
            retrieved_memories=[mem for mem, _ in memories],
            session_id=session_id,
            user_id=user_id
        )
        
        # Get or create conversation context
        context = memory_manager.storage.get_context(session_id)
        if not context:
            context = memory_manager.storage.create_context(
                session_id=session_id,
                user_id=user_id
            )
        
        # Add messages to context
        memory_manager.storage.add_message(session_id, "user", last_message)
        memory_manager.storage.add_message(session_id, "assistant", response_text)
        
        # Store memory context
        memory_context = {
            "retrieved_memories": [mem.id for mem, _ in memories],
            "last_user_message": last_message,
            "last_assistant_message": response_text
        }
        
        return {
            "messages": [AIMessage(content=response_text)],
            "memory_context": memory_context
        }
    
    @measure_time
    def memory_update_node(state: AgentState) -> AgentState:
        """Memory update node - extracts facts and updates memory (Cold Path).
        
        This node implements Stage 2 of mem0 pipeline:
        1. Extract facts from conversation
        2. Update memory database
        
        ⚠️ NOTE: In production, this should run asynchronously in background
        to avoid blocking the response (Hot Path vs Cold Path).
        
        Args:
            state: Current state
            
        Returns:
            Updated state (no changes, just processes memory update)
        """
        user_id = state["user_id"]
        session_id = state["session_id"]
        agent_id = state.get("agent_id", "general")
        memory_context = state.get("memory_context", {})
        
        # Get conversation context
        context = memory_manager.storage.get_context(session_id)
        if not context:
            # No context yet, skip update
            return {}
        
        # Extract facts from last interaction
        user_message = memory_context.get("last_user_message", "")
        assistant_message = memory_context.get("last_assistant_message", "")
        
        if not user_message or not assistant_message:
            return {}
        
        # Extract facts (LLM Call 1)
        facts = memory_manager.extract_facts(
            user_message=user_message,
            assistant_message=assistant_message,
            global_summary=context.global_summary,
            recent_messages=context.recent_messages
        )
        
        if not facts:
            return {}
        
        # Update memories (Update Loop - LLM Call 2 per fact)
        update_results = memory_manager.update_memories(
            facts=facts,
            user_id=user_id,
            agent_id=agent_id,
            scope="shared"
        )
        
        # Log update results (optional)
        logger.info(f"[MEMORY_UPDATE] {update_results}")
        
        # ⚠️ PRODUCTION: Schedule async summarization job here
        # This should run periodically (e.g., every 5-10 turns)
        # Example: celery_task.delay(session_id, context.recent_messages)
        
        return {}
    
    # ==================== BUILD GRAPH ====================
    
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sales_agent", sales_agent_node)
    graph.add_node("support_agent", support_agent_node)
    graph.add_node("general_agent", general_agent_node)
    graph.add_node("memory_update", memory_update_node)
    
    # Set entry point
    graph.set_entry_point("supervisor")
    
    # Add conditional edges from supervisor
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        {
            "sales": "sales_agent",
            "support": "support_agent",
            "general": "general_agent",
            "END": END
        }
    )
    
    # All agents route to memory_update, then END
    graph.add_edge("sales_agent", "memory_update")
    graph.add_edge("support_agent", "memory_update")
    graph.add_edge("general_agent", "memory_update")
    graph.add_edge("memory_update", END)
    
    # Compile and return
    return graph.compile()
