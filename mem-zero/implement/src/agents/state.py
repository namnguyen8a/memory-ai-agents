"""State schema for multi-agent graph."""

from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State schema for multi-agent graph.
    
    This state is shared across all agents in the multi-agent system.
    It includes messages, agent routing information, and memory context.
    
    Attributes:
        messages: List of messages in conversation
        next_agent: Which agent to route to next (determined by supervisor)
        user_id: Current user identifier
        session_id: Current session identifier
        agent_id: Current agent identifier
        memory_context: Retrieved memories for current query
    """
    
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: Literal["sales", "support", "general", "END"]
    user_id: str
    session_id: str
    agent_id: str
    memory_context: dict  # Retrieved memories and context
