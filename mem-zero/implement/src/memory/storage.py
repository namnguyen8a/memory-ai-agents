"""File-based storage for conversation context (Mock for demo).

⚠️ PRODUCTION NOTE: This is a demo implementation using file-based storage.
In production, replace this with:
- Redis (for session-level data with TTL)
- PostgreSQL (for persistent conversation data)
- MongoDB (for flexible schema)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import ConversationContext
from ..utils.observability import logger, measure_time


class FileStorage:
    """File-based storage for conversation context using JSON files.
    
    This stores conversation summaries and recent messages. For production,
    use Redis for session-level data or PostgreSQL for persistent storage.
    
    ⚠️ PRODUCTION: Replace with Redis, PostgreSQL, MongoDB, etc.
    
    Attributes:
        storage_path: Path to storage directory
        contexts: In-memory storage of conversation contexts
    """
    
    def __init__(self, storage_path: str = "./data/storage"):
        """Initialize file-based storage.
        
        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.contexts_file = self.storage_path / "contexts.json"
        
        # In-memory storage
        self.contexts: Dict[str, ConversationContext] = {}
        
        # Load existing data
        self._load()
    
    def _load(self):
        """Load contexts from disk."""
        if self.contexts_file.exists():
            try:
                with open(self.contexts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.contexts = {
                        sid: ConversationContext(**item)
                        for sid, item in data.items()
                    }
            except Exception as e:
                logger.warning(f"Could not load contexts: {e}")
                self.contexts = {}
    
    def _save(self):
        """Save contexts to disk."""
        try:
            data = {
                sid: context.model_dump(mode='json')
                for sid, context in self.contexts.items()
            }
            with open(self.contexts_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save contexts: {e}")
    
    @measure_time
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationContext if found, None otherwise
        """
        return self.contexts.get(session_id)
    
    @measure_time
    def create_context(
        self,
        session_id: str,
        user_id: str,
        global_summary: str = "",
        recent_messages: List[Dict[str, str]] = None
    ) -> ConversationContext:
        """Create a new conversation context.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            global_summary: Initial summary
            recent_messages: Initial messages
            
        Returns:
            Created ConversationContext
        """
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            global_summary=global_summary,
            recent_messages=recent_messages or []
        )
        self.contexts[session_id] = context
        self._save()
        return context
    
    @measure_time
    def update_context(
        self,
        session_id: str,
        global_summary: Optional[str] = None,
        recent_messages: Optional[List[Dict[str, str]]] = None
    ) -> ConversationContext:
        """Update conversation context.
        
        Args:
            session_id: Session identifier
            global_summary: Updated summary (optional)
            recent_messages: Updated messages (optional)
            
        Returns:
            Updated ConversationContext
            
        Raises:
            ValueError: If session_id not found
        """
        if session_id not in self.contexts:
            raise ValueError(f"Context {session_id} not found")
        
        context = self.contexts[session_id]
        
        if global_summary is not None:
            context.global_summary = global_summary
        
        if recent_messages is not None:
            # Keep only last 10 messages (rolling window)
            context.recent_messages = recent_messages[-10:]
        
        context.updated_at = datetime.utcnow()
        
        self._save()
        return context
    
    @measure_time
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> ConversationContext:
        """Add a message to conversation context.
        
        Args:
            session_id: Session identifier
            role: Message role ("user" or "assistant")
            content: Message content
            
        Returns:
            Updated ConversationContext
            
        Raises:
            ValueError: If session_id not found
        """
        if session_id not in self.contexts:
            raise ValueError(f"Context {session_id} not found")
        
        context = self.contexts[session_id]
        
        # Add message
        context.recent_messages.append({
            "role": role,
            "content": content
        })
        
        # Keep only last 10 messages (rolling window)
        context.recent_messages = context.recent_messages[-10:]
        context.updated_at = datetime.utcnow()
        
        self._save()
        return context
