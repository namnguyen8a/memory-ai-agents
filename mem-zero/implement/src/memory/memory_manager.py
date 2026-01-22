"""Memory Manager implementing mem0 architecture pipeline.

This module implements the core mem0 memory pipeline:
1. Retrieval & Generation (Hot Path - Fast response)
2. Extraction & Update (Cold Path - Background memory update)
3. Async Summarization (Background job)

⚠️ PRODUCTION NOTES throughout the code indicate what needs real implementation.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from .models import MemoryItem, ConversationContext, MemoryOperation
from .vector_store import FileVectorStore
from .storage import FileStorage
from ..utils.observability import measure_time, logger


class MemoryManager:
    """Memory Manager implementing mem0 architecture.
    
    This class implements the mem0 memory workflow:
    1. Retrieval: Fast vector search for relevant memories
    2. Generation: Use memories to generate responses
    3. Extraction: Extract facts from conversation
    4. Update: Update memory database using function calling
    5. Summarization: Periodically update global summary
    
    ⚠️ PRODUCTION: Replace mock components with real implementations.
    
    Attributes:
        vector_store: Vector store for memory items
        storage: Storage for conversation context
        llm: Language model for generation/extraction/update
        embedding_model: Embedding model for vector generation (mock in demo)
    """
    
    def __init__(
        self,
        vector_store: Optional[FileVectorStore] = None,
        storage: Optional[FileStorage] = None,
        llm=None,
        embedding_dim: int = 384,  # Mock embedding dimension
        batch_memory_updates: bool = True  # ⚡ OPTIMIZATION: Batch LLM calls
    ):
        """Initialize Memory Manager.
        
        Args:
            vector_store: Vector store instance (creates default if None)
            storage: Storage instance (creates default if None)
            llm: Language model instance (creates default OpenAI if None)
            embedding_dim: Embedding dimension for mock embeddings
            batch_memory_updates: If True, batch all facts into one LLM call (faster)
        """
        self.vector_store = vector_store or FileVectorStore()
        self.storage = storage or FileStorage()
        # LLM is injected from demo (Ollama/OpenAI/etc). Keep it generic.
        self.llm = llm
        self.embedding_dim = embedding_dim
        self.batch_memory_updates = batch_memory_updates
        
        # ⚠️ PRODUCTION: Replace with real embedding model
        # Example: OpenAIEmbeddings(), SentenceTransformerEmbeddings(), etc.
        self._embedding_model = None
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (MOCK for demo).
        
        ⚠️ PRODUCTION: Replace with real embedding model:
        - OpenAI: text-embedding-3-small (1536 dim) or text-embedding-3-large (3072 dim)
        - Sentence Transformers: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
        - Google: textembedding-gecko@003
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Mock: Generate random vector
        # ⚠️ PRODUCTION: Use real embedding model
        # from langchain_openai import OpenAIEmbeddings
        # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # return np.array(embeddings.embed_query(text))
        
        np.random.seed(hash(text) % (2**32))
        vector = np.random.normal(0, 0.1, size=self.embedding_dim)
        return vector / np.linalg.norm(vector)
    
    # ==================== STAGE 1: RETRIEVAL & GENERATION (Hot Path) ====================
    
    @measure_time
    def retrieve_memories(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        agent_id: Optional[str] = None,
        scope: Optional[str] = None
    ) -> List[Tuple[MemoryItem, float]]:
        """Retrieve relevant memories for a query (Hot Path - Fast).
        
        Target latency: < 0.15s for search, < 0.70s total response time.
        
        Args:
            query: User query
            user_id: User identifier
            top_k: Number of memories to retrieve
            agent_id: Optional agent filter
            scope: Optional scope filter ("shared" or "private")
            
        Returns:
            List of (MemoryItem, similarity_score) tuples
        """
        # Generate query embedding
        query_vector = self._generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            user_id=user_id,
            agent_id=agent_id,
            scope=scope
        )
        
        return results
    
    @measure_time
    def generate_response_with_memories(
        self,
        query: str,
        retrieved_memories: List[MemoryItem],
        session_id: str,
        user_id: str
    ) -> str:
        """Generate response using retrieved memories.
        
        This implements the Generation prompt from mem0 paper (Appendix A).
        
        Args:
            query: User query
            retrieved_memories: List of relevant memories
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Generated response text
        """
        # Format memories for prompt
        memories_text = "\n".join([
            f"- {memory.text} (Created: {memory.metadata.get('created_at', 'unknown')})"
            for memory in retrieved_memories
        ])
        
        # System prompt based on mem0 paper
        system_prompt = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information.

# INSTRUCTIONS:
1. Carefully analyze all provided memories.
2. Pay special attention to the timestamps to determine the answer.
3. If the memories contain contradictory information, prioritize the most recent memory.
4. If there is a question about time references (like "last year"), calculate the actual date based on the memory timestamp.
5. Focus only on the content of the memories.
6. Use the memories to provide accurate, contextual answers.

# MEMORIES PROVIDED:
{memories}

# USER QUESTION:
{query}

Answer based on the memories provided. If the memories don't contain relevant information, say so."""

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        # Generate response
        messages = prompt.format_messages(
            memories=memories_text,
            query=query
        )
        
        response = self.llm.invoke(messages)
        return response.content
    
    # ==================== STAGE 2: EXTRACTION & UPDATE (Cold Path) ====================
    
    @measure_time
    def extract_facts(
        self,
        user_message: str,
        assistant_message: str,
        global_summary: str,
        recent_messages: List[Dict[str, str]]
    ) -> List[str]:
        """Extract facts from conversation (LLM Call 1).
        
        This implements the Extraction prompt from mem0 paper.
        
        Args:
            user_message: Last user message
            assistant_message: Last assistant message
            global_summary: Global conversation summary
            recent_messages: Last 10 messages for context
            
        Returns:
            List of extracted facts (strings)
        """
        # Format recent messages
        recent_messages_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in recent_messages[-5:]  # Use last 5 for context
        ])
        
        # Extraction prompt based on mem0 paper
        extraction_prompt = """You are a memory manager AI. Your goal is to extract salient facts from the current conversation turn to update the Knowledge Base.

# INPUT CONTEXT:
Global Summary: {global_summary}

Recent Messages:
{recent_messages}

# CURRENT INTERACTION:
User: {user_message}
Assistant: {assistant_message}

# TASK:
Extract distinct, concise facts (memories) from the "Current Interaction".
- Ignore trivial chit-chat.
- Focus on user preferences, specific events, established facts.
- Output a JSON list of strings. Example: ["User is vegetarian", "User lives in Hanoi"]

Return ONLY a valid JSON array of strings, no other text."""

        prompt = ChatPromptTemplate.from_template(extraction_prompt)
        
        messages = prompt.format_messages(
            global_summary=global_summary or "No summary available yet.",
            recent_messages=recent_messages_text,
            user_message=user_message,
            assistant_message=assistant_message
        )
        
        response = self.llm.invoke(messages)
        
        # Parse JSON response
        import json
        try:
            facts = json.loads(response.content.strip())
            if isinstance(facts, list):
                return facts
            else:
                return []
        except json.JSONDecodeError:
            # Fallback: try to extract list from text
            # This is a workaround for when LLM doesn't return pure JSON
            content = response.content.strip()
            if content.startswith("["):
                try:
                    return json.loads(content)
                except:
                    pass
            return []
    
    @measure_time
    def decide_memory_operation(
        self,
        new_fact: str,
        existing_memories: List[Tuple[MemoryItem, float]],
        user_id: str,
        agent_id: Optional[str] = None
    ) -> MemoryOperation:
        """Decide memory operation using function calling (LLM Call 2).
        
        This implements the Update/Function Calling logic from mem0 paper (Algorithm 1).
        It decides whether to ADD, UPDATE, DELETE, or NOOP.
        
        Args:
            new_fact: New fact to process
            existing_memories: List of similar existing memories with similarity scores
            user_id: User identifier
            agent_id: Optional agent identifier
            
        Returns:
            MemoryOperation decision
        """
        # Format existing memories for prompt
        existing_memories_text = "\n".join([
            f"ID: {mem.id}, Content: {mem.text}, Similarity: {score:.2f}"
            for mem, score in existing_memories
        ])
        
        # Function calling prompt based on mem0 paper
        operation_prompt = """You are a consistent knowledge base maintainer.

New Fact: "{new_fact}"

Existing Similar Memories:
{existing_memories}

Task: Determine the relationship between the New Fact and Existing Memories.

Rules:
- If New Fact contradicts an Existing Memory -> DELETE the Existing Memory ID and ADD the new one (or UPDATE).
- If New Fact adds detail to Existing Memory -> UPDATE Existing Memory ID with the new content.
- If New Fact is completely new -> ADD.
- If New Fact is already covered -> NOOP.

Return a JSON object with:
{{
    "operation": "ADD" | "UPDATE" | "DELETE" | "NOOP",
    "target_memory_id": "memory_id" or null,
    "new_content": "content for ADD or UPDATE" or null
}}"""

        prompt = ChatPromptTemplate.from_template(operation_prompt)
        
        messages = prompt.format_messages(
            new_fact=new_fact,
            existing_memories=existing_memories_text or "No existing memories found."
        )
        
        response = self.llm.invoke(messages)
        
        # Parse JSON response
        import json
        try:
            data = json.loads(response.content.strip())
            return MemoryOperation(**data)
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: default to ADD
            logger.warning(f"Could not parse operation decision: {e}")
            return MemoryOperation(operation="ADD", new_content=new_fact)
    
    @measure_time
    def decide_memory_operations_batch(
        self,
        facts: List[str],
        all_existing_memories: Dict[str, List[Tuple[MemoryItem, float]]],
        user_id: str,
        agent_id: Optional[str] = None
    ) -> List[MemoryOperation]:
        """⚡ OPTIMIZED: Batch version - decide operations for all facts in one LLM call.
        
        This reduces N LLM calls to 1 LLM call, dramatically improving performance.
        
        Args:
            facts: List of facts to process
            all_existing_memories: Dict mapping fact -> list of similar memories
            user_id: User identifier
            agent_id: Optional agent identifier
            
        Returns:
            List of MemoryOperation decisions (one per fact)
        """
        if not facts:
            return []
        
        # Format all facts and their existing memories
        facts_text = []
        for i, fact in enumerate(facts):
            existing_memories = all_existing_memories.get(fact, [])
            existing_memories_text = "\n".join([
                f"  - ID: {mem.id}, Content: {mem.text}, Similarity: {score:.2f}"
                for mem, score in existing_memories
            ]) or "  - No existing memories found."
            
            facts_text.append(f"Fact {i+1}: {fact}\nExisting Memories:\n{existing_memories_text}")
        
        batch_prompt = """You are a consistent knowledge base maintainer. Process multiple facts at once.

# FACTS TO PROCESS:
{facts_list}

# TASK:
For each fact, determine the relationship with its existing memories and decide the operation.

Rules for each fact:
- If New Fact contradicts an Existing Memory -> DELETE the Existing Memory ID and ADD the new one (or UPDATE).
- If New Fact adds detail to Existing Memory -> UPDATE Existing Memory ID with the new content.
- If New Fact is completely new -> ADD.
- If New Fact is already covered -> NOOP.

Return a JSON array with one object per fact:
[
  {{
    "fact_index": 0,
    "operation": "ADD" | "UPDATE" | "DELETE" | "NOOP",
    "target_memory_id": "memory_id" or null,
    "new_content": "content for ADD or UPDATE" or null
  }},
  ...
]

Return ONLY a valid JSON array, no other text."""

        prompt = ChatPromptTemplate.from_template(batch_prompt)
        
        messages = prompt.format_messages(
            facts_list="\n\n".join(facts_text)
        )
        
        response = self.llm.invoke(messages)
        
        # Parse JSON response
        import json
        try:
            data = json.loads(response.content.strip())
            if isinstance(data, list):
                # Sort by fact_index and return
                sorted_data = sorted(data, key=lambda x: x.get("fact_index", 0))
                return [MemoryOperation(**item) for item in sorted_data]
            else:
                # Fallback: process individually
                logger.warning("Batch response not a list, falling back to individual processing")
                return []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Could not parse batch operation decision: {e}, falling back to individual")
            return []
    
    @measure_time
    def update_memories(
        self,
        facts: List[str],
        user_id: str,
        agent_id: Optional[str] = None,
        scope: str = "shared"
    ) -> Dict[str, str]:
        """Update memory database for extracted facts (Update Loop).
        
        This is the core update loop from mem0 paper (Stage 2, Step 3).
        
        ⚡ OPTIMIZATION: If batch_memory_updates=True, uses batch LLM call (1 call instead of N).
        
        Args:
            facts: List of extracted facts
            user_id: User identifier
            agent_id: Optional agent identifier
            scope: Memory scope ("shared" or "private")
            
        Returns:
            Dictionary mapping fact to operation result
        """
        if not facts:
            return {}
        
        results = {}
        
        # ⚡ OPTIMIZATION: Batch mode - one LLM call for all facts
        if self.batch_memory_updates and len(facts) > 1:
            logger.info(f"[OPTIMIZATION] Using batch mode for {len(facts)} facts (1 LLM call instead of {len(facts)})")
            
            # Step 1: Search for similar memories for all facts
            all_existing_memories = {}
            fact_vectors = {}
            for fact in facts:
                fact_vector = self._generate_embedding(fact)
                fact_vectors[fact] = fact_vector
                existing_memories = self.vector_store.search(
                    query_vector=fact_vector,
                    top_k=10,
                    user_id=user_id,
                    agent_id=agent_id,
                    scope=scope
                )
                all_existing_memories[fact] = existing_memories
            
            # Step 2: Batch decide operations (1 LLM call for all facts)
            operations = self.decide_memory_operations_batch(
                facts=facts,
                all_existing_memories=all_existing_memories,
                user_id=user_id,
                agent_id=agent_id
            )
            
            # Step 3: Execute operations
            for i, fact in enumerate(facts):
                if i < len(operations):
                    operation = operations[i]
                else:
                    # Fallback: default to ADD
                    operation = MemoryOperation(operation="ADD", new_content=fact)
                
                fact_vector = fact_vectors[fact]
                results[fact] = self._execute_memory_operation(
                    operation=operation,
                    fact=fact,
                    fact_vector=fact_vector,
                    user_id=user_id,
                    agent_id=agent_id,
                    scope=scope
                )
        
        else:
            # Original mode: one LLM call per fact (slower but more reliable)
            logger.info(f"[STANDARD] Using individual mode for {len(facts)} facts ({len(facts)} LLM calls)")
            
            for fact in facts:
                # Step 3a: Search for similar existing memories
                fact_vector = self._generate_embedding(fact)
                existing_memories = self.vector_store.search(
                    query_vector=fact_vector,
                    top_k=10,
                    user_id=user_id,
                    agent_id=agent_id,
                    scope=scope
                )
                
                # Step 3b: Decide operation using function calling
                operation = self.decide_memory_operation(
                    new_fact=fact,
                    existing_memories=existing_memories,
                    user_id=user_id,
                    agent_id=agent_id
                )
                
                # Step 3c: Execute operation
                results[fact] = self._execute_memory_operation(
                    operation=operation,
                    fact=fact,
                    fact_vector=fact_vector,
                    user_id=user_id,
                    agent_id=agent_id,
                    scope=scope
                )
        
        return results
    
    def _execute_memory_operation(
        self,
        operation: MemoryOperation,
        fact: str,
        fact_vector: np.ndarray,
        user_id: str,
        agent_id: Optional[str],
        scope: str
    ) -> str:
        """Execute a single memory operation.
        
        Args:
            operation: MemoryOperation to execute
            fact: Original fact text
            fact_vector: Embedding vector for the fact
            user_id: User identifier
            agent_id: Optional agent identifier
            scope: Memory scope
            
        Returns:
            Result string describing what was done
        """
        if operation.operation == "ADD":
            memory_id = str(uuid.uuid4())
            new_memory = MemoryItem(
                id=memory_id,
                text=operation.new_content or fact,
                metadata={
                    "user_id": user_id,
                    "agent_id": agent_id or "default",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "scope": scope
                }
            )
            self.vector_store.add(new_memory, vector=fact_vector)
            return f"ADDED: {memory_id}"
            
        elif operation.operation == "UPDATE" and operation.target_memory_id:
            existing_memory = self.vector_store.get(operation.target_memory_id)
            if existing_memory:
                updated_memory = MemoryItem(
                    id=operation.target_memory_id,
                    text=operation.new_content or fact,
                    metadata={
                        **existing_memory.metadata,
                        "updated_at": datetime.utcnow().isoformat()
                    }
                )
                self.vector_store.update(
                    operation.target_memory_id,
                    updated_memory,
                    vector=fact_vector
                )
                return f"UPDATED: {operation.target_memory_id}"
            else:
                return f"UPDATE_FAILED: Memory not found"
                
        elif operation.operation == "DELETE" and operation.target_memory_id:
            self.vector_store.delete(operation.target_memory_id)
            return f"DELETED: {operation.target_memory_id}"
            
        elif operation.operation == "NOOP":
            return "NOOP: Already covered"
        
        # Default fallback
        return f"UNKNOWN_OPERATION: {operation.operation}"
    
    # ==================== STAGE 3: ASYNC SUMMARIZATION ====================
    
    @measure_time
    def summarize_conversation(
        self,
        session_id: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """Generate or update global summary (Background job).
        
        This should run periodically (e.g., after every N turns).
        
        ⚠️ PRODUCTION: Run this as a background job (Celery, RQ, etc.)
        
        Args:
            session_id: Session identifier
            conversation_history: Full conversation history
            
        Returns:
            Updated global summary
        """
        if not conversation_history:
            return ""
        
        # Format conversation
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation_history
        ])
        
        # Get existing summary
        context = self.storage.get_context(session_id)
        existing_summary = context.global_summary if context else ""
        
        # Summarization prompt
        summarization_prompt = """Summarize the following conversation into a concise global summary.

Existing Summary (if any):
{existing_summary}

Conversation:
{conversation}

Create a comprehensive summary that captures:
- User preferences and facts
- Key topics discussed
- Important context

Keep it concise but informative."""

        prompt = ChatPromptTemplate.from_template(summarization_prompt)
        
        messages = prompt.format_messages(
            existing_summary=existing_summary or "No existing summary.",
            conversation=conversation_text
        )
        
        response = self.llm.invoke(messages)
        return response.content.strip()
