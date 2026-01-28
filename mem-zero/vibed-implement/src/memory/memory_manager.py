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
from .embeddings import OllamaEmbeddings
from .background_tasks import get_memory_update_queue
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
        llm_chat=None,  # LLM for normal chat (Llama-3.2-1B)
        llm_tool=None,  # LLM for tool/function calling (functiongemma:270m)
        embedding_model: Optional[OllamaEmbeddings] = None,  # Embedding model (jina)
        batch_memory_updates: bool = True  # ⚡ OPTIMIZATION: Batch LLM calls
    ):
        """Initialize Memory Manager.
        
        Args:
            vector_store: Vector store instance (creates default if None)
            storage: Storage instance (creates default if None)
            llm_chat: LLM for normal chat/generation (default: Llama-3.2-1B)
            llm_tool: LLM for tool/function calling (default: functiongemma:270m)
            embedding_model: Embedding model (default: jina/jina-embeddings-v2-small-en)
            batch_memory_updates: If True, batch all facts into one LLM call (faster)
        """
        self.vector_store = vector_store or FileVectorStore()
        self.storage = storage or FileStorage()
        
        # Separate LLMs for different tasks
        self.llm_chat = llm_chat  # For generation/extraction
        self.llm_tool = llm_tool  # For function calling (memory operations)
        
        # Embedding model (Ollama jina)
        if embedding_model is None:
            try:
                self.embedding_model = OllamaEmbeddings(model="jina/jina-embeddings-v2-small-en:latest")
                self.embedding_dim = self.embedding_model.embedding_dim  # Auto-detected
            except Exception as e:
                logger.warning(f"Could not initialize Ollama embeddings: {e}. Using mock embeddings.")
                self.embedding_model = None
                self.embedding_dim = 512  # Default fallback
        else:
            self.embedding_model = embedding_model
            self.embedding_dim = embedding_model.embedding_dim
        
        # Update vector_store with correct embedding dimension
        if hasattr(self.vector_store, 'embedding_dim'):
            if self.vector_store.embedding_dim is None:
                self.vector_store.embedding_dim = self.embedding_dim
            elif self.vector_store.embedding_dim != self.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: vector_store={self.vector_store.embedding_dim}, "
                    f"embedding_model={self.embedding_dim}. Vector store will handle dimension conversion."
                )
        
        self.batch_memory_updates = batch_memory_updates
        
        # Background queue for memory updates (Cold Path)
        self.update_queue = get_memory_update_queue()
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using Ollama jina model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if self.embedding_model:
            # Use real Ollama embeddings
            embedding = self.embedding_model.embed_query(text)
            vector = np.array(embedding)
            # Normalize
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)
            return vector
        else:
            # Fallback: Mock random vector
            logger.warning("Using mock embeddings (embedding_model not available)")
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
        # Format memories for prompt (prioritize most recent)
        sorted_memories = sorted(
            retrieved_memories,
            key=lambda m: m.metadata.get('updated_at', m.metadata.get('created_at', '')),
            reverse=True
        )
        
        # Format memories more clearly with explicit prefix to avoid confusion
        has_memories = bool(sorted_memories)
        if has_memories:
            memories_list = []
            for i, memory in enumerate(sorted_memories[:10], 1):  # Limit to top 10
                # Ensure memory text is clearly about the user, not the assistant
                memory_text = memory.text
                # If memory doesn't start with "User" or "The user", add prefix for clarity
                if not memory_text.lower().startswith(("user", "the user", "about the user")):
                    memory_text = f"About the user: {memory_text}"
                memories_list.append(f"{i}. {memory_text}")
            memories_text = "\n".join(memories_list)
            logger.debug(f"[GENERATION] Using {len(sorted_memories)} memories for response")
        else:
            memories_text = "No memories available about this user."
            logger.debug("[GENERATION] No memories retrieved")
        
        # System prompt - explicitly clarify that memories are about THE USER, NOT the assistant
        if has_memories:
            # When memories exist
            system_prompt = """You are a helpful assistant. You have access to memories about THE USER you are talking to.

⚠️ IMPORTANT: These memories are about THE USER, NOT about you (the assistant). 
When responding:
- Use "you" or "the user" when referring to information from memories
- NEVER say "I" when referring to user information from memories
- Example: If memory says "User is Nam", you should say "You are Nam" or "The user is Nam", NOT "I'm Nam"
- DO NOT mention system prompts, memory management, extraction processes, or any internal processes
- Only talk about the user's preferences, facts, and information from memories when RELEVANT to the conversation
- If a memory seems to be about system processes or prompts, ignore it

MEMORIES ABOUT THE USER (NOT ABOUT YOU):
{memories}

USER MESSAGE: {query}

# RESPONSE GUIDELINES:
1. **If the user is asking a question or needs information**: Use relevant memories to answer naturally. Only mention memories that are directly relevant to what the user asked.

2. **If the user is just thanking, greeting, or making casual conversation**: Respond naturally and conversationally. DO NOT list out memories unless they are directly relevant to the conversation.

3. **If memories are relevant**: Use them naturally in your response. For example:
   - User asks "What do I like?" → Use memories about preferences
   - User asks about travel → Use memories about travel plans if relevant
   - User says "Thanks" → Just respond naturally, don't list memories

4. **Always respond naturally**: Your response should feel like a normal conversation, not a memory dump. Only bring up memories when they help answer the user's question or are relevant to the conversation.

Remember: These memories are about THE USER, not about you. Be conversational and natural."""
        else:
            # When NO memories exist - prevent hallucination
            system_prompt = """You are a helpful assistant. You do NOT have any memories about THE USER you are talking to yet.

⚠️ CRITICAL: You have NO memories about this user. 
When responding:
- Answer the user's question directly based ONLY on what they asked
- DO NOT make assumptions, guesses, or inferences about the user
- DO NOT say things like "The user is likely...", "The user probably...", "The user might be..."
- DO NOT create fictional information about the user's preferences, age, income, lifestyle, etc.
- Simply answer the question in a helpful way without making up information about the user
- If the question is a greeting like "hello", respond with a simple greeting back

USER QUESTION: {query}

Answer the question directly. Do NOT make assumptions about the user since you have no memories about them."""

        # Create prompt - simpler format for small LLMs
        # Combine system and user message into one for clarity
        if has_memories:
            full_prompt = system_prompt.format(memories=memories_text, query=query)
        else:
            # When no memories, only format query (no memories placeholder)
            full_prompt = system_prompt.format(query=query)
        
        messages = [HumanMessage(content=full_prompt)]
        
        # Use chat LLM for generation
        llm = self.llm_chat
        if not llm:
            raise ValueError("llm_chat must be provided for generation")
        
        response = llm.invoke(messages)
        return response.content
    
    def _filter_prompt_text(self, facts: List[str]) -> List[str]:
        """Filter out facts that contain system prompt text or internal process descriptions.
        
        This prevents extraction of prompt text as user facts.
        
        Args:
            facts: List of extracted facts
            
        Returns:
            Filtered list of facts
        """
        # Keywords that indicate prompt/system text
        prompt_keywords = [
            "memory manager",
            "extract salient facts",
            "extract facts",
            "knowledge base",
            "system prompt",
            "you are a",
            "your goal is to",
            "task:",
            "instructions:",
            "output:",
            "return only",
            "json array",
            "do not extract"
        ]
        
        filtered_facts = []
        for fact in facts:
            if not fact or not fact.strip():
                continue
                
            fact_lower = fact.lower().strip()
            # Check if fact contains prompt keywords
            is_prompt_text = any(keyword in fact_lower for keyword in prompt_keywords)
            
            if is_prompt_text:
                logger.debug(f"[FILTER] Filtered out prompt text: {fact}")
                continue
            
            # Ensure fact is about user (should start with "User" or be clearly about user)
            if fact_lower.startswith(("user", "the user", "about the user")):
                filtered_facts.append(fact)
            elif not any(keyword in fact_lower for keyword in ["assistant", "system", "prompt", "instruction", "llm", "ai model"]):
                # If it doesn't contain system keywords, it might be valid
                # But add "User" prefix if missing
                if not fact_lower.startswith("user"):
                    # Check if it's a valid fact about user (contains personal info keywords)
                    personal_keywords = ["name", "age", "years old", "lives", "planning", "likes", "hates", "prefers", "is", "has"]
                    if any(kw in fact_lower for kw in personal_keywords):
                        filtered_facts.append(f"User {fact}")
                    else:
                        logger.debug(f"[FILTER] Skipping fact that doesn't look like user info: {fact}")
                else:
                    filtered_facts.append(fact)
            else:
                logger.debug(f"[FILTER] Filtered out system-related fact: {fact}")
        
        if len(filtered_facts) < len(facts):
            logger.warning(f"[FILTER] Filtered out {len(facts) - len(filtered_facts)} facts containing prompt text")
        
        return filtered_facts
    
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
        # Quick check: if user_message is just a greeting/acknowledgment, skip extraction
        user_msg_lower = user_message.lower().strip()
        trivial_patterns = [
            'hi', 'hello', 'hey', 'thanks', 'thank you', 'good', 'ok', 'okay', 
            'sure', 'yes', 'no', 'bye', 'goodbye', 'i will consider', 'i will think',
            'sounds good', 'that sounds good', 'i see', 'got it', 'understood'
        ]
        # Check if message is just trivial acknowledgment (short and matches patterns)
        if len(user_message.strip()) < 50 and any(pattern in user_msg_lower for pattern in trivial_patterns):
            logger.info(f"[EXTRACTION] Skipping extraction - user message is trivial: '{user_message}'")
            return []
        
        # Format recent messages (for context only, not for extraction)
        recent_messages_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in recent_messages[-5:]  # Use last 5 for context
        ])
        
        # Extraction prompt - simplified for small LLMs
        # CRITICAL: Only extract facts about THE USER from CURRENT interaction, NOT from recent messages
        extraction_prompt = """Extract facts about the user from the CURRENT interaction only.

CURRENT INTERACTION:
User: {user_message}
Assistant: {assistant_message}

RULES:
1. Extract ONLY from CURRENT interaction above
2. Ignore greetings like "hi", "hello", "thanks", "good" - return []
3. Extract user facts like: name, age, preferences, plans, location
4. Always start with "User" (e.g., "User is vegetarian", "User's name is Nam")
5. If no new facts, return []

EXAMPLES:
User: "hi" → []
User: "I'm Nam, 24 years old" → ["User's name is Nam", "User is 24 years old"]
User: "I'm vegetarian" → ["User is vegetarian"]
User: "I'm planning to New York" → ["User is planning to visit New York"]
User: "I hate green onion" → ["User hates green onion"]

OUTPUT: Return ONLY a JSON array. No other text.

Examples of valid output:
["User is vegetarian"]
["User's name is Nam", "User is 24 years old"]
[]

Now extract from CURRENT interaction above:"""

        prompt = ChatPromptTemplate.from_template(extraction_prompt)
        
        # Only format with user_message and assistant_message - don't include recent_messages
        # to avoid confusing LLM into extracting old facts
        messages = prompt.format_messages(
            user_message=user_message,
            assistant_message=assistant_message
        )
        
        # Use chat LLM for extraction
        llm = self.llm_chat
        if not llm:
            raise ValueError("llm_chat must be provided for extraction")
        
        response = llm.invoke(messages)
        response_content = response.content.strip()
        
        # Log raw response for debugging (use INFO so it's visible)
        logger.info(f"[EXTRACTION] Raw LLM response: {response_content[:500]}")
        
        # Parse JSON response
        import json
        try:
            facts = json.loads(response_content)
            if isinstance(facts, list):
                logger.info(f"[EXTRACTION] Parsed {len(facts)} facts before filtering: {facts}")
                # Filter out facts that contain prompt/system text
                facts = self._filter_prompt_text(facts)
                # Remove duplicates (case-insensitive)
                seen = set()
                unique_facts = []
                for fact in facts:
                    fact_lower = fact.lower().strip()
                    if fact_lower not in seen:
                        seen.add(fact_lower)
                        unique_facts.append(fact)
                if len(unique_facts) < len(facts):
                    logger.info(f"[EXTRACTION] Removed {len(facts) - len(unique_facts)} duplicate facts")
                facts = unique_facts
                logger.info(f"[EXTRACTION] After filtering and deduplication: {len(facts)} facts: {facts}")
                if len(facts) == 0:
                    logger.warning(f"[EXTRACTION] All facts were filtered out! Original: {json.loads(response_content)}")
                return facts
            else:
                logger.warning(f"[EXTRACTION] Response is not a list: {type(facts)}, value: {facts}")
                return []
        except json.JSONDecodeError as e:
            logger.warning(f"[EXTRACTION] JSON decode error: {e}")
            logger.warning(f"[EXTRACTION] Full response content: {response_content}")
            
            # Fallback 1: Try to extract JSON array if response starts with [
            if response_content.startswith("["):
                try:
                    facts = json.loads(response_content)
                    logger.info(f"[EXTRACTION] Successfully parsed after fallback: {facts}")
                    return facts
                except Exception as e2:
                    logger.warning(f"[EXTRACTION] Fallback parsing also failed: {e2}")
            
            # Fallback 2: Try to extract JSON array from markdown code blocks
            import re
            json_match = re.search(r'\[.*?\]', response_content, re.DOTALL)
            if json_match:
                try:
                    facts = json.loads(json_match.group())
                    logger.info(f"[EXTRACTION] Extracted JSON from markdown: {facts}")
                    return facts
                except Exception as e3:
                    logger.warning(f"[EXTRACTION] Markdown extraction failed: {e3}")
            
            # Fallback 3: Parse text response - extract lines that look like facts
            # LLM nhỏ thường trả về text thay vì JSON, cần parse manually
            facts = []
            lines = response_content.split('\n')
            for line in lines:
                line = line.strip()
                # Skip empty lines, headers, or lines that don't look like facts
                if not line or line.startswith(('User:', 'Assistant:', '#', 'RULES', 'EXAMPLES', 'OUTPUT', 'Now')):
                    continue
                # Check if line looks like a fact (starts with "User" or contains user info)
                if line.lower().startswith(('user', "user's", "user is", "user has", "user likes", "user hates", "user prefers")):
                    # Clean up the line
                    fact = line.strip()
                    # Remove any leading numbers or bullets
                    fact = re.sub(r'^[\d\-\*\.\s]+', '', fact)
                    if fact and len(fact) > 5:  # Minimum length check
                        facts.append(fact)
            
            if facts:
                logger.info(f"[EXTRACTION] Extracted {len(facts)} facts from text response: {facts}")
                # Filter out prompt text
                facts = self._filter_prompt_text(facts)
                # Remove duplicates (case-insensitive)
                seen = set()
                unique_facts = []
                for fact in facts:
                    fact_lower = fact.lower().strip()
                    if fact_lower not in seen:
                        seen.add(fact_lower)
                        unique_facts.append(fact)
                if len(unique_facts) < len(facts):
                    logger.info(f"[EXTRACTION] Removed {len(facts) - len(unique_facts)} duplicate facts")
                facts = unique_facts
                logger.info(f"[EXTRACTION] After filtering and deduplication: {len(facts)} facts: {facts}")
                return facts
            
            # Fallback 4: Try to find JSON in the response more aggressively
            json_patterns = [
                r'\[["\'].*?["\'].*?\]',  # Simple array pattern
                r'\[[^\]]+\]',  # Any array pattern
            ]
            for pattern in json_patterns:
                matches = re.findall(pattern, response_content, re.DOTALL)
                for match in matches:
                    try:
                        facts = json.loads(match)
                        if isinstance(facts, list):
                            logger.info(f"[EXTRACTION] Extracted JSON using pattern {pattern}: {facts}")
                            return facts
                    except:
                        continue
            
            logger.error(f"[EXTRACTION] Could not parse JSON from response. Full response: {response_content}")
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
        
        # Function calling prompt based on mem0 paper (improved)
        operation_prompt = """You are a memory manager. Decide how to handle a new fact.

New Fact: "{new_fact}"

Existing Similar Memories:
{existing_memories}

# DECISION RULES:
1. If fact is IDENTICAL to existing memory (same meaning) → NOOP
   Example: "User is vegetarian" vs "User is vegetarian" → NOOP
   
2. If fact CONTRADICTS existing memory → DELETE old + ADD new
   Example: "User likes meat" vs existing "User is vegetarian" → DELETE old + ADD new
   
3. If fact ADDS DETAIL to existing memory → UPDATE
   Example: "User is vegetarian and prefers organic" vs existing "User is vegetarian" → UPDATE
   
4. If fact is COMPLETELY NEW → ADD
   Example: "User lives in Hanoi" (no similar memory) → ADD

# OUTPUT (JSON only, no other text):
{{"operation": "ADD|UPDATE|DELETE|NOOP", "target_memory_id": "id_or_null", "new_content": "text_or_null"}}"""

        prompt = ChatPromptTemplate.from_template(operation_prompt)
        
        messages = prompt.format_messages(
            new_fact=new_fact,
            existing_memories=existing_memories_text or "No existing memories found."
        )
        
        # Use tool LLM for function calling
        llm = self.llm_tool or self.llm_chat  # Fallback to chat LLM if tool LLM not available
        if not llm:
            raise ValueError("llm_tool or llm_chat must be provided for memory operations")
        
        response = llm.invoke(messages)
        
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
        
        batch_prompt = """You are a memory manager. Process multiple facts and decide operations.

# FACTS TO PROCESS:
{facts_list}

# TASK:
For EACH fact, compare with its existing memories and decide: ADD, UPDATE, DELETE, or NOOP.

# RULES:
1. If fact is IDENTICAL to existing memory (same meaning) → NOOP
2. If fact CONTRADICTS existing memory → DELETE old + ADD new (or UPDATE)
3. If fact ADDS DETAIL to existing memory → UPDATE existing memory
4. If fact is COMPLETELY NEW → ADD

# OUTPUT FORMAT (JSON array only, no other text):
[
  {{"fact_index": 0, "operation": "ADD", "target_memory_id": null, "new_content": "fact text"}},
  {{"fact_index": 1, "operation": "NOOP", "target_memory_id": null, "new_content": null}},
  {{"fact_index": 2, "operation": "UPDATE", "target_memory_id": "mem-123", "new_content": "updated fact"}}
]

IMPORTANT: Return ONLY valid JSON array, no explanations, no markdown, no code blocks."""

        prompt = ChatPromptTemplate.from_template(batch_prompt)
        
        messages = prompt.format_messages(
            facts_list="\n\n".join(facts_text)
        )
        
        # Use tool LLM for function calling
        llm = self.llm_tool or self.llm_chat  # Fallback to chat LLM if tool LLM not available
        if not llm:
            raise ValueError("llm_tool or llm_chat must be provided for memory operations")
        
        response = llm.invoke(messages)
        
        # Parse JSON response with better error handling
        import json
        import re
        
        response_text = response.content.strip()
        
        # Try to extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        # Try to find JSON array in response
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        try:
            data = json.loads(response_text)
            if isinstance(data, list) and len(data) > 0:
                # Sort by fact_index and return
                sorted_data = sorted(data, key=lambda x: x.get("fact_index", 0))
                operations = []
                for item in sorted_data:
                    try:
                        operations.append(MemoryOperation(**item))
                    except Exception as e:
                        logger.warning(f"Could not parse operation item: {item}, error: {e}")
                        # Fallback: default to ADD
                        operations.append(MemoryOperation(operation="ADD", new_content=item.get("new_content", "")))
                return operations
            else:
                # Fallback: process individually
                logger.warning("Batch response not a valid list, falling back to individual processing")
                return []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Could not parse batch operation decision: {e}")
            logger.debug(f"Response content: {response.content[:500]}")
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
            # If batch failed, fallback to individual processing
            if not operations or len(operations) != len(facts):
                logger.warning(f"Batch operations failed or incomplete ({len(operations)}/{len(facts)}), falling back to individual")
                # Fallback to individual mode for remaining facts
                for i, fact in enumerate(facts):
                    if i < len(operations) and operations[i]:
                        operation = operations[i]
                    else:
                        # Process individually
                        existing_memories = all_existing_memories.get(fact, [])
                        operation = self.decide_memory_operation(
                            new_fact=fact,
                            existing_memories=existing_memories,
                            user_id=user_id,
                            agent_id=agent_id
                        )
                    
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
                # Batch succeeded, execute all operations
                for i, fact in enumerate(facts):
                    operation = operations[i]
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
                vector=fact_vector.tolist(),  # ✅ Save vector to MemoryItem
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
                    vector=fact_vector.tolist(),  # ✅ Save updated vector
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
        
        # Use chat LLM for summarization
        llm = self.llm_chat
        if not llm:
            raise ValueError("llm_chat must be provided for summarization")
        
        response = llm.invoke(messages)
        return response.content.strip()
    
    # ==================== COLD PATH: Background Memory Update ====================
    
    def schedule_memory_update(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        agent_id: Optional[str] = None
    ):
        """Schedule memory update to run in background (Cold Path).
        
        This implements the mem0 architecture: memory updates should NOT block
        the Hot Path (response generation). Updates run asynchronously.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_message: Last user message
            assistant_message: Last assistant message
            agent_id: Optional agent identifier
        """
        def _update_task():
            """Background task to update memory."""
            try:
                # Get conversation context
                context = self.storage.get_context(session_id)
                if not context:
                    logger.warning(f"[COLD_PATH] No context found for session {session_id}")
                    return
                
                # Extract facts (LLM Call 1)
                facts = self.extract_facts(
                    user_message=user_message,
                    assistant_message=assistant_message,
                    global_summary=context.global_summary,
                    recent_messages=context.recent_messages
                )
                
                if not facts:
                    logger.info("[COLD_PATH] No facts extracted, skipping update")
                    return
                
                logger.info(f"[COLD_PATH] Extracted {len(facts)} facts: {facts}")
                
                # Update memories (Update Loop)
                update_results = self.update_memories(
                    facts=facts,
                    user_id=user_id,
                    agent_id=agent_id,
                    scope="shared"
                )
                
                logger.info(f"[COLD_PATH] Memory update completed: {update_results}")
                
            except Exception as e:
                logger.error(f"[COLD_PATH] Memory update failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Enqueue task to background queue
        self.update_queue.enqueue(_update_task)
        logger.info(f"[COLD_PATH] Memory update scheduled for background processing")
