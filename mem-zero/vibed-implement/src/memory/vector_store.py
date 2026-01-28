"""File-based vector store implementation (Mock for demo).

⚠️ PRODUCTION NOTE: This is a demo implementation using file-based storage.
In production, replace this with a real vector database like:
- Qdrant (recommended for production)
- Pinecone (cloud-based)
- ChromaDB (open-source, easy to deploy)
- Weaviate (self-hosted)
- pgvector (PostgreSQL extension)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .models import MemoryItem
from ..utils.observability import logger, measure_time


class FileVectorStore:
    """File-based vector store using JSON and numpy for demo purposes.
    
    This is a simple implementation that stores vectors in memory and
    persists to disk as JSON. For production, use a real vector database.
    
    ⚠️ PRODUCTION: Replace with QdrantClient, Pinecone, ChromaDB, etc.
    
    Attributes:
        storage_path: Path to storage directory
        memories: In-memory storage of memories (dict by ID)
        vectors: In-memory storage of vectors (list aligned with memories)
    """
    
    def __init__(self, storage_path: str = "./data/vector_store", embedding_dim: Optional[int] = None):
        """Initialize file-based vector store.
        
        Args:
            storage_path: Path to storage directory
            embedding_dim: Expected embedding dimension (auto-detected if None)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memories_file = self.storage_path / "memories.json"
        self.vectors_file = self.storage_path / "vectors.npy"
        
        # In-memory storage
        self.memories: Dict[str, MemoryItem] = {}
        self.vectors: List[Optional[np.ndarray]] = []
        self.id_to_index: Dict[str, int] = {}  # Map memory ID to index in vectors list
        
        # Embedding dimension (will be set after loading or from parameter)
        self.embedding_dim = embedding_dim
        
        # Load existing data if available
        self._load()
        
        # Auto-detect dimension from loaded vectors if not set
        if self.embedding_dim is None:
            if self.vectors and any(v is not None for v in self.vectors):
                # Use dimension from existing vectors
                for v in self.vectors:
                    if v is not None:
                        self.embedding_dim = len(v)
                        logger.info(f"Auto-detected embedding dimension: {self.embedding_dim} from existing vectors")
                        break
            else:
                # Default fallback
                self.embedding_dim = 512  # jina default
                logger.warning(f"No existing vectors found, using default dimension: {self.embedding_dim}")
    
    def _load(self):
        """Load memories and vectors from disk."""
        if self.memories_file.exists():
            try:
                with open(self.memories_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memories = {
                        mid: MemoryItem(**item) 
                        for mid, item in data.items()
                    }
                    self.id_to_index = {mid: idx for idx, mid in enumerate(self.memories.keys())}
            except Exception as e:
                logger.warning(f"Could not load memories: {e}")
                self.memories = {}
                self.id_to_index = {}
        
        # Load vectors if available
        if self.vectors_file.exists():
            try:
                vectors_array = np.load(self.vectors_file)
                loaded_vectors = []
                for i in range(len(self.memories)):
                    if i < len(vectors_array):
                        vec = vectors_array[i]
                        # Handle dimension mismatch: pad or truncate if needed
                        if self.embedding_dim is not None and len(vec) != self.embedding_dim:
                            logger.warning(
                                f"Vector dimension mismatch: stored={len(vec)}, expected={self.embedding_dim}. "
                                f"Will handle during search."
                            )
                        loaded_vectors.append(vec)
                    else:
                        loaded_vectors.append(None)
                self.vectors = loaded_vectors
            except Exception as e:
                logger.warning(f"Could not load vectors: {e}")
                self.vectors = [None] * len(self.memories)
    
    def _save(self):
        """Save memories and vectors to disk."""
        try:
            # Save memories as JSON
            data = {
                mid: item.model_dump() 
                for mid, item in self.memories.items()
            }
            with open(self.memories_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Save vectors as numpy array
            if self.vectors and any(v is not None for v in self.vectors):
                # Use detected dimension or default
                dim = self.embedding_dim or 512
                vectors_array = np.array([
                    v if v is not None else np.zeros(dim)
                    for v in self.vectors
                ])
                np.save(self.vectors_file, vectors_array)
        except Exception as e:
            logger.warning(f"Could not save vector store: {e}")
    
    def clear_all(self):
        """Clear all memories and vectors (reset to empty state).
        
        This removes all in-memory data and deletes the storage files.
        Use this to start fresh for testing.
        """
        logger.info("🧹 Clearing all memories and vectors...")
        self.memories = {}
        self.vectors = []
        self.id_to_index = {}
        
        # Delete files if they exist
        if self.memories_file.exists():
            self.memories_file.unlink()
            logger.info(f"   Deleted: {self.memories_file}")
        
        if self.vectors_file.exists():
            self.vectors_file.unlink()
            logger.info(f"   Deleted: {self.vectors_file}")
        
        logger.info("✅ Vector store cleared")
    
    @measure_time
    def add(self, memory: MemoryItem, vector: Optional[np.ndarray] = None):
        """Add a memory to the vector store.
        
        Args:
            memory: MemoryItem to add
            vector: Optional embedding vector (if None, will use memory.vector)
        """
        if vector is None:
            if memory.vector:
                vector = np.array(memory.vector)
            else:
                # Should not happen - vector should be provided
                raise ValueError("Vector must be provided when adding memory")
        
        # Ensure vector matches expected dimension
        if self.embedding_dim is not None and len(vector) != self.embedding_dim:
            logger.warning(
                f"Vector dimension mismatch: provided={len(vector)}, expected={self.embedding_dim}. "
                f"Truncating or padding to match."
            )
            if len(vector) > self.embedding_dim:
                vector = vector[:self.embedding_dim]
            else:
                # Pad with zeros
                padding = np.zeros(self.embedding_dim - len(vector))
                vector = np.concatenate([vector, padding])
        
        # Update embedding_dim if this is the first vector
        if self.embedding_dim is None:
            self.embedding_dim = len(vector)
            logger.info(f"Set embedding dimension to {self.embedding_dim} from first vector")
        
        # Normalize vector
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        # Store memory
        self.memories[memory.id] = memory
        idx = len(self.vectors)
        self.id_to_index[memory.id] = idx
        self.vectors.append(vector)
        
        # Persist to disk
        self._save()
    
    @measure_time
    def update(self, memory_id: str, memory: MemoryItem, vector: Optional[np.ndarray] = None):
        """Update an existing memory.
        
        Args:
            memory_id: ID of memory to update
            memory: Updated MemoryItem
            vector: Optional new embedding vector
        """
        if memory_id not in self.memories:
            raise ValueError(f"Memory {memory_id} not found")
        
        # Update vector if provided
        if vector is not None:
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)
            idx = self.id_to_index[memory_id]
            self.vectors[idx] = vector
            memory.vector = vector.tolist()
        
        # Update memory
        self.memories[memory_id] = memory
        
        # Persist to disk
        self._save()
    
    @measure_time
    def delete(self, memory_id: str):
        """Delete a memory from the vector store.
        
        Args:
            memory_id: ID of memory to delete
        """
        if memory_id not in self.memories:
            return
        
        # Remove from memories
        del self.memories[memory_id]
        
        # Remove from vectors (mark as None, keep index alignment)
        idx = self.id_to_index[memory_id]
        self.vectors[idx] = None
        del self.id_to_index[memory_id]
        
        # Persist to disk
        self._save()
    
    @measure_time
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        scope: Optional[str] = None,
        filter_metadata: Optional[Dict[str, any]] = None
    ) -> List[Tuple[MemoryItem, float]]:
        """Search for similar memories.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            user_id: Filter by user_id (required for multi-user)
            agent_id: Filter by agent_id (for multi-agent)
            scope: Filter by scope ("shared" or "private")
            filter_metadata: Additional metadata filters
            
        Returns:
            List of (MemoryItem, similarity_score) tuples, sorted by similarity
        """
        if len(self.memories) == 0:
            return []
        
        # Normalize query vector
        if np.linalg.norm(query_vector) > 0:
            query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Filter and compute similarities
        results: List[Tuple[MemoryItem, float]] = []
        
        for memory_id, memory in self.memories.items():
            # Apply filters
            if user_id and memory.metadata.get("user_id") != user_id:
                continue
            if agent_id and memory.metadata.get("agent_id") != agent_id:
                continue
            if scope and memory.metadata.get("scope") != scope:
                continue
            if filter_metadata:
                for key, value in filter_metadata.items():
                    if memory.metadata.get(key) != value:
                        continue
            
            # Get vector
            idx = self.id_to_index.get(memory_id)
            if idx is None or idx >= len(self.vectors) or self.vectors[idx] is None:
                continue
            
            vector = self.vectors[idx]
            
            # Handle dimension mismatch: pad or truncate to match query
            if len(vector) != len(query_vector):
                logger.debug(
                    f"Dimension mismatch: stored={len(vector)}, query={len(query_vector)}. "
                    f"Adjusting stored vector."
                )
                if len(vector) > len(query_vector):
                    # Truncate stored vector
                    vector = vector[:len(query_vector)]
                else:
                    # Pad stored vector with zeros
                    padding = np.zeros(len(query_vector) - len(vector))
                    vector = np.concatenate([vector, padding])
            
            # Compute cosine similarity
            similarity = float(cosine_similarity([query_vector], [vector])[0][0])
            results.append((memory, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return results[:top_k]
    
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            MemoryItem if found, None otherwise
        """
        return self.memories.get(memory_id)
