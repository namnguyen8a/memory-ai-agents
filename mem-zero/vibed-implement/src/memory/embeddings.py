"""Ollama embedding wrapper for jina embeddings model."""

import numpy as np
from typing import List
from ..utils.observability import measure_time, logger

try:
    from ollama import Client
except ImportError:
    logger.warning("ollama package not found. Install with: pip install ollama")
    Client = None


class OllamaEmbeddings:
    """Ollama embeddings wrapper for jina/jina-embeddings-v2-small-en:latest.
    
    This provides a simple interface to generate embeddings using Ollama's
    jina embedding model.
    
    Attributes:
        model: Model name (default: jina/jina-embeddings-v2-small-en:latest)
        client: Ollama client instance
        embedding_dim: Embedding dimension (768 for jina-v2-small-en)
    """
    
    def __init__(self, model: str = "jina/jina-embeddings-v2-small-en:latest"):
        """Initialize Ollama embeddings.
        
        Args:
            model: Ollama model name for embeddings
        """
        if Client is None:
            raise ImportError("ollama package required. Install with: pip install ollama")
        
        self.model = model
        self.client = Client()
        # Detect actual embedding dimension by testing with a dummy query
        self.embedding_dim = self._detect_dimension()
        
        # Ensure model exists
        self._ensure_model_exists()
    
    def _detect_dimension(self) -> int:
        """Detect actual embedding dimension by testing with a dummy query.
        
        Returns:
            Embedding dimension
        """
        try:
            # Test with a short dummy text
            response = self.client.embeddings(model=self.model, prompt="test")
            embedding = response.get("embedding", [])
            if embedding:
                dim = len(embedding)
                logger.info(f"Detected embedding dimension: {dim} for model {self.model}")
                return dim
        except Exception as e:
            logger.warning(f"Could not detect embedding dimension: {e}")
        
        # Fallback: default based on model name
        if "jina" in self.model.lower():
            return 512  # jina models typically use 512
        return 768  # Default fallback
    
    def _ensure_model_exists(self):
        """Ensure the model exists locally, pull if needed."""
        try:
            local_models = self.client.list()["models"]
            model_names = [m.get("name", "") for m in local_models]
            
            if self.model not in model_names:
                logger.info(f"Pulling embedding model: {self.model}")
                self.client.pull(self.model)
                logger.info(f"✅ Model {self.model} ready")
        except Exception as e:
            logger.warning(f"Could not verify model existence: {e}")
    
    @measure_time
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings(model=self.model, prompt=text)
            embedding = response.get("embedding", [])
            
            if not embedding:
                raise ValueError(f"No embedding returned from model {self.model}")
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Fallback: return zero vector
            return [0.0] * self.embedding_dim
    
    @measure_time
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

