
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import torch
from sentence_transformers import SentenceTransformer

from .config import get_config

logger = logging.getLogger(__name__)


class RAGEmbedder:
    
    def __init__(self, config_override: Dict[str, Any] = None):
        """Initialize embedder with configuration."""
        self.config = get_config()
        if config_override:
            for key, value in config_override.items():
                setattr(self.config.embedding, key, value)
        
        self.model = None
        self.device = self.config.embedding.device
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        try:
            logger.info(f"Loading embedding model: {self.config.embedding.model_name}")
            
            self.model = SentenceTransformer(
                self.config.embedding.model_name,
                device=self.device
            )
            
            self.model.max_seq_length = self.config.embedding.max_length
            
            logger.info(f"Embedding model loaded on {self.device}")
            logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            logger.info("Falling back to CPU device")
            
            self.device = "cpu"
            self.model = SentenceTransformer(
                self.config.embedding.model_name,
                device="cpu"
            )
    
    def embed_texts(self, texts: List[str]) -> Optional[np.ndarray]:
   
        if not texts:
            logger.warning("Empty text list provided")
            return None
        
        try:
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                logger.warning("No valid texts after filtering")
                return None
            
            logger.debug(f"Embedding {len(valid_texts)} texts")
            
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.config.embedding.batch_size,
                show_progress_bar=len(valid_texts) > 100,
                convert_to_numpy=True
            )
            
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None
    
    def embed_single_text(self, text: str) -> Optional[np.ndarray]:

        if not text or not text.strip():
            logger.warning("Empty text provided")
            return None
        
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0]  
            
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            return None
    
    def embed_batch(self, texts: List[str], batch_id: int = 0) -> Tuple[np.ndarray, List[int]]:
  
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            logger.warning(f"Batch {batch_id}: No valid texts")
            return np.array([]), []
        
        try:
            logger.debug(f"Batch {batch_id}: Embedding {len(valid_texts)} texts")
            
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.config.embedding.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False  
            )
            
            return embeddings, valid_indices
            
        except Exception as e:
            logger.error(f"Batch {batch_id}: Error generating embeddings: {e}")
            return np.array([]), []
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 384  
    
    def preprocess_text(self, text: str) -> str:

        if not text:
            return ""
        
        processed = " ".join(text.split()) 
        
        max_chars = self.config.embedding.max_length * 4  
        if len(processed) > max_chars:
            processed = processed[:max_chars] + "..."
            logger.debug(f"Truncated text from {len(text)} to {len(processed)} chars")
        
        return processed
    
    def get_device_info(self) -> Dict[str, Any]:
        info = {
            'model_name': self.config.embedding.model_name,
            'device': self.device,
            'max_length': self.config.embedding.max_length,
            'batch_size': self.config.embedding.batch_size,
            'embedding_dimension': self.get_embedding_dimension()
        }
        
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_device_count'] = torch.cuda.device_count()
        else:
            info['cuda_available'] = False
        
        return info
    
        # === LangChain compatibility ===
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents (for FAISS / LangChain).
        This just forwards to self.embed_texts.
        """
        embeddings = self.embed_texts(texts)
        return embeddings.tolist() if embeddings is not None else []

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query (for FAISS / LangChain).
        This just forwards to self.embed_single_text.
        """
        embedding = self.embed_single_text(text)
        return embedding.tolist() if embedding is not None else []



def create_embedder(config_override: Dict[str, Any] = None) -> RAGEmbedder:
    """
    Factory function to create configured embedder.
    
    Simple interface for your main.py integration.
    """
    return RAGEmbedder(config_override)



def embed_questions_and_contexts(
    questions: List[str], 
    contexts: List[str],
    embedder: Optional[RAGEmbedder] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Embed questions and contexts separately.
    
    Useful for question-answering tasks where you want
    separate embeddings for queries and documents.
    """
    if embedder is None:
        embedder = create_embedder()
    
    question_embeddings = embedder.embed_texts(questions)
    context_embeddings = embedder.embed_texts(contexts)
    
    return question_embeddings, context_embeddings


def calculate_similarity(
    embedding1: np.ndarray, 
    embedding2: np.ndarray,
    metric: str = "cosine"
) -> float:

    try:
        if metric == "cosine":
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        elif metric == "dot":
            return np.dot(embedding1, embedding2)
        
        elif metric == "euclidean":
            return -np.linalg.norm(embedding1 - embedding2)
        
        else:
            logger.warning(f"Unknown similarity metric: {metric}")
            return 0.0
    
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0