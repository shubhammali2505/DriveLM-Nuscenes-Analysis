"""
RAG Configuration Module

This module centralizes all configuration for the RAG system.
Simple, effective configuration management following your project's clean architecture.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, efficient
    max_length: int = 512
    batch_size: int = 32
    device: str = "cpu"  # Simple default, auto-detect GPU if available


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    db_type: str = "faiss"  # Simple, no external dependencies
    index_type: str = "flat"  # Simple but effective
    similarity_metric: str = "cosine"
    top_k: int = 5


@dataclass
class LLMConfig:
    """Configuration for Language Model."""
    model_name: str = "gemini-1.5-flash"   # default Gemini model
    max_length: int = 512
    temperature: float = 0.7
    device: str = "cpu"
    api_key: str = os.getenv("GEMINI_API_KEY", "AIzaSyBJDz7aQPXiQeVAyHO1BItMML1QTppvtb8")  #  load from env or config


@dataclass
class VLMConfig:
    """Configuration for Vision-Language Model."""
    model_name: str = "Salesforce/blip-image-captioning-base"  # Simple, effective
    max_length: int = 256
    device: str = "cpu"


@dataclass
class RAGConfig:
    """Main RAG system configuration."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    
    data_dir: str = "results"
    unified_data_path: str = "results/unified_rag_enhanced.json"
    vector_db_path: str = "results/vector_db"
    
    chunk_size: int = 1000  
    max_context_length: int = 2048
    enable_multimodal: bool = True
    
    # Retrieval settings
    retrieval_strategy: str = "hybrid"  # "semantic", "keyword", or "hybrid"
    rerank_results: bool = True
    
    def __post_init__(self):
        """Auto-detect GPU if available."""
        try:
            import torch
            if torch.cuda.is_available():
                self.embedding.device = "cuda"
                self.llm.device = "cuda"
                self.vlm.device = "cuda"
        except ImportError:
            pass  # Keep CPU defaults

    @property
    def top_k(self) -> int:
        """Expose top_k for compatibility"""
        return self.vector_db.top_k

RAG_CONFIG = RAGConfig()


def update_config(**kwargs) -> None:
    """Update configuration with new values."""
    global RAG_CONFIG
    for key, value in kwargs.items():
        if hasattr(RAG_CONFIG, key):
            setattr(RAG_CONFIG, key, value)


def get_config() -> RAGConfig:
    """Get current configuration."""
    return RAG_CONFIG

config = RAG_CONFIG