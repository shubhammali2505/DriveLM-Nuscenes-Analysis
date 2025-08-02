
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .pipeline import setup_and_run_rag_system, create_rag_pipeline, RAGPipeline
    from .config import config, get_config
    from .dataloader import RAGDataLoader
    from .retriever import RAGRetriever
    from .generator import RAGGenerator
    
    __all__ = [
        'setup_and_run_rag_system',
        'create_rag_pipeline',
        'RAGPipeline',
        'config',
        'get_config'
    ]
    
    __version__ = "1.0.0"
    
except ImportError as e:
    logger.error(f"Failed to import RAG modules: {e}")
    
    def setup_and_run_rag_system(output_dir="results", force_rebuild=False, load_vision=True):
        """Fallback implementation when proper modules can't be imported"""
        logger.error("Using fallback RAG implementation due to import errors")
        
        class FallbackPipeline:
            def __init__(self):
                self.output_dir = Path(output_dir)
                logger.warning("Using fallback pipeline - limited functionality")
            
            def answer_question(self, question, **kwargs):
                return {
                    "answer": f"System is in fallback mode. Cannot properly answer: {question}",
                    "confidence": 0.0,
                    "sources_used": 0,
                    "retrieved_contexts": [],
                    "images": [],
                    "total_images": 0,
                    "generation_method": "fallback"
                }
            
            def get_stats(self):
                return {
                    "total_documents": 0,
                    "status": "fallback_mode",
                    "error": "Module import failed"
                }
        
        return FallbackPipeline()
    
    __all__ = ['setup_and_run_rag_system']
    __version__ = "1.0.0"