
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from .config import get_config, update_config
from .dataloader import create_dataloader
from .embedder import create_embedder
from .retriever import create_retriever
from .generator import create_generator

logger = logging.getLogger(__name__)


class RAGPipeline:
    
    def __init__(
        self,
        output_dir: str = "output",
        config_override: Dict[str, Any] = None
    ):
        """Initialize RAG pipeline."""
        self.output_dir = Path(output_dir)
        self.config = get_config()
        
        if config_override:
            update_config(**config_override)
            self.config = get_config()
        
        self.config.data_dir = str(self.output_dir)
        self.config.unified_data_path = str(self.output_dir / "unified_rag_enhanced.json")
        self.config.vector_db_path = str(self.output_dir / "vector_db")
        
        self.dataloader = None
        self.embedder = None
        self.retriever = None
        self.generator = None
        
        self.is_initialized = False
        self.is_index_built = False
        self.is_models_loaded = False
    
    def initialize_components(self) -> bool:

        try:
            logger.info("=== RAG PIPELINE INITIALIZATION ===")
            start_time = time.time()
            
            logger.info("Initializing dataloader...")
            self.dataloader = create_dataloader()
            
            logger.info("Initializing embedder...")
            self.embedder = create_embedder()
            
            logger.info("Initializing retriever...")
            self.retriever = create_retriever()
            if not self.retriever.initialize():
                logger.error("Failed to initialize retriever")
                return False
            
            logger.info("Initializing generator...")
            self.generator = create_generator()
            
            init_time = time.time() - start_time
            logger.info(f"Components initialized in {init_time:.2f} seconds")
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    def build_vector_index(self, force_rebuild=False) -> bool:
        try:
            logger.info("Building vector index...")
            retriever = self.retriever

            if force_rebuild or not retriever.load_index():
                from .dataloader import create_dataloader
                loader = create_dataloader()
                if not loader.load_data() or not loader.process_items():
                    logger.error("❌ Failed to load/process items for index")
                    return False
                if not retriever.build_index(loader.get_processed_items()):
                    return False
                retriever.save_index()

            logger.info("✅ Vector index ready")
            self.is_index_built = True   
            return True
        except Exception as e:
            logger.error(f"❌ Error building vector index: {e}")
            return False



    def load_generation_models(self, load_vision: bool = True) -> bool:

        try:
            logger.info("=== LOADING GENERATION MODELS ===")
            start_time = time.time()
            
            if not self.generator:
                logger.error("Generator not initialized")
                return False
            
            logger.info("Loading text generation model...")
            if not self.generator.initialize_text_model():
                logger.error("Failed to load text model")
                return False
            
            if load_vision:
                logger.info("Loading vision-language model...")
                if not self.generator.initialize_vision_model():
                    logger.warning("Failed to load vision model - continuing with text-only")
            
            model_time = time.time() - start_time
            logger.info(f"Generation models loaded in {model_time:.2f} seconds")
            self.is_models_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading generation models: {e}")
            return False
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        use_vision: bool = True,
        retrieval_strategy: str = "hybrid",
        display_images: bool = True 
    ) -> Dict[str, Any]:

        try:
            if not self.is_index_built:
                return {
                    'error': 'Vector index not built',
                    'answer': 'Please build the vector index first'
                }

            if not self.is_models_loaded:
                return {
                    'error': 'Generation models not loaded',
                    'answer': 'Please load generation models first'
                }

            if not isinstance(question, str):
                question = str(question)

            logger.debug(f"Retrieving contexts for: {question[:50]}...")

            retrieved_items = self.retriever.retrieve(
                question,
                top_k=top_k,
                strategy=retrieval_strategy
            )


            if not retrieved_items:
                return {
                    'answer': 'I could not find relevant information to answer your question.',
                    'confidence': 0.0,
                    'sources_used': 0,
                    'retrieved_items': []
                }

            logger.debug("Generating answer...")
            result = self.generator.generate_answer(
                question,
                retrieved_items,
                use_vision=use_vision
            )

            result['retrieved_items'] = [
                {
                    'id': getattr(item, "item_id", i),
                    'score': getattr(item, "score", 1.0),
                    'text': getattr(item, "text", "")[:200] + "..." if len(getattr(item, "text", "")) > 200 else getattr(item, "text", ""),
                    'has_image': bool(getattr(item, "image_path", None))
                }
                for i, item in enumerate(retrieved_items)
            ]

            result['question'] = question
            result['retrieval_strategy'] = retrieval_strategy

            if not display_images:
                result["images"] = []
                result["total_images"] = 0
            else:
                result["total_images"] = len(result.get("images", []))   # <-- Add this

            return result

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                'error': str(e),
                'answer': 'An error occurred while processing your question.',
                'confidence': 0.0
            }

    def batch_answer_questions(
        self,
        questions: List[str],
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        results = []
        
        logger.info(f"Processing {len(questions)} questions in batch...")
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            
            result = self.answer_question(question)
            results.append({
                'question_id': i,
                'question': question,
                **result
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i+1}/{len(questions)} questions")
        
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        
        return results
    
    def evaluate_on_test_set(
        self,
        test_file: str,
        output_file: str = "evaluation_results.json"
    ) -> Dict[str, Any]:

        try:
            logger.info("=== RAG SYSTEM EVALUATION ===")
            
            test_path = self.output_dir / test_file
            if not test_path.exists():
                logger.error(f"Test file not found: {test_path}")
                return {'error': 'Test file not found'}
            
            with open(test_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            if isinstance(test_data, dict) and 'data' in test_data:
                test_items = test_data['data']
            else:
                test_items = test_data
            
            logger.info(f"Evaluating on {len(test_items)} test items")
            
            results = []
            correct_answers = 0
            total_confidence = 0.0
            
            for i, item in enumerate(test_items):
                if 'question' not in item:
                    continue
                
                question_data = item['question']
                question = question_data.get('question', '')
                reference_answer = question_data.get('answer', '')
                
                if not question or not reference_answer:
                    continue
                
                result = self.answer_question(question)
                generated_answer = result.get('answer', '')
                confidence = result.get('confidence', 0.0)
                
                ref_words = set(reference_answer.lower().split())
                gen_words = set(generated_answer.lower().split())
                overlap = len(ref_words & gen_words) / len(ref_words | gen_words) if ref_words or gen_words else 0
                
                is_correct = overlap > 0.3  
                if is_correct:
                    correct_answers += 1
                
                total_confidence += confidence
                
                results.append({
                    'question_id': i,
                    'question': question,
                    'reference_answer': reference_answer,
                    'generated_answer': generated_answer,
                    'confidence': confidence,
                    'word_overlap': overlap,
                    'is_correct': is_correct,
                    'sources_used': result.get('sources_used', 0)
                })
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Evaluated {i+1}/{len(test_items)} items")
            
            n_items = len(results)
            accuracy = correct_answers / n_items if n_items > 0 else 0.0
            avg_confidence = total_confidence / n_items if n_items > 0 else 0.0
            avg_overlap = sum(r['word_overlap'] for r in results) / n_items if n_items > 0 else 0.0
            avg_sources = sum(r['sources_used'] for r in results) / n_items if n_items > 0 else 0.0
            
            evaluation_results = {
                'total_items': n_items,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'avg_word_overlap': avg_overlap,
                'avg_sources_used': avg_sources,
                'correct_answers': correct_answers,
                'detailed_results': results
            }
            
            output_path = self.output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation completed: {accuracy:.2%} accuracy")
            logger.info(f"Results saved to {output_path}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return {'error': str(e)}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        stats = {
            'pipeline_status': {
                'initialized': self.is_initialized,
                'index_built': self.is_index_built,
                'models_loaded': self.is_models_loaded
            },
            'configuration': {
                'output_dir': str(self.output_dir),
                'data_path': self.config.unified_data_path,
                'vector_db_path': self.config.vector_db_path
            }
        }
        
        if self.dataloader:
            stats['dataloader'] = self.dataloader.get_statistics()
        
        if self.retriever and self.is_index_built:
            stats['retriever'] = self.retriever.get_statistics()
        
        if self.generator and self.is_models_loaded:
            stats['generator'] = self.generator.get_model_info()
        
        return stats
    
    def run_complete_setup(
        self,
        force_rebuild_index: bool = False,
        load_vision_model: bool = True
    ) -> bool:
        
        try:
            logger.info("=== RAG SYSTEM COMPLETE SETUP ===")
            total_start_time = time.time()
            
            if not self.initialize_components():
                logger.error("Failed to initialize components")
                return False
            
            if not self.build_vector_index(force_rebuild=force_rebuild_index):
                logger.error("Failed to build vector index")
                return False
            
            if not self.load_generation_models(load_vision=load_vision_model):
                logger.error("Failed to load generation models")
                return False
            
            total_time = time.time() - total_start_time
            logger.info("=" * 60)
            logger.info(f"RAG SYSTEM SETUP COMPLETED in {total_time:.2f} seconds")
            logger.info("=" * 60)
            
            self._print_system_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in complete setup: {e}")
            return False
    
    def _print_system_status(self) -> None:
        stats = self.get_system_statistics()
        
        print("\n" + "=" * 50)
        print("RAG SYSTEM STATUS")
        print("=" * 50)
        
        status = stats['pipeline_status']
        print(f"✅ Components Initialized: {status['initialized']}")
        print(f"✅ Vector Index Built: {status['index_built']}")
        print(f"✅ Generation Models Loaded: {status['models_loaded']}")
        
        if 'retriever' in stats:
            retriever_stats = stats['retriever']
            print(f"\n📊 Vector Database:")
            print(f"  • Total Items: {retriever_stats.get('total_items', 0)}")
            print(f"  • Embedding Dimension: {retriever_stats.get('embedding_dimension', 0)}")
            print(f"  • Similarity Metric: {retriever_stats.get('similarity_metric', 'N/A')}")
        
        if 'generator' in stats:
            gen_stats = stats['generator']
            print(f"\n🤖 Generation Models:")
            print(f"  • Text Model: {gen_stats.get('text_model_loaded', False)}")
            print(f"  • Vision Model: {gen_stats.get('vision_model_loaded', False)}")
            print(f"  • Device: {gen_stats.get('device', 'N/A')}")
        
        print("\n🚀 System ready for question answering!")
        print("=" * 50)


def create_rag_pipeline(
    output_dir: str = "output",
    config_override: Dict[str, Any] = None
) -> RAGPipeline:
 
    return RAGPipeline(output_dir, config_override)


def setup_and_run_rag_system(
    output_dir: str = "output",
    config_override: Dict[str, Any] = None,
    force_rebuild: bool = False,
    load_vision: bool = True
) -> Optional[RAGPipeline]:
   
    try:
        pipeline = create_rag_pipeline(output_dir, config_override)
        
        if pipeline.run_complete_setup(
            force_rebuild_index=force_rebuild,
            load_vision_model=load_vision
        ):
            return pipeline
        else:
            logger.error("RAG system setup failed")
            return None
            
    except Exception as e:
        logger.error(f"Error setting up RAG system: {e}")
        return None