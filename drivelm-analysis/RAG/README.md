# RAG Module

This module implements a **Retrieval-Augmented Generation (RAG) system** for the DriveLM dataset, implementing **Step 2** of the GenAI Coding Assignment: RAG Model & Pipeline.

## 📁 Code Structure

```
RAG/
├── __init__.py              # Package initialization with error handling
├── config.py                # Centralized configuration management
├── dataloader.py            # Data loading and FAISS vectorstore creation
├── embedder.py              # Text embedding using sentence-transformers
├── retriever.py             # Document retrieval with FAISS
├── generator.py             # Answer generation using Google Gemini
└── pipeline.py              # Main orchestrator and workflow management
```

## 🎯 What This Module Does

### 1. Configuration Management (`config.py`)
- **Centralized Settings**: All model and system configurations in one place
- **Auto GPU Detection**: Automatically uses CUDA if available, falls back to CPU
- **Model Selection**: Configures embedding, LLM, and VLM models
- **Path Management**: Handles data paths and output directories

### 2. Data Loading (`dataloader.py`)
- **JSON Processing**: Converts unified JSON data to LangChain Documents
- **FAISS Integration**: Creates and manages FAISS vectorstore
- **Metadata Extraction**: Preserves question types, entities, and scene information
- **Statistics Generation**: Provides dataset insights and quality metrics

### 3. Text Embedding (`embedder.py`)
- **Sentence Transformers**: Uses `all-MiniLM-L6-v2` for efficient embeddings
- **Batch Processing**: Memory-efficient processing of large datasets
- **Device Management**: GPU acceleration with CPU fallback
- **LangChain Compatibility**: Implements required interfaces for FAISS

### 4. Document Retrieval (`retriever.py`)
- **FAISS-based Search**: Fast similarity search using Facebook AI Similarity Search
- **LangChain Integration**: Uses LangChain's retriever interface
- **Configurable Top-K**: Adjustable number of retrieved documents
- **Metadata Preservation**: Maintains question context and scene information

### 5. Answer Generation (`generator.py`)
- **Google Gemini Integration**: Uses Gemini-1.5-flash for answer generation
- **Driving Context Prompts**: Specialized prompts for autonomous driving scenarios
- **Short Natural Answers**: Generates concise 2-3 sentence responses
- **Context-Aware**: Uses retrieved documents to inform generation

### 6. Pipeline Orchestration (`pipeline.py`)
- **Complete Workflow**: Manages entire RAG pipeline from data loading to answer generation
- **Phase-based Setup**: Initialization → Index Building → Model Loading → Question Answering
- **Error Handling**: Graceful error recovery and logging
- **Statistics Tracking**: Comprehensive system monitoring and reporting

## 🚀 Key Features

### Multi-Modal RAG System
- **Text + Metadata**: Processes questions, answers, and scene metadata
- **Image Path Integration**: Links questions to corresponding camera images
- **Entity-Aware Retrieval**: Uses extracted entities for better context matching
- **Scene Context**: Incorporates driving scenario information

### Efficient Implementation
- **FAISS Vectorstore**: Fast approximate nearest neighbor search
- **Sentence Transformers**: Lightweight, high-quality embeddings
- **Google Gemini API**: Cloud-based generation with API key support
- **Batch Processing**: Efficient handling of multiple questions

### Robust Architecture
- **Modular Design**: Each component can be used independently
- **Configuration-Driven**: Easy to modify models and parameters
- **Error Recovery**: Handles missing data and model failures gracefully
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 📊 Models Used

### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Justification**: Fast, efficient, good balance of speed and quality
- **Dimension**: 384-dimensional embeddings
- **Performance**: Optimized for semantic similarity in conversational contexts

### Language Model  
- **Model**: Google Gemini-1.5-flash
- **Justification**: Fast inference, strong reasoning capabilities, API-based scalability
- **Context**: Specialized prompts for driving assistance scenarios
- **Output**: Natural, concise responses suitable for in-vehicle interaction

### Vector Database
- **System**: FAISS (Facebook AI Similarity Search)
- **Justification**: Fast, memory-efficient, no external dependencies
- **Index Type**: Flat index for exact search
- **Similarity**: Cosine similarity for semantic matching

## 🔧 Usage

### Basic Pipeline Setup
```python
from RAG import setup_and_run_rag_system

# Initialize complete RAG system
pipeline = setup_and_run_rag_system(
    output_dir="results",
    force_rebuild=False,
    load_vision=True
)

# Answer questions
result = pipeline.answer_question(
    "Is the car in front of me braking?",
    top_k=5,
    use_vision=True
)
print(result['answer'])
```

### Manual Component Usage
```python
from RAG import create_rag_pipeline

# Create pipeline with custom config
pipeline = create_rag_pipeline(
    output_dir="results",
    config_override={'top_k': 10}
)

# Run setup phases
pipeline.initialize_components()
pipeline.build_vector_index(force_rebuild=False)
pipeline.load_generation_models(load_vision=True)

# Answer questions
result = pipeline.answer_question("What objects are visible?")
```

### Batch Processing
```python
questions = [
    "Is it safe to change lanes?",
    "What is the speed limit here?",
    "Are there pedestrians nearby?"
]

results = pipeline.batch_answer_questions(
    questions,
    output_file="batch_results.json"
)
```

## 📈 Pipeline Phases

### Phase 1: Component Initialization
- Load configuration settings
- Initialize embedding model
- Setup FAISS retriever
- Initialize Gemini generator

### Phase 2: Vector Index Building
- Load unified JSON data
- Convert to LangChain Documents
- Generate embeddings for all texts
- Build and save FAISS index

### Phase 3: Model Loading
- Initialize Gemini API connection
- Load text generation model
- Setup vision capabilities (if enabled)
- Verify model functionality

### Phase 4: Question Answering
- Embed user question
- Retrieve top-k relevant documents
- Generate context-aware answer
- Return structured response with sources

## 📊 Results Structure

### Answer Response Format
```json
{
  "answer": "The car in front appears to be slowing down based on the brake lights visible in the image.",
  "confidence": 0.9,
  "sources_used": 3,
  "retrieved_contexts": [...],
  "images": [
    {"path": "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg", "rank": 1}
  ],
  "total_images": 1,
  "question": "Is the car in front of me braking?",
  "retrieval_strategy": "hybrid"
}
```

### System Statistics
```json
{
  "pipeline_status": {
    "initialized": true,
    "index_built": true,
    "models_loaded": true
  },
  "retriever": {
    "total_items": 1000,
    "embedding_dimension": 384,
    "similarity_metric": "cosine"
  },
  "generator": {
    "text_model_loaded": true,
    "vision_model_loaded": false,
    "device": "google-cloud",
    "gemini_model": "gemini-1.5-flash"
  }
}
```

## 🛠 Technical Implementation

### Data Flow
1. **Input**: User question + unified dataset
2. **Embedding**: Question embedded using sentence-transformers
3. **Retrieval**: FAISS similarity search for top-k documents
4. **Context**: Retrieved documents formatted as context
5. **Generation**: Gemini generates answer using context
6. **Output**: Structured response with sources and confidence

### Error Handling
- **Graceful Degradation**: System continues with reduced functionality on component failures
- **Fallback Modes**: CPU fallback for GPU failures, text-only mode for vision failures
- **Comprehensive Logging**: Detailed error tracking and debugging information
- **Input Validation**: Robust handling of malformed data and edge cases

### Performance Optimizations
- **Batch Embedding**: Process multiple texts efficiently
- **Index Caching**: Save/load FAISS index to avoid rebuilding
- **Memory Management**: Efficient handling of large datasets
- **API Rate Limiting**: Proper handling of Gemini API limits

## 🎯 Assignment Requirements Met

✅ **Model Selection**: Open-source sentence-transformers + cloud-based Gemini  
✅ **Architecture Documentation**: Clear explanation of RAG components and design choices  
✅ **Information Retrieval**: FAISS-based semantic search with metadata filtering  
✅ **Multi-modal Processing**: Text + image path integration  
✅ **Generation Pipeline**: Context-aware answer generation with source attribution  
✅ **Code Quality**: Modular design, error handling, comprehensive logging  

## 🔑 API Configuration

### Google Gemini Setup
```python
# In RAG/config.py, find LLMConfig and update:
@dataclass
class LLMConfig:
    model_name: str = "microsoft/DialoGPT-medium"
    api_key: str = "your_new_api_key_here"  # Update this line
    max_length: int = 512
    temperature: float = 0.7
    device: str = "cpu"
```

Or the system will use the hardcoded key for development/testing.

## 📝 Configuration Options

### Key Settings (config.py)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Top-K Retrieval**: 5 documents (configurable)
- **Max Context Length**: 2048 tokens
- **Temperature**: 0.7 for balanced creativity/accuracy
- **Batch Size**: 32 for efficient processing

### Path Configuration
- **Data Path**: `results/unified_rag_enhanced.json`
- **Vector DB**: `results/vector_db/`
- **Output Directory**: Configurable base path

## 🚀 Ready for Integration

This RAG module is designed to integrate seamlessly with your existing pipeline:
- Follows same architectural patterns as data_analysis module
- Uses same output directory structure
- Provides similar logging and error handling
- Ready for main.py orchestration