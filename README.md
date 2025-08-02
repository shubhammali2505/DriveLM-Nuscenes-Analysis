# DriveLM Dataset Analysis & RAG System

A comprehensive multi-modal question-answering system for autonomous driving scenarios using the DriveLM dataset built on NuScenes v1.0-mini. This project implements data analysis, retrieval-augmented generation (RAG), and evaluation components with an interactive Streamlit web interface.

## 🚗 Project Overview

This system provides end-to-end analysis and question-answering capabilities for autonomous driving scenarios:

- **Data Analysis**: Comprehensive parsing and statistical analysis of DriveLM dataset
- **RAG System**: Multi-modal retrieval-augmented generation using sentence transformers and Google Gemini
- **Evaluation**: Quantitative and qualitative performance assessment with multiple metrics
- **Web Interface**: Interactive Streamlit application for pipeline execution and Q&A

## 📁 Project Structure

```
drivelm_analysis/
├── data_analysis/              # Data parsing, analysis, and visualization
│   ├── __init__.py
│   ├── parsers.py             # NuScenes and DriveLM data parsers
│   ├── analysis.py            # Statistical analysis and pattern recognition
│   ├── visualizations.py     # Chart generation and dashboards
│   └── rag_enhancer.py        # RAG optimization enhancements
├── RAG/                       # Retrieval-Augmented Generation system
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── dataloader.py          # Data loading and FAISS vectorstore
│   ├── embedder.py            # Text embedding using sentence-transformers
│   ├── retriever.py           # Document retrieval with FAISS
│   ├── generator.py           # Answer generation using Google Gemini
│   └── pipeline.py            # RAG pipeline orchestration
├── evaluation/                # System evaluation and metrics
│   ├── __init__.py
│   ├── evaluator.py           # Main evaluation orchestrator
│   ├── metrics.py             # Quantitative metrics (BLEU, ROUGE, METEOR)
│   ├── qualitative.py         # Error analysis and failure cases
│   └── visualizer.py          # Evaluation result visualization
├── output/                    # Generated results (created automatically)
├── results/                   # Analysis outputs (created automatically)
├── visualizations/            # Generated charts (created automatically)
├── main.py                    # Complete pipeline orchestrator
├── app.py                     # Streamlit web application
├── startup.py                 # Application launcher with system checks
├── requirements.txt           # Python dependencies
├── analysis.log              # Application logs
└── README.md                  # This file
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**Prerequisites**: Docker and Docker Compose installed

```bash
# Navigate to the project directory
cd drivelm_analysis

# Build and run with Docker Compose
docker-compose up --build

# Access the application at http://localhost:8501
```

The Docker container automatically runs `startup.py` which launches the Streamlit interface.

### Option 2: Local Installation

**Prerequisites**: Python 3.8+ and pip

```bash
# Navigate to the project directory
cd drivelm_analysis

# Install dependencies
pip install -r requirements.txt

# Launch the application
python3 startup.py

# Alternative: Direct Streamlit launch
streamlit run app.py
```

## 🛠 Data Path Configuration

### Expected Data Structure
```
../data/                       # Data directory (sibling to drivelm_analysis)
├── drivelm_data/
│   └── train_sample.json     # DriveLM question-answer data
└── nusccens/                 # NuScenes v1.0-mini dataset
    └── v1.0-mini/
        ├── scene.json
        ├── sample.json
        ├── sample_data.json
        └── ...
```

### Path Configuration
The system expects data in the parent directory (`../data/`). Update paths in:

1. **Streamlit Interface**: Configure paths in the Pipeline tab
2. **Command Line**: Use flags when running `main.py`
3. **Environment Variables**: Set `NUSCENES_PATH` and `DRIVELM_PATH`

```bash
# Example with custom paths
python main.py --nuscenes_path ../data/nusccens --drivelm_path ../data/drivelm_data/train_sample.json
```

## 📊 Features

### 1. Complete Data Analysis Pipeline (`main.py`)
- **Multi-modal Data Parsing**: Links questions to scenes, images, and vehicle states
- **Statistical Analysis**: Distribution analysis, pattern recognition, quality assessment
- **Visualization Generation**: Interactive charts and static publication-quality figures
- **RAG Enhancement**: Optimization for retrieval-augmented generation systems

### 2. RAG Question-Answering System
- **Semantic Retrieval**: FAISS-based similarity search with sentence transformers
- **Multi-modal Context**: Integrates text, metadata, and image information
- **Advanced Generation**: Google Gemini-powered answer generation
- **Configurable Pipeline**: Adjustable retrieval strategies and model parameters

### 3. Comprehensive Evaluation
- **Quantitative Metrics**: BLEU, ROUGE, METEOR, F1, accuracy scores
- **Qualitative Analysis**: Error categorization and failure case identification
- **Visual Assessment**: Distribution plots and quality categorization
- **Performance Benchmarking**: System performance across different question types

### 4. Interactive Web Interface (`app.py`)
- **Pipeline Execution**: Real-time monitoring of data processing
- **Interactive Q&A**: Natural language interface for driving questions
- **Results Visualization**: Dynamic charts and data exploration
- **System Management**: Component status monitoring and configuration

## 🔧 Configuration

### Google Gemini API Key Setup
**Option 1: Environment Variable (Recommended)**
```bash
# Set environment variable
export GEMINI_API_KEY="your_api_key_here"
```

**Option 2: Direct Configuration (If API key expires)**
If the Gemini API key expires, update your API key (free version) in:
- **File**: `RAG/config.py`
- **Location**: Inside `LLMConfig` class
- **Field**: `api_key` parameter

```python
# In RAG/config.py, find LLMConfig and update:
@dataclass
class LLMConfig:
    model_name: str = "your gemini model"
    api_key: str = "your_new_api_key_here"  # Update this line
    max_length: int = 512
    temperature: float = 0.7
    device: str = "cpu"
```

### Data Paths Configuration
Update paths in the Streamlit interface or via command line:
```bash
# Data paths (relative to project root)
export NUSCENES_PATH="../data/nusccens"
export DRIVELM_PATH="../data/drivelm_data/train_sample.json"
```

### Docker Environment
Environment variables are configured in `docker-compose.yml`. Update the data volume mounts if your data is in a different location:

```yaml
volumes:
  - ../data:/app/data:ro  # Mounts sibling data directory
```

## 📈 Usage Workflow

### 1. Pipeline Execution
```bash
# Run complete analysis pipeline
python main.py --nuscenes_path ../data/nusccens --drivelm_path ../data/drivelm_data/train_sample.json

# Or use the web interface Pipeline tab
```

### 2. Interactive Q&A
Access the Streamlit interface and:
1. Run the pipeline in the **Pipeline** tab
2. Load RAG system in the **RAG System** tab
3. Ask questions about driving scenarios

### 3. Results Analysis
- View analysis results in **Data Analysis** tab
- Explore interactive visualizations in **Dashboards** tab
- Review system performance in **Evaluation** tab

## 📊 Generated Outputs

### Core Results (`results/` or `output/`)
- `unified.json` - Structured dataset linking questions to scenes
- `unified_rag_enhanced.json` - RAG-optimized dataset with retrieval features
- `analysis_results.json` - Complete statistical analysis
- `analysis_summary.txt` - Human-readable findings
- `findings.md` - Comprehensive analysis report

### Visualizations (`visualizations/`)
- `index.html` - Interactive dashboard
- Various PNG charts for question types, objects, and scene analysis
- Interactive HTML visualizations with Plotly

### Evaluation Results (`results/evaluation/`)
- `rag_evaluation.json` - Performance metrics and detailed results
- Error analysis and performance visualization plots

## 🎯 Key Capabilities

### Question Types Supported
- **Object Detection**: "What objects are visible in the scene?"
- **Safety Assessment**: "Is it safe to change lanes now?"
- **Spatial Reasoning**: "Where is the pedestrian located?"
- **Motion Prediction**: "Will the car in front keep moving?"
- **Planning Decisions**: "What should I do at this intersection?"

### Advanced Features
- **Multi-camera Integration**: Processes all 6 NuScenes camera views
- **Contextual Retrieval**: Uses scene complexity and object relationships
- **Confidence Scoring**: Provides answer confidence estimates
- **Real-time Processing**: Efficient pipeline for interactive use

## 🚨 Troubleshooting

### Common Issues

**Pipeline Fails to Start**
- Check data paths: `../data/nusccens` and `../data/drivelm_data/train_sample.json`
- Verify Python dependencies are installed
- Ensure sufficient disk space (2GB+)

**RAG System Not Loading**
- Verify GEMINI_API_KEY is set correctly
- Check internet connection for API access
- Ensure FAISS and sentence-transformers are installed

**Docker Issues**
- **Fallback**: Use `python3 startup.py` or `streamlit run app.py`
- Check Docker daemon is running
- Verify port 8501 is available
- Ensure data directory is properly mounted

**Data Path Issues**
- Verify data directory structure matches expected layout
- Check file permissions for data access
- Update paths in configuration or command line arguments

### Getting Help
1. Run `python3 startup.py` for system diagnostics
2. Check `analysis.log` for detailed error messages
3. Verify data paths and file permissions
4. Ensure all dependencies are correctly installed

## 🐳 Docker Details

### Volume Mounting
The Docker setup mounts:
- `../data` → `/app/data` (read-only data access)
- `./results` → `/app/results` (persistent results)
- `./output` → `/app/output` (persistent outputs)

### Custom Data Paths
If your data is in a different location, update `docker-compose.yml`:
```yaml
volumes:
  - /your/custom/path:/app/data:ro
```

## 📄 System Requirements

### Core Dependencies
- Python 3.8+
- PyTorch (CPU/GPU support)
- Streamlit for web interface
- FAISS for vector search
- Google Gemini API access

### Data Requirements
- NuScenes v1.0-mini dataset (~1GB)
- DriveLM question-answering data (~100MB)
- Approximately 2GB storage for processed results

### Performance
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 3GB total (data + results)
- **Compute**: CPU sufficient, GPU accelerates embedding generation

## 📄 License

This project is developed for academic and research purposes. Please ensure compliance with NuScenes and DriveLM dataset licensing terms.

---

**🚗 DriveLM Analysis Suite** - Comprehensive Multi-modal Question-Answering for Autonomous Driving

**Quick Start**: `docker-compose up --build` or `python3 startup.py`
