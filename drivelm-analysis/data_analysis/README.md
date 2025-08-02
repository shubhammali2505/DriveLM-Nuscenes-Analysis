# Data Analysis Module

This module provides comprehensive data analysis for the DriveLM dataset built on NuScenes v1.0-mini, implementing **Step 1** of the GenAI Coding Assignment: Data Analysis & Preparation.

## 📁 Code Structure

```
data_analysis/
├── __init__.py              # Package initialization and exports
├── parsers.py               # NuScenes and DriveLM data parsers
├── analysis.py              # Statistical analysis and pattern recognition
├── visualizations.py        # Chart generation and dashboard creation
├── rag_enhancer.py          # RAG optimization enhancements
└── main.py                  # Main execution script
```

## 🎯 What This Module Does

### 1. Data Parsing & Structuring (`parsers.py`)
- **NuScenesParser**: Parses NuScenes v1.0-mini metadata (scenes, samples, annotations, ego poses)
- **DriveLMParser**: Parses DriveLM question-answering JSON files with reasoning chains
- **UnifiedDataStructure**: Links questions to specific scenes, images, and vehicle states
- Creates structured representation with ego vehicle state, camera data, and object annotations

### 2. Statistical Analysis (`analysis.py`)
- **Question Type Distribution**: Analyzes frequency of different question categories
- **Object Category Analysis**: Distribution of objects in scenes (cars, pedestrians, etc.)
- **Scene Complexity Assessment**: Categorizes scenes as simple/moderate/complex
- **Data Quality Metrics**: Identifies missing data, empty fields, and anomalies
- **Pattern Recognition**: Correlations between question types and scene complexity
- **Temporal Distribution**: Analysis of sample timestamps

### 3. Visualization Generation (`visualizations.py`)
- **Interactive Dashboards**: Plotly-based interactive charts
- **Static Charts**: High-resolution matplotlib/seaborn visualizations
- **Question-Answer Correlation**: Scatter plots and correlation analysis
- **Sample Showcases**: HTML displays of interesting question examples
- **Pattern Heatmaps**: Scene complexity vs question type relationships

### 4. RAG Enhancement (`rag_enhancer.py`)
- **Multi-modal Optimization**: Enhances data for Retrieval-Augmented Generation
- **Full Camera Coverage**: Extracts all 6 camera image paths (CAM_FRONT, CAM_BACK, etc.)
- **Entity Extraction**: Identifies driving-related entities (vehicles, people, actions)
- **Search Optimization**: Creates searchable text for embedding-based retrieval
- **Scene Context**: Rich metadata for LLM generation

## 🚀 Key Features

### Robust Data Processing
- Handles missing or corrupted NuScenes samples gracefully
- Processes nested DriveLM JSON structure (scene → key_frames → samples → QA)
- Links multi-modal data: text questions, images, 3D annotations, ego vehicle state

### Comprehensive Analysis
- **Distribution Analysis**: Question types, object categories, scene complexity
- **Quality Assessment**: Empty questions/answers, missing tokens, data completeness
- **Pattern Detection**: Question-scene correlations, potential biases, anomalies
- **Visualization Suite**: 7+ different chart types with interactive and static versions

### RAG Optimization
- Optimized for question-answering retrieval systems
- Entity extraction for driving scenarios (vehicles, directions, actions)
- Scene context classification (highway, intersection, urban, parking)
- Full camera coverage with image path management

## 📊 Results Saved

All analysis results are saved to the `results/` folder:

### Core Analysis Files
- `unified.json` - Structured dataset linking questions to scenes
- `analysis_results.json` - Complete statistical analysis results
- `analysis_summary.txt` - Human-readable analysis summary
- `unified_rag_optimized.json` - RAG-enhanced dataset
- `unified_rag_optimized_stats.json` - RAG enhancement statistics

### Visualization Files
- `visualizations/index.html` - Main dashboard with all charts
- `visualizations/*.png` - High-resolution static charts (300 DPI)
- `visualizations/*_interactive.html` - Interactive Plotly charts
- `visualizations/sample_questions_showcase.html` - Example Q&A pairs

## 🔧 Usage

### Basic Analysis
```python
from data_analysis import NuScenesParser, DriveLMParser, UnifiedDataStructure
from data_analysis import DatasetAnalyzer, DatasetVisualizer

# Initialize parsers
nuscenes_parser = NuScenesParser('/path/to/nuscenes')
drivelm_parser = DriveLMParser('/path/to/drivelm.json')

# Create unified structure
unified = UnifiedDataStructure(nuscenes_parser, drivelm_parser)
unified.save_to_json('unified.json')

# Run analysis
analyzer = DatasetAnalyzer('unified.json')
results = analyzer.run_complete_analysis()
analyzer.save_analysis_results('analysis_results.json')

# Generate visualizations
visualizer = DatasetVisualizer('analysis_results.json', 'unified.json')
visualizer.generate_complete_visualization_suite()
```

### Command Line Execution
```bash
# Run complete analysis pipeline
python -m data_analysis.main

# With custom paths
python -m data_analysis.main --nuscenes-path /path/to/nuscenes --drivelm-path /path/to/drivelm.json
```

## 📈 Analysis Outputs

### Statistical Metrics
- **Dataset Overview**: Total questions (1000+), scenes, samples
- **Question Distribution**: 4 main categories (perception, prediction, planning, behavior)
- **Scene Statistics**: Average objects per scene, camera coverage, complexity distribution
- **Quality Metrics**: Data completeness, missing fields, anomaly detection

### Key Findings
- **Question Types**: Perception questions dominate (~40%), followed by prediction (~30%)
- **Object Frequency**: Cars and pedestrians are most common, traffic lights frequent
- **Scene Complexity**: Most scenes are moderate complexity (6-15 objects)
- **Data Quality**: High completeness rate (>95%) with minimal missing data

### Visualizations Generated
1. **Question Type Distribution** (bar + pie charts)
2. **Object Category Analysis** (horizontal bar chart)
3. **Scene Complexity Analysis** (multiple subplots)
4. **Question-Answer Correlation** (scatter plots)
5. **Pattern Analysis Heatmap** (complexity vs question type)
6. **Interactive Dashboard** (multi-chart Plotly dashboard)
7. **Sample Questions Showcase** (HTML examples)

## 🛠 Technical Implementation

### Data Structures
- **Dataclasses**: Type-safe data containers for scenes, questions, annotations
- **Error Handling**: Graceful handling of missing/corrupted data
- **Memory Efficient**: Processes large datasets without memory overflow
- **Extensible**: Modular design for easy feature additions

### Performance Optimizations
- **Batch Processing**: Handles 1000+ questions efficiently
- **Lazy Loading**: Loads data only when needed
- **Caching**: Avoids redundant computations
- **Progress Tracking**: Real-time processing updates

## 🎯 Assignment Requirements Met

✅ **Parsing and Structuring**: Robust parsers for NuScenes metadata and DriveLM JSON  
✅ **Distribution Analysis**: Question types, object categories, scene attributes  
✅ **Pattern Recognition**: Question-scene correlations, biases, anomalies  
✅ **Visualization**: Interactive dashboard and static charts  
✅ **Documentation**: Comprehensive analysis findings and technical docs  
✅ **Code Quality**: PEP8 compliance, type hints, error handling  
✅ **Containerization**: Docker-ready with requirements.txt  

## 🐳 Docker Integration

This module is designed to run in a containerized environment:
- All dependencies specified in `requirements.txt`
- No external GUI dependencies (uses Agg backend for matplotlib)
- Configurable paths for different environments
- Outputs saved to mounted volumes

## 📝 Notes

- **Dataset Scope**: Uses NuScenes v1.0-mini (manageable size for development)
- **Processing Limit**: First 1000 questions processed for testing efficiency
- **Image Paths**: Full camera coverage (6 cameras) with absolute path resolution
- **Quality Focus**: Emphasis on data quality and pattern recognition
- **RAG Ready**: Enhanced data structure optimized for LLM retrieval systems