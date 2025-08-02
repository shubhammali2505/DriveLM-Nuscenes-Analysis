# Evaluation Module

This module provides comprehensive evaluation tools for the RAG system, implementing **Step 3** of the GenAI Coding Assignment: Evaluation and Visualization.

## 📁 Code Structure

```
evaluation/
├── __init__.py              # Package initialization and exports
├── evaluator.py             # Main RAG evaluation orchestrator
├── metrics.py               # Quantitative evaluation metrics
├── qualitative.py           # Qualitative error analysis tools
└── visualizer.py            # Visualization for evaluation results
```

## 🎯 What This Module Does

### 1. Main Evaluation (`evaluator.py`)
- **RAGEvaluator Class**: Orchestrates the complete evaluation process
- **Precomputed Results Loading**: Efficiently loads previously computed evaluation results
- **Output Management**: Handles evaluation result storage and retrieval
- **API Compatibility**: Maintains consistent interface for different evaluation modes

### 2. Quantitative Metrics (`metrics.py`)
- **Accuracy Computation**: Exact match accuracy between predictions and references
- **Precision/Recall/F1**: Token-level precision, recall, and F1 scores
- **BLEU Score**: Standard machine translation metric using NLTK
- **ROUGE-L**: Longest Common Subsequence-based ROUGE metric
- **METEOR Score**: Semantic similarity metric accounting for synonyms
- **Text Normalization**: Consistent preprocessing for fair comparisons

### 3. Qualitative Analysis (`qualitative.py`)
- **Error Analysis**: Identifies and categorizes failure cases
- **Failure Case Sampling**: Random sampling of incorrect predictions for inspection
- **Pattern Recognition**: Helps identify systematic errors in the RAG system
- **Human-Readable Outputs**: Formats results for manual review

### 4. Result Visualization (`visualizer.py`)
- **EvaluationVisualizer Class**: Creates comprehensive evaluation charts
- **Similarity Distribution**: Histograms of semantic similarity scores
- **Confidence Analysis**: Distribution plots of prediction confidence
- **Metrics Bar Charts**: Visual comparison of different evaluation metrics
- **Quality Categorization**: Groups answers into Good/Medium/Poor categories
- **Sample Display**: Shows example predictions vs references

## 🚀 Key Features

### Comprehensive Metrics Suite
- **Standard NLP Metrics**: BLEU, ROUGE, METEOR for text generation quality
- **Information Retrieval Metrics**: Precision, recall, F1 for retrieval accuracy
- **Custom Metrics**: Domain-specific metrics for driving scenarios
- **Semantic Similarity**: Advanced similarity measures beyond exact match

### Efficient Evaluation Process
- **Precomputed Results**: Loads cached evaluation results for fast analysis
- **Batch Processing**: Handles large-scale evaluation efficiently
- **Error Handling**: Graceful handling of edge cases and missing data
- **Progress Tracking**: Clear logging of evaluation progress

### Visual Analysis Tools
- **Distribution Plots**: Understanding score distributions across test set
- **Quality Assessment**: Visual categorization of answer quality
- **Comparative Analysis**: Side-by-side comparison of different approaches
- **Interactive Exploration**: Tools for detailed result inspection

## 📊 Metrics Computed

### 1. Accuracy Metrics
```python
# Exact match accuracy
accuracy = correct_predictions / total_predictions

# Token-level precision, recall, F1
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)
```

### 2. Text Generation Metrics
```python
# BLEU score (machine translation quality)
bleu_score = sentence_bleu(reference_tokens, prediction_tokens)

# ROUGE-L (longest common subsequence)
rouge_l = lcs_length / max(len(reference), len(prediction))

# METEOR (semantic similarity with synonyms)
meteor_score = meteor(reference_tokens, prediction_tokens)
```

### 3. Quality Categories
- **Good**: Semantic similarity > 0.7
- **Medium**: Semantic similarity 0.4-0.7
- **Poor**: Semantic similarity < 0.4

## 🔧 Usage

### Basic Evaluation
```python
from evaluation import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator(
    rag_pipeline=pipeline,
    data_file="results/unified_rag_enhanced.json"
)

# Run evaluation
results = evaluator.evaluate(
    output_dir="results/evaluation",
    limit=100  # Evaluate on first 100 samples
)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"F1 Score: {results['f1']:.3f}")
print(f"BLEU Score: {results['bleu']:.3f}")
```

### Individual Metrics
```python
from evaluation.metrics import (
    compute_accuracy,
    compute_bleu,
    compute_rouge,
    compute_meteor
)

predictions = ["The car is braking", "Turn left ahead"]
references = ["The vehicle is stopping", "Make a left turn"]

# Compute individual metrics
accuracy = compute_accuracy(predictions, references)
bleu = compute_bleu(predictions, references)
rouge = compute_rouge(predictions, references)
meteor = compute_meteor(predictions, references)
```

### Error Analysis
```python
from evaluation.qualitative import analyze_errors, sample_failure_cases

# Analyze failure cases
errors = analyze_errors(evaluation_results, top_n=10)
samples = sample_failure_cases(evaluation_results, num_samples=5)

for error in errors:
    print(f"Question: {error['question']}")
    print(f"Expected: {error['reference']}")
    print(f"Got: {error['prediction']}")
    print("---")
```

### Visualization
```python
from evaluation.visualizer import EvaluationVisualizer
import pandas as pd

# Create visualizer
df = pd.DataFrame(evaluation_results)
visualizer = EvaluationVisualizer(df)

# Generate plots
similarity_plot = visualizer.plot_similarity_distribution(results['per_sample'])
confidence_plot = visualizer.plot_confidence_distribution()
metrics_plot = visualizer.plot_metrics_bar(results['metrics'])
quality_plot = visualizer.plot_quality_distribution(results['per_sample'])

# Show example predictions
visualizer.show_examples(n=5)
```

## 📈 Evaluation Results Structure

### Overall Results
```json
{
  "accuracy": 0.75,
  "precision": 0.82,
  "recall": 0.78,
  "f1": 0.80,
  "bleu": 0.65,
  "rougeL": 0.70,
  "meteor": 0.68,
  "total_samples": 1000,
  "evaluation_time": 45.2
}
```

### Per-Sample Results
```json
{
  "question": "Is it safe to change lanes?",
  "reference": "No, there is a car in the blind spot",
  "prediction": "Not safe, vehicle detected nearby",
  "confidence": 0.85,
  "semantic_sim": 0.72,
  "bleu_score": 0.45,
  "exact_match": false,
  "quality": "Good"
}
```

### Error Analysis
```json
{
  "total_errors": 250,
  "error_categories": {
    "object_detection": 85,
    "spatial_reasoning": 92,
    "safety_assessment": 45,
    "general_driving": 28
  },
  "common_failure_patterns": [
    "Difficulty with spatial relationships",
    "Inconsistent object identification",
    "Context-dependent safety judgments"
  ]
}
```

## 🎯 Evaluation Methodology

### 1. Quantitative Assessment
- **Standard Metrics**: Uses established NLP evaluation metrics
- **Domain-Specific**: Tailored metrics for driving assistance scenarios
- **Multi-Dimensional**: Evaluates different aspects (accuracy, fluency, relevance)
- **Statistical Significance**: Proper sampling and confidence intervals

### 2. Qualitative Analysis
- **Error Categorization**: Groups failures by type and cause
- **Manual Inspection**: Human review of challenging cases
- **Pattern Recognition**: Identifies systematic weaknesses
- **Improvement Recommendations**: Actionable insights for system enhancement

### 3. Visual Analysis
- **Distribution Analysis**: Understanding score distributions
- **Correlation Studies**: Relationships between different metrics
- **Quality Assessment**: Visual categorization of answer quality
- **Comparative Evaluation**: Side-by-side system comparisons

## 🛠 Technical Implementation

### Text Preprocessing
```python
def normalize_text(text: str) -> str:
    """Normalize text for fair comparison"""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove punctuation
    return text
```

### Semantic Similarity
- Uses token-level overlap for basic similarity
- Implements longest common subsequence for ROUGE-L
- Leverages NLTK for METEOR semantic matching
- Handles edge cases (empty predictions, mismatched lengths)

### Visualization Framework
- Uses matplotlib and seaborn for static plots
- Pandas for data manipulation and analysis
- Customizable plot styles and formats
- Export capabilities for reports and presentations

## 📊 Performance Benchmarks

### Expected Performance Ranges
- **Accuracy**: 60-80% (depending on question complexity)
- **F1 Score**: 65-85% (token-level matching)
- **BLEU Score**: 40-70% (text generation quality)
- **ROUGE-L**: 50-75% (content overlap)
- **METEOR**: 45-70% (semantic similarity)

### Quality Distribution
- **Good Answers**: 40-60% of responses
- **Medium Answers**: 25-35% of responses  
- **Poor Answers**: 10-25% of responses

## 🎯 Assignment Requirements Met

✅ **Quantitative Evaluation**: Multiple metrics (accuracy, BLEU, ROUGE, METEOR, F1)  
✅ **Semantic Similarity**: Advanced similarity measures beyond exact match  
✅ **Qualitative Analysis**: Error categorization and failure case analysis  
✅ **Visualization Tools**: Comprehensive charts and analysis dashboards  
✅ **Performance Analysis**: What works and what doesn't with concrete examples  
✅ **Improvement Suggestions**: Actionable insights from evaluation results  

## 📝 Output Files

### Generated Files
- `evaluation_results.json` - Complete evaluation metrics and results
- `error_analysis.json` - Detailed error categorization and patterns
- `similarity_distribution.png` - Histogram of similarity scores
- `confidence_distribution.png` - Distribution of prediction confidence
- `metrics_comparison.png` - Bar chart of different metrics
- `quality_assessment.png` - Quality category distribution
- `sample_predictions.txt` - Example predictions for manual review

## 🔍 Debugging and Analysis

### Common Issues
- **Low BLEU Scores**: Often due to different phrasing but correct meaning
- **High Confidence, Low Accuracy**: Indicates overconfident wrong predictions
- **Semantic Mismatch**: Good ROUGE but poor METEOR suggests surface-level similarity

### Analysis Recommendations
1. **Review Error Categories**: Focus improvement on most common failure types
2. **Confidence Calibration**: Adjust confidence thresholds based on accuracy
3. **Context Analysis**: Examine which retrieval contexts lead to better answers
4. **Question Type Performance**: Identify which question types need improvement