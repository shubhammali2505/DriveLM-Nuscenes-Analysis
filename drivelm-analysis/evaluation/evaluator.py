"""
Evaluator for RAG QA system
Now simplified: ONLY loads precomputed evaluation results from JSON.
"""

import json
from pathlib import Path
from typing import Dict, Any


class RAGEvaluator:
    def __init__(self, rag_pipeline=None, data_file: str = "results/unified_rag_enhanced.json"):
  
        self.rag_pipeline = rag_pipeline
        self.data_file = Path(data_file)

    def evaluate(self, output_dir: str = "results/evaluation", limit: int = None) -> Dict[str, Any]:
       
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "rag_evaluation.json"
        if not json_path.exists():
            raise FileNotFoundError(f"❌ Precomputed evaluation not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"⚡ Loaded precomputed evaluation from {json_path}")
        return data

    def run_evaluation(self, output_dir: str = "results/evaluation", limit: int = None):
        return self.evaluate(output_dir=output_dir, limit=limit)

    def evaluate_on_test_set(self, test_file: str = None, output_file: str = "evaluation_results.json") -> Dict[str, Any]:
     
        return self.evaluate(output_dir="results/evaluation")
