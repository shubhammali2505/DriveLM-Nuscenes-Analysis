"""
Statistical Analysis Module for NuScenes and DriveLM Data

This module provides  statistical analysis of the unified
dataset including distribution analysis, pattern recognition, and anomaly detection.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    question_type_distribution: Dict[str, int]
    object_category_distribution: Dict[str, int]
    scene_statistics: Dict[str, Any]
    temporal_distribution: Dict[str, int]
    quality_metrics: Dict[str, Any]
    patterns_and_anomalies: Dict[str, Any]


class DatasetAnalyzer:
    
    def __init__(self, unified_data_path: str):
        """
        Initialize the analyzer with unified data.
        
        Args:
            unified_data_path: Path to unified JSON data file
        """
        self.data_path = Path(unified_data_path)
        self.data = self._load_data()
        self.analysis_results = None
    
    def _load_data(self) -> List[Dict[str, Any]]:
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} unified data items")
            return data
        except Exception as e:
            logger.error(f"Error loading unified data: {e}")
            return []
    
    def analyze_question_types(self) -> Dict[str, int]:
        question_types = []
        
        for item in self.data:
            question_type = item['question']['question_type']
            question_types.append(question_type)
        
        distribution = dict(Counter(question_types))
        logger.info(f"Found {len(distribution)} unique question types")
        
        return distribution
    
    def analyze_object_categories(self) -> Dict[str, int]:
        object_categories = []
        
        for item in self.data:
            scene_data = item.get('scene_data')
            if scene_data and scene_data['object_annotations']:
                for obj in scene_data['object_annotations']:
                    object_categories.append(obj['category_name'])
        
        distribution = dict(Counter(object_categories))
        logger.info(f"Found {len(distribution)} unique object categories")
        
        return distribution
    
    def analyze_scene_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_scenes': len(set(item['question']['scene_token'] for item in self.data)),
            'total_samples': len(set(item['question']['sample_token'] for item in self.data)),
            'questions_with_scene_data': sum(1 for item in self.data if item['metadata']['has_scene_data']),
            'questions_without_scene_data': sum(1 for item in self.data if not item['metadata']['has_scene_data']),
            'avg_objects_per_scene': 0,
            'avg_cameras_per_scene': 0,
            'scene_complexity_distribution': {}
        }
        
        scenes_with_data = [
            item for item in self.data 
            if item['metadata']['has_scene_data']
        ]
        
        if scenes_with_data:
            stats['avg_objects_per_scene'] = np.mean([
                item['metadata']['num_objects'] 
                for item in scenes_with_data
            ])
            stats['avg_cameras_per_scene'] = np.mean([
                item['metadata']['num_cameras'] 
                for item in scenes_with_data
            ])
            complexity_bins = {'simple': 0, 'moderate': 0, 'complex': 0}
            for item in scenes_with_data:
                num_objects = item['metadata']['num_objects']
                if num_objects <= 5:
                    complexity_bins['simple'] += 1
                elif num_objects <= 15:
                    complexity_bins['moderate'] += 1
                else:
                    complexity_bins['complex'] += 1
            
            stats['scene_complexity_distribution'] = complexity_bins
        
        return stats
    
    def analyze_temporal_distribution(self) -> Dict[str, int]:
        timestamps = []
        
        for item in self.data:
            if item['metadata']['timestamp']:
                timestamps.append(item['metadata']['timestamp'])
        
        if not timestamps:
            return {}
        
        timestamps = np.array(timestamps)
        min_timestamp = timestamps.min()
        normalized_timestamps = (timestamps - min_timestamp) / 1e6
        hour_bins = (normalized_timestamps / 3600).astype(int)
        
        distribution = dict(Counter(hour_bins))
        
        return {f"hour_{k}": v for k, v in distribution.items()}
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        quality_metrics = {
            'total_questions': len(self.data),
            'empty_questions': 0,
            'empty_answers': 0,
            'missing_scene_tokens': 0,
            'missing_sample_tokens': 0,
            'questions_with_reasoning': 0,
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'avg_reasoning_steps': 0
        }
        
        question_lengths = []
        answer_lengths = []
        reasoning_lengths = []
        
        for item in self.data:
            question_data = item['question']
            
            if not question_data['question'].strip():
                quality_metrics['empty_questions'] += 1
            
            if not question_data['answer'].strip():
                quality_metrics['empty_answers'] += 1
            
            if not question_data['scene_token']:
                quality_metrics['missing_scene_tokens'] += 1
            
            if not question_data['sample_token']:
                quality_metrics['missing_sample_tokens'] += 1
            
            question_lengths.append(len(question_data['question'].split()))
            answer_lengths.append(len(question_data['answer'].split()))
            
            reasoning_chain = question_data.get('reasoning_chain', [])
            if reasoning_chain and len(reasoning_chain) > 0:
                quality_metrics['questions_with_reasoning'] += 1
                reasoning_lengths.append(len(reasoning_chain))
        
        if question_lengths:
            quality_metrics['avg_question_length'] = np.mean(question_lengths)
        if answer_lengths:
            quality_metrics['avg_answer_length'] = np.mean(answer_lengths)
        if reasoning_lengths:
            quality_metrics['avg_reasoning_steps'] = np.mean(reasoning_lengths)
        
        return quality_metrics
    
    def identify_patterns_and_anomalies(self) -> Dict[str, Any]:
        patterns = {
            'question_answer_correlation': {},
            'scene_question_patterns': {},
            'object_question_correlation': {},
            'potential_biases': {},
            'data_anomalies': []
        }
        
        qa_pairs = []
        for item in self.data:
            q_len = len(item['question']['question'].split())
            a_len = len(item['question']['answer'].split())
            qa_pairs.append((q_len, a_len))
        
        if qa_pairs:
            q_lengths, a_lengths = zip(*qa_pairs)
            correlation = np.corrcoef(q_lengths, a_lengths)[0, 1]
            patterns['question_answer_correlation'] = {
                'correlation_coefficient': correlation,
                'interpretation': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
            }
        
        scene_question_map = defaultdict(list)
        for item in self.data:
            if item['metadata']['has_scene_data']:
                complexity = 'simple' if item['metadata']['num_objects'] <= 5 else \
                           'moderate' if item['metadata']['num_objects'] <= 15 else 'complex'
                scene_question_map[complexity].append(item['question']['question_type'])
        
        for complexity, question_types in scene_question_map.items():
            patterns['scene_question_patterns'][complexity] = dict(Counter(question_types))
        
        object_question_map = defaultdict(list)
        for item in self.data:
            scene_data = item.get('scene_data')
            if scene_data and scene_data['object_annotations']:
                question_type = item['question']['question_type']
                for obj in scene_data['object_annotations']:
                    object_question_map[obj['category_name']].append(question_type)
        
        for obj_category, question_types in object_question_map.items():
            patterns['object_question_correlation'][obj_category] = dict(Counter(question_types))
        
        question_type_dist = self.analyze_question_types()
        total_questions = sum(question_type_dist.values())
        
        biases = {}
        for q_type, count in question_type_dist.items():
            percentage = (count / total_questions) * 100
            if percentage > 40:
                biases[q_type] = f"Over-represented ({percentage:.1f}%)"
            elif percentage < 5:
                biases[q_type] = f"Under-represented ({percentage:.1f}%)"
        
        patterns['potential_biases'] = biases
        
        anomalies = []
        
        no_scene_data = sum(1 for item in self.data if not item['metadata']['has_scene_data'])
        if no_scene_data > 0:
            anomalies.append(f"{no_scene_data} questions lack corresponding scene data")
        
        for i, item in enumerate(self.data):
            q_len = len(item['question']['question'].split())
            a_len = len(item['question']['answer'].split())
            
            if q_len > 50: 
                anomalies.append(f"Question {i}: Unusually long question ({q_len} words)")
            
            if a_len > 100:  
                anomalies.append(f"Question {i}: Unusually long answer ({a_len} words)")
        
        patterns['data_anomalies'] = anomalies[:10]  
        
        return patterns
    
    def run_complete_analysis(self) -> AnalysisResults:
        logger.info("Starting comprehensive data analysis...")
        
        question_type_dist = self.analyze_question_types()
        object_category_dist = self.analyze_object_categories()
        scene_stats = self.analyze_scene_statistics()
        temporal_dist = self.analyze_temporal_distribution()
        quality_metrics = self.analyze_data_quality()
        patterns_anomalies = self.identify_patterns_and_anomalies()
        
        self.analysis_results = AnalysisResults(
            question_type_distribution=question_type_dist,
            object_category_distribution=object_category_dist,
            scene_statistics=scene_stats,
            temporal_distribution=temporal_dist,
            quality_metrics=quality_metrics,
            patterns_and_anomalies=patterns_anomalies
        )
        
        logger.info("Analysis complete!")
        return self.analysis_results
    
    def save_analysis_results(self, output_path: str) -> None:
        if not self.analysis_results:
            logger.warning("No analysis results to save. Run analysis first.")
            return
        
        try:
            results_dict = {
                'question_type_distribution': self.analysis_results.question_type_distribution,
                'object_category_distribution': self.analysis_results.object_category_distribution,
                'scene_statistics': self.analysis_results.scene_statistics,
                'temporal_distribution': self.analysis_results.temporal_distribution,
                'quality_metrics': self.analysis_results.quality_metrics,
                'patterns_and_anomalies': self.analysis_results.patterns_and_anomalies
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Analysis results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
    
    def generate_summary_report(self) -> str:
        if not self.analysis_results:
            return "No analysis results available. Run analysis first."
        
        report = []
        report.append("=== DATASET ANALYSIS SUMMARY ===\n")
        
        # Basic statistics
        quality = self.analysis_results.quality_metrics
        report.append(f"Total Questions: {quality['total_questions']}")
        report.append(f"Questions with Scene Data: {self.analysis_results.scene_statistics['questions_with_scene_data']}")
        report.append(f"Unique Scenes: {self.analysis_results.scene_statistics['total_scenes']}")
        report.append(f"Average Objects per Scene: {self.analysis_results.scene_statistics['avg_objects_per_scene']:.1f}")
        report.append("")
        
        # Question types
        report.append("=== QUESTION TYPE DISTRIBUTION ===")
        for q_type, count in sorted(self.analysis_results.question_type_distribution.items(), 
                                  key=lambda x: x[1], reverse=True):
            percentage = (count / quality['total_questions']) * 100
            report.append(f"{q_type}: {count} ({percentage:.1f}%)")
        report.append("")
        
        report.append("=== TOP OBJECT CATEGORIES ===")
        top_objects = sorted(self.analysis_results.object_category_distribution.items(), 
                           key=lambda x: x[1], reverse=True)[:10]
        for obj_type, count in top_objects:
            report.append(f"{obj_type}: {count}")
        report.append("")
        
        report.append("=== DATA QUALITY ISSUES ===")
        if quality['empty_questions'] > 0:
            report.append(f"Empty questions: {quality['empty_questions']}")
        if quality['empty_answers'] > 0:
            report.append(f"Empty answers: {quality['empty_answers']}")
        if quality['missing_scene_tokens'] > 0:
            report.append(f"Missing scene tokens: {quality['missing_scene_tokens']}")
        if quality['missing_sample_tokens'] > 0:
            report.append(f"Missing sample tokens: {quality['missing_sample_tokens']}")
        report.append("")
        
        biases = self.analysis_results.patterns_and_anomalies['potential_biases']
        if biases:
            report.append("=== POTENTIAL BIASES ===")
            for bias_type, description in biases.items():
                report.append(f"{bias_type}: {description}")
            report.append("")
        
        anomalies = self.analysis_results.patterns_and_anomalies['data_anomalies']
        if anomalies:
            report.append("=== DATA ANOMALIES ===")
            for anomaly in anomalies[:5]:  # Show top 5
                report.append(f"- {anomaly}")
            report.append("")
        
        return "\n".join(report)


# def main():
#     """Main function to demonstrate analysis usage."""
#     # Example usage
#     unified_data_path = "unified_data.json"  # Update with your path
    
#     # Initialize analyzer
#     analyzer = DatasetAnalyzer(unified_data_path)
    
#     # Run complete analysis
#     results = analyzer.run_complete_analysis()
    
#     # Save results
#     analyzer.save_analysis_results("analysis_results.json")
    
#     # Generate and print summary report
#     summary = analyzer.generate_summary_report()
#     print(summary)
    
#     # Save summary report
#     with open("analysis_summary.txt", "w", encoding='utf-8') as f:
#         f.write(summary)


# if __name__ == "__main__":
#     main()