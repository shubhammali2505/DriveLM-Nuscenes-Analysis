
import sys
import os
if os.name == 'nt': 
    os.system('chcp 65001 >nul') 
import argparse
import logging
import sys
from pathlib import Path
import json
import time
from typing import Optional

from data_analysis.parsers import (
    NuScenesParser, 
    DriveLMParser, 
    UnifiedDataStructure
)
from data_analysis.analysis import DatasetAnalyzer
from data_analysis.visualizations import DatasetVisualizer

try:
    from data_analysis.rag_enhancer import enhance_unified_json_in_project
    RAG_ENHANCER_AVAILABLE = True
except ImportError:
    RAG_ENHANCER_AVAILABLE = False
    logging.warning("RAG enhancer not available - skipping RAG optimization phase")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AnalysisPipeline:
    
    def __init__(
        self, 
        nuscenes_path: str, 
        drivelm_path: str, 
        output_dir: str = "output",
        enable_rag_enhancement: bool = True
    ):
        
        self.nuscenes_path = Path(nuscenes_path)
        self.drivelm_path = Path(drivelm_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_rag_enhancement = enable_rag_enhancement and RAG_ENHANCER_AVAILABLE
        
        self._validate_inputs()
        
        self.nuscenes_parser = None
        self.drivelm_parser = None
        self.unified_structure = None
        self.analyzer = None
        self.visualizer = None
    
    def _validate_inputs(self) -> None:
        if not self.nuscenes_path.exists():
            raise FileNotFoundError(f"NuScenes path not found: {self.nuscenes_path}")
        
        if not self.drivelm_path.exists():
            raise FileNotFoundError(f"DriveLM file not found: {self.drivelm_path}")
        
        version_dir = self.nuscenes_path / "v1.0-mini"
        if not version_dir.exists():
            raise FileNotFoundError(f"NuScenes v1.0-mini directory not found: {version_dir}")
        
        essential_files = ['scene.json', 'sample.json', 'sample_data.json', 
                          'sample_annotation.json', 'category.json']
        
        for file_name in essential_files:
            file_path = version_dir / file_name
            if not file_path.exists():
                logger.warning(f"Essential NuScenes file missing: {file_path}")
        
        logger.info("Input validation completed successfully")
    
    def run_parsing_phase(self) -> None:
        logger.info("=== PHASE 1: DATA PARSING ===")
        start_time = time.time()
        
        try:
            logger.info("Initializing NuScenes parser...")
            self.nuscenes_parser = NuScenesParser(str(self.nuscenes_path))
            
            logger.info("Initializing DriveLM parser...")
            self.drivelm_parser = DriveLMParser(str(self.drivelm_path))
            
            logger.info("Building unified data structure...")
            self.unified_structure = UnifiedDataStructure(
                self.nuscenes_parser, 
                self.drivelm_parser
            )
            
            unified_data_path = self.output_dir / "unified.json"  
            self.unified_structure.save_to_json(str(unified_data_path))
            
            data = self.unified_structure.get_data()
            logger.info(f"Parsing completed: {len(data)} unified data items")
            logger.info(f"Items with scene data: {sum(1 for item in data if item['metadata']['has_scene_data'])}")
            
            parsing_time = time.time() - start_time
            logger.info(f"Parsing phase completed in {parsing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in parsing phase: {e}")
            raise
    
    def run_analysis_phase(self) -> None:
        """Execute the statistical analysis phase."""
        logger.info("=== PHASE 2: STATISTICAL ANALYSIS ===")
        start_time = time.time()
        
        try:
            unified_data_path = self.output_dir / "unified.json"  # Updated path
            self.analyzer = DatasetAnalyzer(str(unified_data_path))
            
            logger.info("Running comprehensive analysis...")
            analysis_results = self.analyzer.run_complete_analysis()
            
            results_path = self.output_dir / "analysis_results.json"
            self.analyzer.save_analysis_results(str(results_path))
            
            logger.info("Generating summary report...")
            summary_report = self.analyzer.generate_summary_report()
            
            summary_path = self.output_dir / "analysis_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            
            logger.info("Analysis Summary:")
            logger.info("-" * 50)
            print(summary_report[:500] + "..." if len(summary_report) > 500 else summary_report)
            
            analysis_time = time.time() - start_time
            logger.info(f"Analysis phase completed in {analysis_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in analysis phase: {e}")
            raise
    
    def run_visualization_phase(self) -> None:
        logger.info("=== PHASE 3: VISUALIZATION GENERATION ===")
        start_time = time.time()
        
        try:
            analysis_results_path = self.output_dir / "analysis_results.json"
            unified_data_path = self.output_dir / "unified.json"  
            
            self.visualizer = DatasetVisualizer(
                str(analysis_results_path),
                str(unified_data_path)
            )
            
            viz_output_dir = self.output_dir / "visualizations"
            self.visualizer.output_dir = viz_output_dir
            
            logger.info("Generating visualization suite...")
            self.visualizer.generate_complete_visualization_suite()
            
            visualization_time = time.time() - start_time
            logger.info(f"Visualization phase completed in {visualization_time:.2f} seconds")
            logger.info(f"Visualizations saved to: {viz_output_dir}")
            
        except Exception as e:
            logger.error(f"Error in visualization phase: {e}")
            raise
    
    def run_rag_enhancement_phase(self) -> None:
        logger.info("=== PHASE 4: RAG ENHANCEMENT ===")
        start_time = time.time()
        
        if not self.enable_rag_enhancement:
            logger.info("RAG enhancement disabled - skipping phase")
            return
        
        try:
            input_path = self.output_dir / "unified.json"
            output_path = self.output_dir / "unified_rag_enhanced.json"
            nuscenes_root = str(self.nuscenes_path)
            
            logger.info("Enhancing unified data with RAG optimization features...")
            logger.info(f"Input: {input_path}")
            logger.info(f"Output: {output_path}")
            
            success = self._run_rag_enhancement_custom(
                str(input_path),
                str(output_path), 
                nuscenes_root
            )
            
            if success:
                logger.info("RAG enhancement completed successfully!")
                
                indices_path = output_path.with_name(f"{output_path.stem}_indices.json")
                stats_path = output_path.with_name(f"{output_path.stem}_statistics.json")
                
                logger.info(f"Enhanced data: {output_path}")
                logger.info(f"Retrieval indices: {indices_path}")
                logger.info(f"Statistics: {stats_path}")
                
                try:
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                    
                    overview = stats.get('dataset_overview', {})
                    logger.info("RAG Enhancement Summary:")
                    logger.info(f"  Total items: {overview.get('total_items', 0)}")
                    logger.info(f"  Items with scene data: {overview.get('items_with_scene_data', 0)}")
                    logger.info(f"  Items with images: {overview.get('items_with_images', 0)}")
                    logger.info(f"  Unique scenes: {overview.get('unique_scenes', 0)}")
                    
                except Exception as e:
                    logger.warning(f"Could not load RAG enhancement statistics: {e}")
                
            else:
                logger.error("RAG enhancement failed")
                
            rag_time = time.time() - start_time
            logger.info(f"RAG enhancement phase completed in {rag_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in RAG enhancement phase: {e}")
            logger.warning("Continuing pipeline without RAG enhancement...")
    
    def _run_rag_enhancement_custom(self, input_path: str, output_path: str, nuscenes_root: str) -> bool:
        """Run RAG enhancement with custom paths."""
        try:
            from data_analysis.rag_enhancer import enhance_unified_json_in_project
            
            import json
            import numpy as np
            from datetime import datetime
            from collections import defaultdict
            from data_analysis.rag_enhancer import RAGEnhancer, generate_statistics
            
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error("Expected a list of items in unified.json")
                return False
            
            logger.info(f"Processing {len(data)} items for RAG enhancement...")
            
            enhancer = RAGEnhancer(nuscenes_root)
            
            enhanced_data = []
            retrieval_indices = {
                'by_scene_token': {},
                'by_question_type': defaultdict(list),
                'by_semantic_intent': defaultdict(list),
                'by_entity_category': defaultdict(list),
                'by_driving_scenario': defaultdict(list),
                'by_complexity': defaultdict(list)
            }
            
            for i, item in enumerate(data):
                try:
                    enhanced_item = {
                        'id': f"qa_{i:06d}",
                        'question': enhancer.enhance_question(item.get('question', {})),
                        'scene_data': enhancer.enhance_scene_data(item.get('scene_data')),
                        'metadata': item.get('metadata', {})
                    }
                    
                    enhanced_item['metadata'].update({
                        'processing_timestamp': datetime.now().isoformat(),
                        'rag_optimized': True,
                        'enhancement_version': '2.0'
                    })
                    
                    question = enhanced_item['question']
                    scene = enhanced_item['scene_data']
                    
                    retrieval_features = {
                        'embedding_text': question.get('embedding_text', ''),
                        'search_keywords': list(set(
                            question.get('keywords', []) + 
                            question.get('extracted_entities', []) +
                            ([scene.get('driving_scenario')] if scene else []) +
                            (list(scene.get('spatial_summary', {}).get('objects_by_category', {}).keys()) if scene else [])
                        )),
                        'complexity_level': question.get('complexity_score', 1),
                        'requires_visual': bool(scene.get('image_paths') if scene else False),
                        'scene_complexity': len(scene.get('object_annotations', [])) if scene else 0,
                        'spatial_regions': list(scene.get('spatial_summary', {}).get('objects_by_position', {}).keys()) if scene else []
                    }
                    
                    enhanced_item['retrieval_features'] = retrieval_features
                    
                    item_id = enhanced_item['id']
                    retrieval_indices['by_scene_token'][question.get('scene_token', '')] = item_id
                    retrieval_indices['by_question_type'][question.get('question_type', 'unknown')].append(item_id)
                    retrieval_indices['by_semantic_intent'][question.get('semantic_intent', 'general')].append(item_id)
                    
                    for entity in question.get('extracted_entities', []):
                        retrieval_indices['by_entity_category'][entity].append(item_id)
                    
                    if scene:
                        scenario = scene.get('driving_scenario', 'unknown')
                        retrieval_indices['by_driving_scenario'][scenario].append(item_id)
                    
                    complexity = question.get('complexity_score', 1)
                    retrieval_indices['by_complexity'][f"level_{complexity}"].append(item_id)
                    
                    enhanced_data.append(enhanced_item)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Enhanced {i + 1}/{len(data)} items")
                    
                except Exception as e:
                    logger.error(f"Error enhancing item {i}: {e}")
                    continue
            
            output_data = {
                'metadata': {
                    'version': '2.0_rag_optimized',
                    'created_timestamp': datetime.now().isoformat(),
                    'total_items': len(enhanced_data),
                    'items_with_scene_data': sum(1 for item in enhanced_data if item['scene_data']),
                    'optimization_features': [
                        'semantic_enhancement',
                        'spatial_processing',
                        'retrieval_indices', 
                        'embedding_ready_text',
                        'full_image_paths',
                        'complexity_scoring'
                    ],
                    'rag_ready': True
                },
                'data': enhanced_data
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            indices_path = Path(output_path).with_name(f"{Path(output_path).stem}_indices.json")
            with open(indices_path, 'w', encoding='utf-8') as f:
                json.dump(dict(retrieval_indices), f, indent=2, ensure_ascii=False)
            
            stats = generate_statistics(enhanced_data, retrieval_indices)
            stats_path = Path(output_path).with_name(f"{Path(output_path).stem}_statistics.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in custom RAG enhancement: {e}")
            return False
    
    def generate_final_report(self) -> None:
        logger.info("=== GENERATING FINAL REPORT ===")
        
        try:
            findings_path = self.output_dir / "findings.md"
            
            analysis_results_path = self.output_dir / "analysis_results.json"
            with open(analysis_results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            findings_content = self._generate_findings_content(results)
            
            with open(findings_path, 'w', encoding='utf-8') as f:
                f.write(findings_content)
            
            logger.info(f"Final report generated: {findings_path}")
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
    
    def _generate_findings_content(self, results: dict) -> str:
        
        quality_metrics = results.get('quality_metrics', {})
        question_dist = results.get('question_type_distribution', {})
        scene_stats = results.get('scene_statistics', {})
        
        most_common = max(question_dist.items(), key=lambda x: x[1]) if question_dist else ("N/A", 0)
        least_common = min(question_dist.items(), key=lambda x: x[1]) if question_dist else ("N/A", 0)
        
        rag_enhanced_path = self.output_dir / "unified_rag_enhanced.json"
        rag_section = ""
        
        if rag_enhanced_path.exists():
            rag_section = """

## RAG Enhancement

The dataset has been enhanced with RAG optimization features:
- **Semantic Intent Detection**: Automatic classification of question intents
- **Entity Extraction**: Identification of driving-related entities
- **Spatial Processing**: Computed object relationships and positions  
- **Retrieval Indices**: Multiple indices for efficient question matching
- **Embedding-Ready Text**: Pre-formatted text for vector embeddings

Enhanced files available:
- `unified_rag_enhanced.json` - Main enhanced dataset
- `unified_rag_enhanced_indices.json` - Retrieval indices
- `unified_rag_enhanced_statistics.json` - Enhancement statistics

This RAG-optimized dataset is ready for building efficient retrieval-augmented generation systems.
"""
        
        findings_template = f"""# DriveLM Dataset Analysis Report - Results

## Executive Summary

This report presents a comprehensive analysis of the DriveLM dataset built on NuScenes v1.0-mini. Our analysis processed **{quality_metrics.get('total_questions', 'N/A')} questions** across **{scene_stats.get('total_scenes', 'N/A')} unique scenes**.

## Key Findings

### Dataset Scale
- **Total Questions**: {quality_metrics.get('total_questions', 'N/A')}
- **Questions with Scene Data**: {scene_stats.get('questions_with_scene_data', 'N/A')}
- **Unique Scenes**: {scene_stats.get('total_scenes', 'N/A')}
- **Average Objects per Scene**: {scene_stats.get('avg_objects_per_scene', 'N/A'):.1f}

### Question Type Distribution
- **Most Common Question Type**: {most_common[0]} ({most_common[1]} instances)
- **Least Common Question Type**: {least_common[0]} ({least_common[1]} instances)
- **Question Types Identified**: {len(question_dist)}

### Data Quality Assessment
- **Empty Questions**: {quality_metrics.get('empty_questions', 0)}
- **Empty Answers**: {quality_metrics.get('empty_answers', 0)}
- **Questions with Reasoning**: {quality_metrics.get('questions_with_reasoning', 0)}
- **Average Question Length**: {quality_metrics.get('avg_question_length', 0):.1f} words
- **Average Answer Length**: {quality_metrics.get('avg_answer_length', 0):.1f} words

## Question Type Analysis

The following question types were identified in the dataset:

"""
        
        for q_type, count in sorted(question_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / quality_metrics.get('total_questions', 1)) * 100
            findings_template += f"- **{q_type.replace('_', ' ').title()}**: {count} questions ({percentage:.1f}%)\n"
        
        findings_template += f"""

## Scene Complexity Analysis

Scene complexity distribution:
"""
        
        # Add scene complexity if available
        complexity_dist = scene_stats.get('scene_complexity_distribution', {})
        for complexity, count in complexity_dist.items():
            findings_template += f"- **{complexity.title()} Scenes**: {count}\n"
        
        findings_template += rag_section  # Add RAG section here
        
        findings_template += """

## Identified Patterns and Biases

### Potential Biases
Our analysis identified several potential biases in the dataset:

"""
        
        # Add bias information
        biases = results.get('patterns_and_anomalies', {}).get('potential_biases', {})
        for bias_type, description in biases.items():
            findings_template += f"- **{bias_type.replace('_', ' ').title()}**: {description}\n"
        
        findings_template += """

## Data Quality Issues

### Identified Anomalies
"""
        
        # Add anomalies
        anomalies = results.get('patterns_and_anomalies', {}).get('data_anomalies', [])
        for anomaly in anomalies[:5]:  # Top 5 anomalies
            findings_template += f"- {anomaly}\n"
        
        findings_template += """

## Recommendations for RAG System Development

### Data Preprocessing
1. **Address Missing Links**: Implement robust handling for questions without scene data
2. **Quality Filtering**: Remove or flag empty questions and answers
3. **Bias Mitigation**: Consider data augmentation for underrepresented question types

### Model Architecture
1. **Multi-modal Processing**: Design architecture to handle both visual and textual inputs
2. **Context Retrieval**: Implement effective retrieval for complex scenes with many objects
3. **Reasoning Chain Support**: Enable step-by-step reasoning for complex questions

### Evaluation Strategy
1. **Stratified Evaluation**: Test across all question types and complexity levels
2. **Multi-dimensional Metrics**: Beyond text metrics to include visual grounding accuracy
3. **Domain-specific Assessment**: Evaluation criteria specific to autonomous driving tasks

## Visualizations Generated

The following visualizations are available in the `visualizations/` directory:
- Question type distribution charts
- Object category analysis
- Scene complexity heatmaps
- Quality metrics dashboard
- Interactive analysis dashboard
- Sample question showcases

## Conclusion

The DriveLM dataset provides a solid foundation for multi-modal question-answering in autonomous driving contexts. While quality issues and biases exist, the rich multi-modal nature and realistic driving scenarios make it valuable for developing robust RAG systems. The identified patterns will guide our model development approach.

---

*Analysis completed on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return findings_template
    
    def run_complete_pipeline(self) -> None:
        """Execute the complete analysis pipeline."""
        total_start_time = time.time()
        
        try:
            logger.info("Starting DriveLM Dataset Analysis Pipeline")
            logger.info("=" * 60)
            
            # Phase 1: Data Parsing
            self.run_parsing_phase()
            
            # Phase 2: Statistical Analysis
            self.run_analysis_phase()
            
            # Phase 3: Visualization Generation
            self.run_visualization_phase()
            
            # Phase 4: RAG Enhancement (NEW!)
            self.run_rag_enhancement_phase()
            
            # Generate Final Report
            self.generate_final_report()
            
            # Pipeline completion
            total_time = time.time() - total_start_time
            logger.info("=" * 60)
            logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f} seconds")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("=" * 60)
            
            # Print final summary
            self._print_completion_summary()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _print_completion_summary(self) -> None:
        """Print completion summary with file locations."""
        print("\n" + "=" * 60)
        print("DRIVELM DATASET ANALYSIS COMPLETED")
        print("=" * 60)
        print(f"Output Directory: {self.output_dir}")
        print("\nGenerated Files:")
        print(f"   Unified Data: {self.output_dir}/unified.json")
        print(f"   Analysis Results: {self.output_dir}/analysis_results.json")
        print(f"   Summary Report: {self.output_dir}/analysis_summary.txt")
        print(f"   Findings Report: {self.output_dir}/findings.md")
        print(f"   Visualizations: {self.output_dir}/visualizations/")
        print(f"   Dashboard: {self.output_dir}/visualizations/index.html")
        
        # Check if RAG enhancement was successful
        rag_enhanced_path = self.output_dir / "unified_rag_enhanced.json"
        if rag_enhanced_path.exists():
            print("\n RAG Enhancement Files:")
            print(f"   Enhanced Data: {self.output_dir}/unified_rag_enhanced.json")
            print(f"   Retrieval Indices: {self.output_dir}/unified_rag_enhanced_indices.json")
            print(f"   RAG Statistics: {self.output_dir}/unified_rag_enhanced_statistics.json")
        
        print("\nNext Steps:")
        print("  1. Review the findings report for key insights")
        print("  2. Open the interactive dashboard in your browser")
        if rag_enhanced_path.exists():
            print("  3. Use the RAG-enhanced data for building your retrieval system")
        else:
            print("  3. Use the unified data for RAG system development")
        print("=" * 60)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="DriveLM Dataset Analysis Pipeline with RAG Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --nuscenes_path /data/nuscenes --drivelm_path /data/v1_0_train_nus.json
  python main.py --nuscenes_path ./nuscenes --drivelm_path ./drivelm.json --output_dir ./results
  python main.py --nuscenes_path ./nuscenes --drivelm_path ./drivelm.json --no-rag-enhancement
        """
    )
    
    parser.add_argument(
        "--nuscenes_path",
        type=str,
        required=True,
        help="Path to NuScenes dataset root directory"
    )
    
    parser.add_argument(
        "--drivelm_path", 
        type=str,
        required=True,
        help="Path to DriveLM JSON file (e.g., v1_0_train_nus.json)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for analysis results (default: output)"
    )
    
    parser.add_argument(
        "--no-rag-enhancement",
        action="store_true",
        help="Skip RAG enhancement phase"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        pipeline = AnalysisPipeline(
            nuscenes_path=args.nuscenes_path,
            drivelm_path=args.drivelm_path,
            output_dir=args.output_dir,
            enable_rag_enhancement=not args.no_rag_enhancement
        )
        
        pipeline.run_complete_pipeline()
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()