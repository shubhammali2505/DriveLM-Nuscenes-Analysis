"""
Visualization Module for NuScenes and DriveLM Data Analysis

This module provides comprehensive visualizations including charts, 
dashboards, and interactive plots for the dataset analysis.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import logging
from PIL import Image
import io
import base64
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
logger = logging.getLogger(__name__)


class DatasetVisualizer:
    
    def __init__(
        self, 
        analysis_results_path: str, 
        unified_data_path: str
    ):

        self.analysis_results_path = Path(analysis_results_path)
        self.unified_data_path = Path(unified_data_path)
        self.analysis_results = self._load_analysis_results()
        self.unified_data = self._load_unified_data()
        
        self.output_dir = Path("visualizations")
        
        self._ensure_output_directory()
    
    def _ensure_output_directory(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory created/verified: {self.output_dir}")
            
            test_file = self.output_dir / ".test_write"
            test_file.write_text("test")
            test_file.unlink()  
            
        except Exception as e:
            logger.error(f"Error creating output directory {self.output_dir}: {e}")
            self.output_dir = Path.cwd() / "visualizations_fallback"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Using fallback directory: {self.output_dir}")
    
    def _load_analysis_results(self) -> Dict[str, Any]:
        try:
            if not self.analysis_results_path.exists():
                logger.error(f"Analysis results file not found: {self.analysis_results_path}")
                return {}
                
            with open(self.analysis_results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                logger.info(f"Loaded analysis results from {self.analysis_results_path}")
                return results
        except Exception as e:
            logger.error(f"Error loading analysis results: {e}")
            return {}
    
    def _load_unified_data(self) -> List[Dict[str, Any]]:
        try:
            if not self.unified_data_path.exists():
                logger.error(f"Unified data file not found: {self.unified_data_path}")
                return []
                
            with open(self.unified_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, dict) and 'data' in data:
                    unified_data = data['data']
                    logger.info(f"Loaded {len(unified_data)} items from enhanced format")
                elif isinstance(data, list):
                    unified_data = data
                    logger.info(f"Loaded {len(unified_data)} items from list format")
                else:
                    logger.error(f"Unexpected data format in {self.unified_data_path}")
                    return []
                
                return unified_data
                
        except Exception as e:
            logger.error(f"Error loading unified data: {e}")
            return []
    
    def create_question_type_distribution_chart(self) -> None:
        try:
            question_dist = self.analysis_results.get('question_type_distribution', {})
            if not question_dist:
                logger.warning("No question type distribution data available")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            categories = list(question_dist.keys())
            counts = list(question_dist.values())
            
            bars = ax1.bar(categories, counts, color=sns.color_palette("husl", len(categories)))
            ax1.set_title('Question Type Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Question Type')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width()/2., 
                    height,
                    f'{count}', 
                    ha='center', 
                    va='bottom'
                )
            
            ax2.pie(
                counts, 
                labels=categories, 
                autopct='%1.1f%%', 
                startangle=90
            )
            ax2.set_title(
                'Question Type Distribution (Percentage)', 
                fontsize=14, 
                fontweight='bold'
            )
            
            plt.tight_layout()
            
            output_path = self.output_dir / 'question_type_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            fig_plotly = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Question Type Counts', 'Question Type Percentage'),
                specs=[[{"type": "bar"}, {"type": "pie"}]]
            )
            
            fig_plotly.add_trace(
                go.Bar(x=categories, y=counts, name="Count",
                      marker=dict(color=px.colors.qualitative.Set3[:len(categories)])),
                row=1, col=1
            )
            
            fig_plotly.add_trace(
                go.Pie(labels=categories, values=counts, name="Distribution"),
                row=1, col=2
            )
            
            fig_plotly.update_layout(
                title_text="Question Type Distribution Analysis",
                height=500,
                showlegend=False
            )
            
            interactive_path = self.output_dir / 'question_type_distribution_interactive.html'
            fig_plotly.write_html(str(interactive_path))
            
            logger.info("Question type distribution charts created successfully")
            
        except Exception as e:
            logger.error(f"Error creating question type distribution chart: {e}")
    
    def create_object_category_analysis(self) -> None:
        try:
            object_dist = self.analysis_results.get('object_category_distribution', {})
            if not object_dist:
                logger.warning("No object category distribution data available")
                return
            
            sorted_objects = sorted(object_dist.items(), key=lambda x: x[1], reverse=True)
            top_objects = sorted_objects[:15]
            
            categories, counts = zip(*top_objects)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            bars = ax.barh(categories, counts, color=sns.color_palette("viridis", len(categories)))
            ax.set_title('Top 15 Object Categories in Scenes', fontsize=14, fontweight='bold')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Object Category')
            
            for bar, count in zip(bars, counts):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{count}', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            
            output_path = self.output_dir / 'object_category_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Interactive version
            fig_plotly = px.bar(
                x=counts, y=categories,
                orientation='h',
                title='Object Category Distribution in Scenes',
                labels={'x': 'Frequency', 'y': 'Object Category'},
                color=counts,
                color_continuous_scale='viridis'
            )
            
            fig_plotly.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            
            interactive_path = self.output_dir / 'object_category_distribution_interactive.html'
            fig_plotly.write_html(str(interactive_path))
            
            logger.info("Object category distribution charts created successfully")
            
        except Exception as e:
            logger.error(f"Error creating object category analysis: {e}")
    
    def create_scene_complexity_analysis(self) -> None:
        try:
            scene_stats = self.analysis_results.get('scene_statistics', {})
            complexity_dist = scene_stats.get('scene_complexity_distribution', {})
            
            if not complexity_dist:
                logger.warning("No scene complexity data available")
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            labels = list(complexity_dist.keys())
            sizes = list(complexity_dist.values())
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('Scene Complexity Distribution', fontweight='bold')
            
            stats_to_plot = {
                'Total Scenes': scene_stats.get('total_scenes', 0),
                'Total Samples': scene_stats.get('total_samples', 0),
                'With Scene Data': scene_stats.get('questions_with_scene_data', 0),
                'Without Scene Data': scene_stats.get('questions_without_scene_data', 0)
            }
            
            ax2.bar(stats_to_plot.keys(), stats_to_plot.values(), 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax2.set_title('Dataset Statistics', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            for i, (key, value) in enumerate(stats_to_plot.items()):
                ax2.text(i, value, str(value), ha='center', va='bottom', fontweight='bold')
            
            objects_per_scene = []
            for item in self.unified_data:
                if item.get('metadata', {}).get('has_scene_data', False):
                    objects_per_scene.append(item['metadata'].get('num_objects', 0))
            
            if objects_per_scene:
                ax3.hist(objects_per_scene, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.set_title('Distribution of Objects per Scene', fontweight='bold')
                ax3.set_xlabel('Number of Objects')
                ax3.set_ylabel('Frequency')
                ax3.axvline(np.mean(objects_per_scene), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(objects_per_scene):.1f}')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No scene data available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Distribution of Objects per Scene', fontweight='bold')
            
            quality_metrics = self.analysis_results.get('quality_metrics', {})
            quality_to_plot = {
                'Empty Questions': quality_metrics.get('empty_questions', 0),
                'Empty Answers': quality_metrics.get('empty_answers', 0),
                'Missing Scene Tokens': quality_metrics.get('missing_scene_tokens', 0),
                'Questions with Reasoning': quality_metrics.get('questions_with_reasoning', 0)
            }
            
            ax4.bar(quality_to_plot.keys(), quality_to_plot.values(), 
                   color=['#ff4444', '#ff8844', '#ffcc44', '#44ff44'])
            ax4.set_title('Data Quality Metrics', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            
            for i, (key, value) in enumerate(quality_to_plot.items()):
                ax4.text(i, value, str(value), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            output_path = self.output_dir / 'scene_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Scene complexity analysis charts created successfully")
            
        except Exception as e:
            logger.error(f"Error creating scene complexity analysis: {e}")
    
    def create_question_answer_correlation_analysis(self) -> None:
        try:
            question_lengths = []
            answer_lengths = []
            question_types = []
            
            for item in self.unified_data:
                question_data = item.get('question', {})
                q_text = question_data.get('question', '')
                a_text = question_data.get('answer', '')
                q_type = question_data.get('question_type', 'unknown')
                
                question_lengths.append(len(q_text.split()) if q_text else 0)
                answer_lengths.append(len(a_text.split()) if a_text else 0)
                question_types.append(q_type)
            
            if not question_lengths:
                logger.warning("No question data available for correlation analysis")
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            ax1.scatter(question_lengths, answer_lengths, alpha=0.6, c='blue')
            ax1.set_xlabel('Question Length (words)')
            ax1.set_ylabel('Answer Length (words)')
            ax1.set_title('Question vs Answer Length Correlation', fontweight='bold')
            
            # Add  coefficient
            if len(question_lengths) > 1:
                correlation = np.corrcoef(question_lengths, answer_lengths)[0, 1]
                ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=ax1.transAxes, fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            df = pd.DataFrame({
                'question_length': question_lengths,
                'answer_length': answer_lengths,
                'question_type': question_types
            })
            
            question_type_lengths = df.groupby('question_type')['question_length'].mean().sort_values()
            ax2.barh(question_type_lengths.index, question_type_lengths.values, color='lightcoral')
            ax2.set_xlabel('Average Question Length (words)')
            ax2.set_title('Average Question Length by Type', fontweight='bold')
            
            answer_type_lengths = df.groupby('question_type')['answer_length'].mean().sort_values()
            ax3.barh(answer_type_lengths.index, answer_type_lengths.values, color='lightblue')
            ax3.set_xlabel('Average Answer Length (words)')
            ax3.set_title('Average Answer Length by Type', fontweight='bold')
            
            ax4.hist(question_lengths, bins=30, alpha=0.5, label='Questions', color='red')
            ax4.hist(answer_lengths, bins=30, alpha=0.5, label='Answers', color='blue')
            ax4.set_xlabel('Length (words)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Question and Answer Length Distributions', fontweight='bold')
            ax4.legend()
            
            plt.tight_layout()
            
            output_path = self.output_dir / 'question_answer_correlation.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Question-answer correlation analysis created successfully")
            
        except Exception as e:
            logger.error(f"Error creating question-answer correlation analysis: {e}")
    
    def create_pattern_analysis_heatmap(self) -> None:
        try:
            patterns = self.analysis_results.get('patterns_and_anomalies', {})
            scene_patterns = patterns.get('scene_question_patterns', {})
            
            if not scene_patterns:
                logger.warning("No scene-question patterns available")
                self._create_alternative_pattern_visualization()
                return
            
            all_question_types = set()
            for complexity_patterns in scene_patterns.values():
                all_question_types.update(complexity_patterns.keys())
            
            all_question_types = sorted(list(all_question_types))
            complexities = sorted(scene_patterns.keys())
            
            heatmap_data = np.zeros((len(complexities), len(all_question_types)))
            
            for i, complexity in enumerate(complexities):
                for j, q_type in enumerate(all_question_types):
                    heatmap_data[i, j] = scene_patterns[complexity].get(q_type, 0)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            sns.heatmap(heatmap_data, 
                       xticklabels=all_question_types,
                       yticklabels=complexities,
                       annot=True, 
                       fmt='g',
                       cmap='YlOrRd',
                       ax=ax)
            
            ax.set_title('Question Types vs Scene Complexity Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Question Type')
            ax.set_ylabel('Scene Complexity')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            output_path = self.output_dir / 'pattern_analysis_heatmap.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Pattern analysis heatmap created successfully")
            
        except Exception as e:
            logger.error(f"Error creating pattern analysis heatmap: {e}")
    
    def _create_alternative_pattern_visualization(self) -> None:
        """Create alternative pattern visualization when heatmap data is not available."""
        try:
            question_dist = self.analysis_results.get('question_type_distribution', {})
            if not question_dist:
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            categories = list(question_dist.keys())
            counts = list(question_dist.values())
            
            bars = ax.bar(categories, counts, color=sns.color_palette("Set2", len(categories)))
            ax.set_title('Question Type Frequency Analysis', fontsize=14, fontweight='bold')
            ax.set_xlabel('Question Type')
            ax.set_ylabel('Frequency')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            output_path = self.output_dir / 'pattern_analysis_heatmap.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Alternative pattern analysis chart created")
            
        except Exception as e:
            logger.error(f"Error creating alternative pattern visualization: {e}")
    
    def create_interactive_dashboard(self) -> None:
        try:
            question_dist = self.analysis_results.get('question_type_distribution', {})
            object_dist = self.analysis_results.get('object_category_distribution', {})
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Question Type Distribution', 'Top Object Categories',
                              'Scene Complexity', 'Data Quality Overview'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "pie"}, {"type": "bar"}]]
            )
            
            if question_dist:
                fig.add_trace(
                    go.Pie(labels=list(question_dist.keys()), 
                          values=list(question_dist.values()),
                          name="Question Types"),
                    row=1, col=1
                )
            
            if object_dist:
                top_objects = sorted(object_dist.items(), key=lambda x: x[1], reverse=True)[:10]
                categories, counts = zip(*top_objects)
                fig.add_trace(
                    go.Bar(x=list(categories), y=list(counts), name="Objects"),
                    row=1, col=2
                )
            
            scene_stats = self.analysis_results.get('scene_statistics', {})
            complexity_dist = scene_stats.get('scene_complexity_distribution', {})
            if complexity_dist:
                fig.add_trace(
                    go.Pie(labels=list(complexity_dist.keys()),
                          values=list(complexity_dist.values()),
                          name="Complexity"),
                    row=2, col=1
                )
            
            quality_metrics = self.analysis_results.get('quality_metrics', {})
            quality_items = ['total_questions', 'questions_with_reasoning', 
                            'empty_questions', 'empty_answers']
            quality_values = [quality_metrics.get(item, 0) for item in quality_items]
            quality_labels = ['Total Questions', 'With Reasoning', 'Empty Questions', 'Empty Answers']
            
            fig.add_trace(
                go.Bar(x=quality_labels, y=quality_values, name="Quality"),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="DriveLM Dataset Analysis Dashboard",
                height=800,
                showlegend=False
            )
            
            output_path = self.output_dir / 'interactive_dashboard.html'
            fig.write_html(str(output_path))
            
            logger.info("Interactive dashboard created successfully")
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
    
    def create_sample_question_showcase(self, num_samples: int = 5) -> None:
        try:
            if not self.unified_data:
                logger.warning("No unified data available for sample showcase")
                return
            
            question_types = {}
            for item in self.unified_data:
                question_data = item.get('question', {})
                q_type = question_data.get('question_type', 'unknown')
                if q_type not in question_types:
                    question_types[q_type] = []
                question_types[q_type].append(item)
            
            selected_samples = []
            for q_type, questions in question_types.items():
                if len(selected_samples) >= num_samples:
                    break
                best_question = max(questions, 
                                  key=lambda x: x.get('metadata', {}).get('num_objects', 0))
                selected_samples.append(best_question)
            
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Sample Questions Showcase</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .sample { border: 1px solid #ccc; margin: 20px 0; padding: 15px; border-radius: 5px; }
                    .question { font-weight: bold; color: #2c3e50; }
                    .answer { color: #27ae60; margin: 10px 0; }
                    .metadata { color: #7f8c8d; font-size: 0.9em; }
                    .reasoning { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 3px solid #007bff; }
                </style>
            </head>
            <body>
                <h1>DriveLM Sample Questions Showcase</h1>
            """
            
            for i, sample in enumerate(selected_samples):
                question_data = sample.get('question', {})
                metadata = sample.get('metadata', {})
                
                html_content += f"""
                <div class="sample">
                    <h3>Sample {i+1}: {question_data.get('question_type', 'Unknown').replace('_', ' ').title()}</h3>
                    <div class="question">Q: {question_data.get('question', 'No question available')}</div>
                    <div class="answer">A: {question_data.get('answer', 'No answer available')}</div>
                    <div class="metadata">
                        Scene Token: {question_data.get('scene_token', 'N/A')}<br>
                        Objects in Scene: {metadata.get('num_objects', 0)}<br>
                        Cameras: {metadata.get('num_cameras', 0)}<br>
                        Has Scene Data: {metadata.get('has_scene_data', False)}
                    </div>
                """
                
                reasoning_chain = question_data.get('reasoning_chain', [])
                if reasoning_chain:
                    html_content += '<div class="reasoning"><strong>Reasoning Chain:</strong><ul>'
                    for step in reasoning_chain:
                        html_content += f'<li>{step}</li>'
                    html_content += '</ul></div>'
                
                html_content += '</div>'
            
            html_content += """
            </body>
            </html>
            """
            
            output_path = self.output_dir / 'sample_questions_showcase.html'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info("Sample questions showcase created successfully")
            
        except Exception as e:
            logger.error(f"Error creating sample questions showcase: {e}")
    
    def generate_complete_visualization_suite(self) -> None:
        """Generate complete set of visualizations."""
        logger.info("Generating complete visualization suite...")
        
        self._ensure_output_directory()
        
        visualization_methods = [
            ("Question Type Distribution", self.create_question_type_distribution_chart),
            ("Object Category Analysis", self.create_object_category_analysis),
            ("Scene Complexity Analysis", self.create_scene_complexity_analysis),
            ("Question-Answer Correlation", self.create_question_answer_correlation_analysis),
            ("Pattern Analysis Heatmap", self.create_pattern_analysis_heatmap),
            ("Interactive Dashboard", self.create_interactive_dashboard),
            ("Sample Questions Showcase", self.create_sample_question_showcase)
        ]
        
        successful_visualizations = []
        failed_visualizations = []
        
        for viz_name, viz_method in visualization_methods:
            try:
                logger.info(f"Creating {viz_name}...")
                viz_method()
                successful_visualizations.append(viz_name)
            except Exception as e:
                logger.error(f"Failed to create {viz_name}: {e}")
                failed_visualizations.append(viz_name)
        
        try:
            self._create_visualization_index()
            successful_visualizations.append("Visualization Index")
        except Exception as e:
            logger.error(f"Failed to create visualization index: {e}")
            failed_visualizations.append("Visualization Index")
        
        logger.info(f"Visualization generation complete!")
        logger.info(f"Successfully created: {len(successful_visualizations)} visualizations")
        if failed_visualizations:
            logger.warning(f"Failed to create: {len(failed_visualizations)} visualizations")
            logger.warning(f"Failed visualizations: {', '.join(failed_visualizations)}")
        
        logger.info(f"All visualizations saved to {self.output_dir}")
    
    def _create_visualization_index(self) -> None:
        """Create index HTML file linking all visualizations."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DriveLM Dataset Analysis Visualizations</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .viz-section { margin: 30px 0; }
                .viz-link { display: block; margin: 10px 0; padding: 10px; 
                           background-color: #f8f9fa; border-radius: 5px; text-decoration: none; }
                .viz-link:hover { background-color: #e9ecef; }
                img { max-width: 100%; height: auto; margin: 10px 0; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .success { background-color: #d4edda; color: #155724; }
                .warning { background-color: #fff3cd; color: #856404; }
            </style>
        </head>
        <body>
            <h1>DriveLM Dataset Analysis Visualizations</h1>
            
            <div class="status success">
                <strong>Status:</strong> Visualization suite generated successfully!
            </div>
            
            <div class="viz-section">
                <h2>Static Charts</h2>
                <p>These PNG charts provide detailed statistical analysis:</p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
                    <div>
                        <h4>Question Type Distribution</h4>
                        <img src="question_type_distribution.png" alt="Question Type Distribution" style="border: 1px solid #ddd;">
                    </div>
                    <div>
                        <h4>Object Category Distribution</h4>
                        <img src="object_category_distribution.png" alt="Object Category Distribution" style="border: 1px solid #ddd;">
                    </div>
                    <div>
                        <h4>Scene Analysis</h4>
                        <img src="scene_analysis.png" alt="Scene Analysis" style="border: 1px solid #ddd;">
                    </div>
                    <div>
                        <h4>Question-Answer Correlation</h4>
                        <img src="question_answer_correlation.png" alt="Question Answer Correlation" style="border: 1px solid #ddd;">
                    </div>
                    <div>
                        <h4>Pattern Analysis</h4>
                        <img src="pattern_analysis_heatmap.png" alt="Pattern Analysis Heatmap" style="border: 1px solid #ddd;">
                    </div>
                </div>
            </div>
            
            <div class="viz-section">
                <h2>Interactive Visualizations</h2>
                <p>These interactive charts allow for detailed exploration:</p>
                <a href="interactive_dashboard.html" class="viz-link">📊 <strong>Interactive Dashboard</strong> - Main overview with multiple charts</a>
                <a href="question_type_distribution_interactive.html" class="viz-link">📈 <strong>Interactive Question Type Analysis</strong> - Detailed question type exploration</a>
                <a href="object_category_distribution_interactive.html" class="viz-link">📊 <strong>Interactive Object Category Analysis</strong> - Object frequency analysis</a>
                <a href="sample_questions_showcase.html" class="viz-link">🔍 <strong>Sample Questions Showcase</strong> - Example questions and answers</a>
            </div>
            
            <div class="viz-section">
                <h2>Analysis Summary</h2>
                <p>This visualization suite provides comprehensive insights into the DriveLM dataset:</p>
                <ul>
                    <li><strong>Question Distribution:</strong> Shows the types of questions and their frequency</li>
                    <li><strong>Object Analysis:</strong> Reveals the most common objects in driving scenes</li>
                    <li><strong>Scene Complexity:</strong> Analyzes the complexity distribution of scenes</li>
                    <li><strong>Quality Metrics:</strong> Highlights data quality issues and patterns</li>
                    <li><strong>Correlations:</strong> Shows relationships between different data aspects</li>
                </ul>
                
                <h3>Key Findings</h3>
                <div class="status warning">
                    <strong>Note:</strong> Detailed findings are available in the main analysis report (findings.md)
                </div>
            </div>
            
            <div class="viz-section">
                <h2>Technical Notes</h2>
                <ul>
                    <li>All charts are saved in high resolution (300 DPI) for publication quality</li>
                    <li>Interactive charts use Plotly for dynamic exploration</li>
                    <li>Static charts use matplotlib and seaborn for publication-ready figures</li>
                    <li>All visualizations are self-contained and can be shared independently</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        output_path = self.output_dir / 'index.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Visualization index created at {output_path}")


def main():
    analysis_results_path = "analysis_results.json"
    unified_data_path = "unified.json" 
    visualizer = DatasetVisualizer(analysis_results_path, unified_data_path)
    
    visualizer.generate_complete_visualization_suite()
    
    print("Visualization suite generated successfully!")
    print("Open 'visualizations/index.html' to view all charts and dashboards.")


if __name__ == "__main__":
    main()