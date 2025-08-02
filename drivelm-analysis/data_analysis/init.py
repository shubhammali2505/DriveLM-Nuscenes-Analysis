"""
DriveLM Data Analysis Package

This package provides comprehensive tools for analyzing the DriveLM dataset
built on NuScenes v1.0-mini, including data parsing, statistical analysis,
and visualization generation.

Modules:
    parsers: NuScenes and DriveLM data parsers
    analysis: Statistical analysis and pattern recognition
    visualizations: Chart generation and dashboard creation
"""

__version__ = "1.0.0"
__author__ = "DriveLM Analysis Team"
__email__ = "contact@example.com"

# Import main classes for easy access
from .parsers import (
    NuScenesParser,
    DriveLMParser,
    UnifiedDataStructure,
    EgoVehicleState,
    ObjectAnnotation,
    CameraData,
    SceneData,
    DriveLMQuestion,
)

from .analysis import (
    DatasetAnalyzer,
    AnalysisResults,
)

from .visualizations import (
    DatasetVisualizer,
)

__all__ = [
    # Parsers
    "NuScenesParser",
    "DriveLMParser", 
    "UnifiedDataStructure",
    "EgoVehicleState",
    "ObjectAnnotation",
    "CameraData",
    "SceneData",
    "DriveLMQuestion",
    # Analysis
    "DatasetAnalyzer",
    "AnalysisResults",
    # Visualizations
    "DatasetVisualizer",
]
