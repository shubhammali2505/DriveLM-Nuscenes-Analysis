"""
Optimized RAG Enhancer - Focused on QA Retrieval with Full Camera Coverage

This script enhances your unified.json with essential RAG features:
- All 6 camera image paths (CAM_FRONT, CAM_BACK, CAM_FRONT_LEFT, etc.)
- Rich metadata for accurate QA retrieval
- Clean structure optimized for LLM generation
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import logging
from collections import defaultdict, Counter
import re
from datetime import datetime
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedRAGEnhancer:
    
    def __init__(self, nuscenes_root: str = None):
        self.nuscenes_root = Path(nuscenes_root) if nuscenes_root else None
        
        self.camera_channels = [
            'CAM_FRONT', 'CAM_BACK', 
            'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        self.driving_entities = {
            'vehicles': ['car', 'truck', 'bus', 'motorcycle', 'vehicle', 'van', 'trailer'],
            'people': ['person', 'pedestrian', 'cyclist', 'human', 'adult', 'child'],
            'directions': ['front', 'behind', 'left', 'right', 'ahead', 'side'],
            'actions': ['turning', 'stopping', 'moving', 'braking', 'parked', 'crossing'],
            'infrastructure': ['traffic light', 'sign', 'road', 'lane', 'intersection'],
            'conditions': ['rain', 'sunny', 'night', 'day', 'clear']
        }
    
    def enhance_qa_pair(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance QA pair with retrieval-optimized features."""
        question_text = question_data.get('question', '').lower()
        answer_text = question_data.get('answer', '').lower()
        
        entities = self._extract_entities(question_text + ' ' + answer_text)
        keywords = self._extract_keywords(question_text + ' ' + answer_text)
        
        question_intent = self._classify_question_intent(question_text)
        answer_type = self._classify_answer_type(question_text)
        
        search_text = self._create_search_text(question_data, entities, keywords)
        
        return {
            'question_id': question_data.get('question_id', ''),
            'scene_token': question_data.get('scene_token', ''),
            'sample_token': question_data.get('sample_token', ''),
            'question': question_data.get('question', ''),
            'answer': question_data.get('answer', ''),
            'question_type': question_data.get('question_type', ''),
            'reasoning_chain': question_data.get('reasoning_chain', []),
            
            # Retrieval features
            'search_text': search_text,
            'entities': entities,
            'keywords': keywords,
            'question_intent': question_intent,
            'answer_type': answer_type,
            'complexity': self._calculate_complexity(question_text, answer_text),
            
            # Capability flags
            'requires_counting': 'how many' in question_text or 'count' in question_text,
            'requires_spatial': any(word in question_text for word in ['where', 'position', 'front', 'behind', 'left', 'right']),
            'requires_identification': any(word in question_text for word in ['what', 'which', 'identify']),
            'is_safety_related': any(word in question_text for word in ['safe', 'danger', 'collision', 'brake']),
            'is_yes_no': answer_type == 'yes_no'
        }
    
    def enhance_scene_metadata(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract essential scene metadata for retrieval context."""
        if not scene_data:
            return None
        
        ego_state = scene_data.get('ego_vehicle_state', {})
        ego_info = {
            'location': ego_state.get('location', [0, 0, 0]),
            'speed_ms': ego_state.get('speed', 0),
            'speed_kmh': round(ego_state.get('speed', 0) * 3.6, 1) if ego_state.get('speed') else 0,
            'is_moving': ego_state.get('speed', 0) > 0.5,
            'timestamp': ego_state.get('timestamp', 0)
        }
        
        objects = scene_data.get('object_annotations', [])
        object_summary = self._analyze_objects(objects, ego_info['location'])
        
        scene_context = {
            'scene_token': scene_data.get('scene_token', ''),
            'sample_token': scene_data.get('sample_token', ''),
            'timestamp': scene_data.get('timestamp', 0),
            'description': scene_data.get('description', ''),
            'total_objects': len(objects),
            'traffic_density': self._assess_traffic_density(objects),
            'driving_scenario': self._classify_scenario(ego_info, objects)
        }
        
        return {
            'ego_vehicle': ego_info,
            'objects': object_summary,
            'scene_context': scene_context
        }
    
    def create_full_image_paths(self, camera_data: Dict[str, Any]) -> Dict[str, Any]:
        image_info = {
            'available_cameras': [],
            'image_paths': {},
            'camera_timestamps': {},
            'missing_cameras': []
        }
        
        for channel in self.camera_channels:
            if channel in camera_data:
                cam_data = camera_data[channel]
                filename = cam_data.get('filename', '')
                timestamp = cam_data.get('timestamp', 0)
                
                if self.nuscenes_root and filename:
                    full_path = self.nuscenes_root / filename
                    image_info['image_paths'][channel] = str(full_path)
                    image_info['available_cameras'].append(channel)
                    image_info['camera_timestamps'][channel] = timestamp
                elif filename:
                    image_info['image_paths'][channel] = filename
                    image_info['available_cameras'].append(channel)
                    image_info['camera_timestamps'][channel] = timestamp
            else:
                image_info['missing_cameras'].append(channel)
        
        image_info['has_front_camera'] = 'CAM_FRONT' in image_info['available_cameras']
        image_info['has_all_cameras'] = len(image_info['available_cameras']) == 6
        image_info['camera_count'] = len(image_info['available_cameras'])
        image_info['primary_image'] = image_info['image_paths'].get('CAM_FRONT', 
                                     next(iter(image_info['image_paths'].values()), None))
        
        return image_info
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract driving-related entities from text."""
        entities = set()
        for category, entity_list in self.driving_entities.items():
            for entity in entity_list:
                if entity in text:
                    entities.add(entity)
        return sorted(list(entities))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords, removing stop words."""
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'in', 'at', 'on', 'for', 'to', 'of', 
            'and', 'or', 'but', 'with', 'by', 'from', 'this', 'that', 'there'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]
    
    def _classify_question_intent(self, question: str) -> str:
        """Classify the main intent of the question."""
        if any(word in question for word in ['what', 'which', 'identify']):
            return 'identification'
        elif any(word in question for word in ['where', 'position', 'location']):
            return 'localization'
        elif any(phrase in question for phrase in ['how many', 'count', 'number']):
            return 'counting'
        elif any(word in question for word in ['moving', 'speed', 'direction']):
            return 'motion'
        elif any(word in question for word in ['safe', 'danger', 'collision']):
            return 'safety'
        elif any(word in question for word in ['should', 'action', 'plan']):
            return 'planning'
        elif any(word in question for word in ['is', 'are', 'can', 'will']):
            return 'verification'
        else:
            return 'description'
    
    def _classify_answer_type(self, question: str) -> str:
        """Classify expected answer type."""
        if any(word in question for word in ['is', 'are', 'can', 'will', 'does']):
            return 'yes_no'
        elif any(phrase in question for phrase in ['how many', 'count']):
            return 'numeric'
        elif any(word in question for word in ['what', 'which']):
            return 'categorical'
        elif any(word in question for word in ['where']):
            return 'spatial'
        else:
            return 'descriptive'
    
    def _calculate_complexity(self, question: str, answer: str) -> int:
        """Calculate question complexity (1-3 scale)."""
        complexity = 1
        
        entity_count = sum(len([e for e in entities if e in question or e in answer]) 
                          for entities in self.driving_entities.values())
        if entity_count > 3:
            complexity += 1
        
        if any(word in question for word in ['and', 'both', 'while', 'also']):
            complexity += 1
        
        return min(complexity, 3)
    
    def _create_search_text(self, question_data: Dict, entities: List[str], keywords: List[str]) -> str:
        """Create optimized text for embedding-based retrieval."""
        components = [
            question_data.get('question', ''),
            question_data.get('answer', ''),
            f"Type: {question_data.get('question_type', '')}",
            f"Entities: {' '.join(entities[:5])}" if entities else "",
            f"Keywords: {' '.join(keywords[:8])}" if keywords else ""
        ]
        
        return " | ".join([comp for comp in components if comp])
    
    def _analyze_objects(self, objects: List[Dict], ego_location: List[float]) -> Dict[str, Any]:
        """Analyze objects in the scene for context."""
        if not objects:
            return {'count': 0, 'categories': {}, 'nearby_objects': []}
        
        categories = Counter()
        nearby_objects = []
        moving_objects = 0
        
        ego_x, ego_y = ego_location[0], ego_location[1] if len(ego_location) >= 2 else (0, 0)
        
        for obj in objects:
            # Count by category
            category = obj.get('category_name', 'unknown')
            categories[category] += 1
            
            # Check if moving
            if obj.get('speed', 0) > 0.5:
                moving_objects += 1
            
            # Check if nearby (within 20m)
            if obj.get('translation') and len(obj['translation']) >= 2:
                obj_x, obj_y = obj['translation'][0], obj['translation'][1]
                distance = np.sqrt((obj_x - ego_x)**2 + (obj_y - ego_y)**2)
                if distance < 20:
                    nearby_objects.append({
                        'category': category,
                        'distance': round(distance, 1),
                        'moving': obj.get('speed', 0) > 0.5
                    })
        
        return {
            'count': len(objects),
            'categories': dict(categories),
            'moving_count': moving_objects,
            'nearby_objects': nearby_objects[:10],  # Limit to 10 nearest
            'vehicle_count': sum(1 for obj in objects if 'vehicle' in obj.get('category_name', '').lower() or 'car' in obj.get('category_name', '').lower()),
            'pedestrian_count': sum(1 for obj in objects if 'pedestrian' in obj.get('category_name', '').lower() or 'person' in obj.get('category_name', '').lower())
        }
    
    def _assess_traffic_density(self, objects: List[Dict]) -> str:
        """Assess traffic density based on vehicle count."""
        vehicle_count = sum(1 for obj in objects 
                           if any(keyword in obj.get('category_name', '').lower() 
                                 for keyword in ['car', 'truck', 'bus', 'vehicle']))
        
        if vehicle_count > 8:
            return 'heavy'
        elif vehicle_count > 3:
            return 'moderate'
        else:
            return 'light'
    
    def _classify_scenario(self, ego_info: Dict, objects: List[Dict]) -> str:
        """Classify driving scenario."""
        speed = ego_info.get('speed_ms', 0)
        vehicle_count = sum(1 for obj in objects 
                           if any(keyword in obj.get('category_name', '').lower() 
                                 for keyword in ['car', 'truck', 'bus', 'vehicle']))
        pedestrian_count = sum(1 for obj in objects 
                              if 'pedestrian' in obj.get('category_name', '').lower())
        
        if speed > 20:
            return 'highway'
        elif vehicle_count > 6:
            return 'intersection'
        elif pedestrian_count > 2:
            return 'urban'
        elif speed < 2:
            return 'parking'
        else:
            return 'suburban'


def enhance_unified_json_for_rag(
    input_path: str = None,
    output_path: str = None,
    nuscenes_root: str = None
) -> bool:

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if input_path is None:
        input_path = project_root / "output" / "unified.json"
    if output_path is None:
        output_path = project_root / "output" / "unified_rag_optimized.json"
    if nuscenes_root is None:
        nuscenes_root = project_root.parent / "data" / "nuscenes"
    
    logger.info(f"Processing: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"NuScenes root: {nuscenes_root}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error("Expected a list of items in unified.json")
            return False
        
        logger.info(f"Processing {len(data)} QA pairs...")
        
        enhancer = OptimizedRAGEnhancer(str(nuscenes_root) if Path(nuscenes_root).exists() else None)
        
        enhanced_data = []
        retrieval_index = {
            'by_intent': defaultdict(list),
            'by_entity': defaultdict(list),
            'by_scenario': defaultdict(list),
            'by_complexity': defaultdict(list)
        }
        
        successful_enhancements = 0
        
        for i, item in enumerate(data):
            try:
                question_data = item.get('question', {})
                scene_data = item.get('scene_data')
                original_metadata = item.get('metadata', {})
                
                enhanced_qa = enhancer.enhance_qa_pair(question_data)
                
                enhanced_scene = enhancer.enhance_scene_metadata(scene_data)
                
                image_info = None
                if scene_data and scene_data.get('camera_data'):
                    image_info = enhancer.create_full_image_paths(scene_data['camera_data'])
                
                enhanced_item = {
                    'id': f"qa_{i:06d}",
                    'qa_pair': enhanced_qa,
                    'scene_metadata': enhanced_scene,
                    'image_info': image_info,
                    'retrieval_metadata': {
                        'has_scene_data': enhanced_scene is not None,
                        'has_images': image_info is not None and image_info.get('camera_count', 0) > 0,
                        'image_count': image_info.get('camera_count', 0) if image_info else 0,
                        'primary_image_available': image_info.get('has_front_camera', False) if image_info else False,
                        'processing_timestamp': datetime.now().isoformat()
                    }
                }
                
                item_id = enhanced_item['id']
                qa = enhanced_item['qa_pair']
                
                retrieval_index['by_intent'][qa.get('question_intent', 'unknown')].append(item_id)
                retrieval_index['by_complexity'][f"level_{qa.get('complexity', 1)}"].append(item_id)
                
                for entity in qa.get('entities', []):
                    retrieval_index['by_entity'][entity].append(item_id)
                
                if enhanced_scene:
                    scenario = enhanced_scene.get('scene_context', {}).get('driving_scenario', 'unknown')
                    retrieval_index['by_scenario'][scenario].append(item_id)
                
                enhanced_data.append(enhanced_item)
                successful_enhancements += 1
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Enhanced {i + 1}/{len(data)} items")
                
            except Exception as e:
                logger.error(f"Error enhancing item {i}: {e}")
                continue
        
        output_structure = {
            'metadata': {
                'version': '1.0_rag_optimized',
                'created': datetime.now().isoformat(),
                'total_qa_pairs': len(enhanced_data),
                'successful_enhancements': successful_enhancements,
                'qa_pairs_with_images': sum(1 for item in enhanced_data if item['retrieval_metadata']['has_images']),
                'qa_pairs_with_scene_data': sum(1 for item in enhanced_data if item['retrieval_metadata']['has_scene_data']),
                'features': [
                    'full_camera_coverage',
                    'optimized_retrieval',
                    'semantic_enhancement',
                    'scene_context',
                    'driving_scenarios'
                ]
            },
            'qa_pairs': enhanced_data,
            'retrieval_index': dict(retrieval_index)
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_structure, f, indent=2, ensure_ascii=False)
        
        stats = generate_summary_stats(enhanced_data)
        stats_path = output_path.with_name(f"{output_path.stem}_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ RAG optimization complete!")
        logger.info(f"   Enhanced data: {output_path}")
        logger.info(f"   Statistics: {stats_path}")
        logger.info(f"   Total QA pairs: {len(enhanced_data)}")
        logger.info(f"   With images: {sum(1 for item in enhanced_data if item['retrieval_metadata']['has_images'])}")
        logger.info(f"   With full scene data: {sum(1 for item in enhanced_data if item['retrieval_metadata']['has_scene_data'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during enhancement: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def generate_summary_stats(enhanced_data: List[Dict]) -> Dict[str, Any]:
    total_items = len(enhanced_data)
    
    with_images = sum(1 for item in enhanced_data if item['retrieval_metadata']['has_images'])
    full_camera_coverage = sum(1 for item in enhanced_data 
                              if item.get('image_info', {}).get('has_all_cameras', False))
    
    intent_counts = Counter()
    complexity_counts = Counter()
    scenario_counts = Counter()
    
    for item in enhanced_data:
        qa = item.get('qa_pair', {})
        intent_counts[qa.get('question_intent', 'unknown')] += 1
        complexity_counts[f"level_{qa.get('complexity', 1)}"] += 1
        
        scene = item.get('scene_metadata', {})
        if scene:
            scenario = scene.get('scene_context', {}).get('driving_scenario', 'unknown')
            scenario_counts[scenario] += 1
    
    return {
        'dataset_summary': {
            'total_qa_pairs': total_items,
            'with_images': with_images,
            'with_full_camera_coverage': full_camera_coverage,
            'with_scene_data': sum(1 for item in enhanced_data if item['retrieval_metadata']['has_scene_data'])
        },
        'question_analysis': {
            'by_intent': dict(intent_counts),
            'by_complexity': dict(complexity_counts),
            'avg_entities_per_question': np.mean([len(item.get('qa_pair', {}).get('entities', [])) for item in enhanced_data]),
            'avg_keywords_per_question': np.mean([len(item.get('qa_pair', {}).get('keywords', [])) for item in enhanced_data])
        },
        'scene_analysis': {
            'by_scenario': dict(scenario_counts),
            'avg_objects_per_scene': np.mean([
                item.get('scene_metadata', {}).get('objects', {}).get('count', 0) 
                for item in enhanced_data if item.get('scene_metadata')
            ]) if any(item.get('scene_metadata') for item in enhanced_data) else 0
        },
        'image_coverage': {
            'camera_distribution': {
                camera: sum(1 for item in enhanced_data 
                           if camera in item.get('image_info', {}).get('available_cameras', []))
                for camera in ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            }
        }
    }


def main():
    """Main function to run the optimized RAG enhancement."""
    print("=== OPTIMIZED RAG ENHANCEMENT ===")
    print("Enhancing unified.json for QA retrieval with full camera coverage")
    print()
    
    success = enhance_unified_json_for_rag()
    
    if success:
        print("✅ Enhancement successful!")
        print()
        print("📁 Generated files:")
        print("   • unified_rag_optimized.json - Enhanced QA pairs with full metadata")
        print("   • unified_rag_optimized_stats.json - Dataset statistics")
        print()
        print("🎯 Features included:")
        print("   • All 6 camera image paths")
        print("   • Optimized QA retrieval metadata")
        print("   • Scene context and driving scenarios")
        print("   • Entity extraction and keyword analysis")
        print("   • Retrieval indices for fast lookup")
        print()
        print("🚀 Ready for RAG system integration!")
    else:
        print("❌ Enhancement failed - check logs for details")


if __name__ == "__main__":
    main()