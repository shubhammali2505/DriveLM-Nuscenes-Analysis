
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EgoVehicleState:
    """Represents the ego vehicle state at a specific timestamp."""
    timestamp: int
    location: List[float]  
    rotation: List[float]  
    velocity: Optional[List[float]] = None  
    acceleration: Optional[List[float]] = None  

@dataclass
class ObjectAnnotation:
    token: str
    category_name: str
    instance_token: str
    visibility: int
    translation: List[float] 
    size: List[float]  
    rotation: List[float]  
    velocity: Optional[List[float]] = None
    acceleration: Optional[List[float]] = None


@dataclass
class CameraData:
    """Represents camera sensor data."""
    token: str
    filename: str
    camera_intrinsic: List[List[float]]
    ego_pose_token: str
    calibrated_sensor_token: str
    timestamp: int


@dataclass
class SceneData:
    """Unified scene representation"""
    scene_token: str
    sample_token: str
    timestamp: int
    ego_vehicle_state: EgoVehicleState
    camera_data: Dict[str, CameraData]  # key: camera channel
    object_annotations: List[ObjectAnnotation]
    description: Optional[str] = None


@dataclass
class DriveLMQuestion:
    """Represents a DriveLM question with associated metadata."""
    question_id: str
    scene_token: str
    sample_token: str
    question: str
    answer: str
    question_type: str
    reasoning_chain: Optional[List[str]] = None
    image_path: Optional[str] = None


class NuScenesParser:
    
    def __init__(self, dataroot: str):

        self.dataroot = Path(dataroot)
        self.version = "v1.0-mini"
        self.tables = {}
        self._load_tables()
    
    def _load_tables(self) -> None:
        """Load all NuScenes metadata tables."""
        table_names = [
            'scene', 'sample', 'sample_data', 'sample_annotation',
            'ego_pose', 'calibrated_sensor', 'sensor', 'log',
            'category', 'attribute', 'visibility', 'instance', 'map'
        ]
        
        for table_name in table_names:
            table_path = self.dataroot / self.version / f"{table_name}.json"
            if table_path.exists():
                with open(table_path, 'r', encoding='utf-8') as f:
                    self.tables[table_name] = {
                        record['token']: record for record in json.load(f)
                    }
                logger.info(f"Loaded {len(self.tables[table_name])} {table_name} records")
            else:
                logger.warning(f"Table {table_name} not found at {table_path}")
                self.tables[table_name] = {}
    
    def get_scene_samples(self, scene_token: str) -> List[str]:
        scene = self.tables['scene'].get(scene_token)
        if not scene:
            return []
        
        samples = []
        sample_token = scene['first_sample_token']
        
        while sample_token != '':
            samples.append(sample_token)
            sample = self.tables['sample'].get(sample_token)
            sample_token = sample['next'] if sample else ''
        
        return samples
    
    def parse_ego_vehicle_state(self, sample_token: str) -> Optional[EgoVehicleState]:
        """Parse ego vehicle state for a given sample."""
        sample = self.tables['sample'].get(sample_token)
        if not sample:
            logger.debug(f"Sample not found: {sample_token}")
            return None

        lidar_sample_data = None
        for sample_data_token, sample_data in self.tables['sample_data'].items():
            if (sample_data.get('sample_token') == sample_token and 
                'LIDAR' in sample_data.get('channel', '')):
                lidar_sample_data = sample_data
                break
        
        if not lidar_sample_data:
            for sample_data_token, sample_data in self.tables['sample_data'].items():
                if sample_data.get('sample_token') == sample_token:
                    lidar_sample_data = sample_data
                    break
        
        if not lidar_sample_data:
            logger.debug(f"No sample data found for sample {sample_token}")
            return None
        
        ego_pose = self.tables['ego_pose'].get(lidar_sample_data['ego_pose_token'])
        if not ego_pose:
            logger.debug(f"Ego pose not found for token {lidar_sample_data['ego_pose_token']}")
            return None
        
        return EgoVehicleState(
            timestamp=sample['timestamp'],
            location=ego_pose['translation'],
            rotation=ego_pose['rotation']
        )
    
    def parse_camera_data(self, sample_token: str) -> Dict[str, CameraData]:
        """Parse camera data for all cameras in a sample."""
        sample = self.tables['sample'].get(sample_token)
        if not sample:
            return {}
        
        camera_data = {}
        
        for sample_data_token, sample_data in self.tables['sample_data'].items():
            if (sample_data.get('sample_token') == sample_token and 
                'CAM' in sample_data.get('channel', '')):
                
                sensor_channel = sample_data.get('channel', '')
                
                calibrated_sensor = self.tables['calibrated_sensor'].get(
                    sample_data['calibrated_sensor_token']
                )
                
                if calibrated_sensor:
                    camera_data[sensor_channel] = CameraData(
                        token=sample_data_token,
                        filename=sample_data['filename'],
                        camera_intrinsic=calibrated_sensor['camera_intrinsic'],
                        ego_pose_token=sample_data['ego_pose_token'],
                        calibrated_sensor_token=sample_data['calibrated_sensor_token'],
                        timestamp=sample_data['timestamp']
                    )
        
        return camera_data
    
    def parse_object_annotations(self, sample_token: str) -> List[ObjectAnnotation]:
        """Parse object annotations for a given sample."""
        sample = self.tables['sample'].get(sample_token)
        if not sample:
            return []
        
        if 'anns' not in sample:
            logger.debug(f"Sample {sample_token} missing 'anns' key")
            return []
        
        annotations = []
        
        for ann_token in sample['anns']:
            annotation = self.tables['sample_annotation'].get(ann_token)
            if not annotation:
                continue
            
            category = self.tables['category'].get(annotation['category_token'])
            category_name = category['name'] if category else 'unknown'
            
            annotations.append(ObjectAnnotation(
                token=ann_token,
                category_name=category_name,
                instance_token=annotation['instance_token'],
                visibility=annotation['visibility_token'],
                translation=annotation['translation'],
                size=annotation['size'],
                rotation=annotation['rotation'],
                velocity=annotation.get('velocity'),
                acceleration=annotation.get('acceleration')
            ))
        
        return annotations
    
    def parse_scene(self, scene_token: str, sample_token: str) -> Optional[SceneData]:
        """Parse complete scene data for a given sample."""
        sample = self.tables['sample'].get(sample_token)
        if not sample:
            logger.debug(f"Sample {sample_token} not found in NuScenes tables")
            return None
        
        scene = self.tables['scene'].get(scene_token)
        if not scene:
            logger.debug(f"Scene {scene_token} not found in NuScenes tables")
            return None
        
        ego_state = self.parse_ego_vehicle_state(sample_token)
        if not ego_state:
            logger.debug(f"Could not parse ego vehicle state for sample {sample_token}")
            return None
        
        camera_data = self.parse_camera_data(sample_token)
        object_annotations = self.parse_object_annotations(sample_token)
        
        return SceneData(
            scene_token=scene_token,
            sample_token=sample_token,
            timestamp=ego_state.timestamp,
            ego_vehicle_state=ego_state,
            camera_data=camera_data,
            object_annotations=object_annotations,
            description=scene.get('description')
        )


class DriveLMParser:
    """Parser for DriveLM question-answering data."""
    
    def __init__(self, json_path: str):
        """
        Initialize DriveLM parser.
        
        Args:
            json_path: Path to DriveLM JSON file
        """
        self.json_path = Path(json_path)
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load DriveLM JSON data - corrected to handle dictionary structure."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                logger.info(f"Loaded DriveLM data with {len(data)} scenes")
                return data
            elif isinstance(data, list):
                logger.warning("DriveLM data appears to be in list format - converting")
                converted_data = {}
                for i, item in enumerate(data):
                    converted_data[f"scene_{i}"] = item
                return converted_data
            else:
                logger.error(f"Unexpected DriveLM data format: {type(data)}")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading DriveLM data: {e}")
            return {}
        
    def parse_questions(self) -> List[DriveLMQuestion]:
        questions = []
        
        for scene_token, scene_data in self.data.items():
            if not isinstance(scene_data, dict):
                logger.warning(f"Invalid scene data for {scene_token}")
                continue
            
            key_frames = scene_data.get('key_frames', {})
            if not key_frames:
                logger.warning(f"No key_frames found for scene {scene_token}")
                continue
            
            for sample_token, frame_data in key_frames.items():
                if not isinstance(frame_data, dict):
                    continue
                
                qa_data = frame_data.get('QA', {})
                if not qa_data:
                    continue
                
                image_paths = frame_data.get('image_paths', {})
                
                for category, questions_list in qa_data.items():
                    if not isinstance(questions_list, list):
                        continue
                    
                    for qa_idx, qa_pair in enumerate(questions_list):
                        if not isinstance(qa_pair, dict):
                            continue
                        
                        question_text = qa_pair.get('Q', '').strip()
                        answer_text = qa_pair.get('A', '').strip()
                        
                        if not question_text or not answer_text:
                            continue
                        
                        question_id = f"{scene_token}_{sample_token}_{category}_{qa_idx}"
                        
                        reasoning_chain = qa_pair.get('reasoning_chain', [])
                        if isinstance(reasoning_chain, str):
                            reasoning_chain = [reasoning_chain]
                        
                        primary_image = None
                        if image_paths:
                            primary_image = image_paths.get('CAM_FRONT') or \
                                        list(image_paths.values())[0] if image_paths else None
                        
                        question_obj = DriveLMQuestion(
                            question_id=question_id,
                            scene_token=scene_token,
                            sample_token=sample_token,
                            question=question_text,
                            answer=answer_text,
                            question_type=category,  
                            reasoning_chain=reasoning_chain,
                            image_path=primary_image
                        )
                        
                        questions.append(question_obj)
        
        logger.info(f"Successfully parsed {len(questions)} questions from {len(self.data)} scenes")
        
        category_counts = {}
        for q in questions:
            category_counts[q.question_type] = category_counts.get(q.question_type, 0) + 1
        
        logger.info("Question categories found:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} questions")
        
        return questions
        
    def _classify_question_type(self, question: str) -> str:
        """Classify question type based on content."""
        question_lower = question.lower()
        
        patterns = {
            'object_detection': [
                'what', 'which', 'identify', 'see', 'visible'
            ],
            'object_location': [
                'where', 'position', 'location', 'behind', 'front', 'left',
                'right'
            ],
            'object_behavior': [
                'moving', 'speed', 'direction', 'turning', 'stopping'
            ],
            'safety_assessment': [
                'safe', 'danger', 'risk', 'collision', 'brake'
            ],
            'planning': [
                'should', 'action', 'next', 'plan', 'decision'
            ],
            'count': ['how many', 'number of', 'count'],
            'yes_no': ['is', 'are', 'can', 'will', 'does']
        }
        
        for q_type, keywords in patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                return q_type
        
        return 'other'


class UnifiedDataStructure:
    
    def __init__(
        self, 
        nuscenes_parser: NuScenesParser, 
        drivelm_parser: DriveLMParser
    ):
 
        self.nuscenes_parser = nuscenes_parser
        self.drivelm_parser = drivelm_parser
        self.unified_data = []
        self._build_unified_structure()
    
    def _build_unified_structure(self) -> None:
        """Build unified data structure linking all components."""
        questions = self.drivelm_parser.parse_questions()
        
        questions = questions[:1000]
        logger.info(f"Processing first {len(questions)} questions for testing...")
        
        matched = 0
        failed = 0
        
        available_samples = set(self.nuscenes_parser.tables['sample'].keys())
        logger.info(f"Available NuScenes samples: {len(available_samples)}")
        
        question_samples = set(q.sample_token for q in questions[:10])  # First 10
        logger.info(f"Sample tokens from first 10 questions: {question_samples}")
        
        overlap = question_samples.intersection(available_samples)
        logger.info(f"Overlapping sample tokens (first 10): {overlap}")
        
        for i, question in enumerate(questions):
            try:
                scene_data = None
                if question.scene_token and question.sample_token:
                    scene_data = self.nuscenes_parser.parse_scene(
                        question.scene_token, 
                        question.sample_token
                    )
                
                if scene_data:
                    matched += 1
                    unified_item = {
                        'question': {
                            'question_id': question.question_id,
                            'scene_token': question.scene_token,
                            'sample_token': question.sample_token,
                            'question': question.question,
                            'answer': question.answer,
                            'question_type': question.question_type,
                            'reasoning_chain': question.reasoning_chain or [],
                            'image_path': question.image_path
                        },
                        'scene_data': {
                            'scene_token': scene_data.scene_token,
                            'sample_token': scene_data.sample_token,
                            'timestamp': scene_data.timestamp,
                            'ego_vehicle_state': asdict(scene_data.ego_vehicle_state),
                            'camera_data': {k: asdict(v) for k, v in scene_data.camera_data.items()},
                            'object_annotations': [asdict(obj) for obj in scene_data.object_annotations],
                            'description': scene_data.description
                        },
                        'metadata': {
                            'has_scene_data': True,
                            'num_objects': len(scene_data.object_annotations),
                            'num_cameras': len(scene_data.camera_data),
                            'timestamp': scene_data.timestamp
                        }
                    }
                else:
                    failed += 1
                    if failed <= 5:  # Log first 5 failures for debugging
                        logger.debug(f"No scene data found for question {i}: scene={question.scene_token}, sample={question.sample_token}")
                    
                    unified_item = {
                        'question': {
                            'question_id': question.question_id,
                            'scene_token': question.scene_token,
                            'sample_token': question.sample_token,
                            'question': question.question,
                            'answer': question.answer,
                            'question_type': question.question_type,
                            'reasoning_chain': question.reasoning_chain or [],
                            'image_path': question.image_path
                        },
                        'scene_data': None,
                        'metadata': {
                            'has_scene_data': False,
                            'num_objects': 0,
                            'num_cameras': 0,
                            'timestamp': None
                        }
                    }
                
                self.unified_data.append(unified_item)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(questions)} questions, matched: {matched}, failed: {failed}")
                
            except Exception as e:
                failed += 1
                logger.error(f"Error processing question {i}: {e}")
                logger.error(f"Question ID: {question.question_id}")
                logger.error(f"Scene token: {question.scene_token}")
                logger.error(f"Sample token: {question.sample_token}")
                
                unified_item = {
                    'question': {
                        'question_id': question.question_id,
                        'scene_token': question.scene_token,
                        'sample_token': question.sample_token,
                        'question': question.question,
                        'answer': question.answer,
                        'question_type': question.question_type,
                        'reasoning_chain': question.reasoning_chain or [],
                        'image_path': question.image_path
                    },
                    'scene_data': None,
                    'metadata': {
                        'has_scene_data': False,
                        'num_objects': 0,
                        'num_cameras': 0,
                        'timestamp': None,
                        'error': str(e)
                    }
                }
                self.unified_data.append(unified_item)
                continue
        
        logger.info(f"Built unified structure with {len(self.unified_data)} items")
        logger.info(f"Questions with matching scene data: {matched}")
        logger.info(f"Questions without scene data: {failed}")
        
    def get_data(self) -> List[Dict[str, Any]]:
        return self.unified_data
    
    def save_to_json(self, output_path: str) -> None:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.unified_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved unified data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving unified data: {e}")