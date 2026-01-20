"""
YOLOv8 Object Detection Module for offline image understanding.

Detects objects, people, animals, and scene elements in images.
Uses ultralytics YOLOv8 with local model caching.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image


# COCO class names for YOLOv8 (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Synonym mappings for common search terms
OBJECT_SYNONYMS: Dict[str, List[str]] = {
    'person': ['man', 'woman', 'boy', 'girl', 'child', 'kid', 'human', 'people'],
    'dog': ['puppy', 'canine', 'pet', 'hound', 'pup'],
    'cat': ['kitten', 'feline', 'kitty'],
    'car': ['vehicle', 'automobile', 'auto'],
    'bicycle': ['bike', 'cycle'],
    'motorcycle': ['motorbike', 'bike'],
    'airplane': ['plane', 'aircraft', 'jet'],
    'bus': ['coach', 'vehicle'],
    'truck': ['lorry', 'vehicle'],
    'bird': ['fowl', 'avian'],
    'horse': ['pony', 'stallion', 'mare'],
    'cow': ['cattle', 'bull'],
    'elephant': ['pachyderm'],
    'cell phone': ['phone', 'mobile', 'smartphone', 'cellphone'],
    'laptop': ['computer', 'notebook', 'pc'],
    'tv': ['television', 'screen', 'monitor'],
    'couch': ['sofa', 'settee'],
    'chair': ['seat'],
    'bed': ['mattress'],
    'dining table': ['table', 'desk'],
    'potted plant': ['plant', 'flower', 'houseplant'],
    'sports ball': ['ball', 'football', 'soccer ball', 'basketball'],
    'wine glass': ['glass', 'goblet'],
    'cup': ['mug', 'glass'],
    'book': ['novel', 'textbook'],
    'clock': ['watch', 'timepiece'],
    'teddy bear': ['stuffed animal', 'plush', 'toy'],
}

# Build reverse mapping (synonym -> canonical)
SYNONYM_TO_CANONICAL: Dict[str, str] = {}
for canonical, synonyms in OBJECT_SYNONYMS.items():
    for syn in synonyms:
        SYNONYM_TO_CANONICAL[syn.lower()] = canonical
    SYNONYM_TO_CANONICAL[canonical.lower()] = canonical


class ObjectDetector:
    """
    YOLOv8-based object detector for image understanding.
    
    Detects objects and returns labels with confidence scores.
    Models are cached locally for offline use.
    """
    
    MODEL_NAME = "yolov8n.pt"  # Nano model - fast and small (~6MB)
    
    def __init__(self, cache_dir: Union[str, Path]) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._model = None
        self._init_error: Optional[str] = None
        self._available = False
        
        # Set YOLO cache directory
        os.environ.setdefault("YOLO_CONFIG_DIR", str(self.cache_dir / "ultralytics"))
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize YOLOv8 model."""
        try:
            from ultralytics import YOLO
            
            # Load YOLOv8 nano model (downloads on first use, then cached)
            model_path = self.cache_dir / "ultralytics" / self.MODEL_NAME
            
            if model_path.exists():
                self._model = YOLO(str(model_path))
            else:
                # Download model (will be cached in ultralytics folder)
                self._model = YOLO(self.MODEL_NAME)
            
            self._available = True
            
        except ImportError as e:
            self._init_error = f"ultralytics not installed: {e}"
        except Exception as e:
            self._init_error = f"Failed to load YOLOv8: {e}"
    
    @property
    def available(self) -> bool:
        return self._available and self._model is not None
    
    @property
    def init_error(self) -> Optional[str]:
        return self._init_error
    
    def detect_objects(
        self,
        image_path: Union[str, Path],
        confidence_threshold: float = 0.25,
        max_objects: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Detect objects in a single image.
        
        Args:
            image_path: Path to the image file.
            confidence_threshold: Minimum confidence score (0-1).
            max_objects: Maximum number of objects to return.
            
        Returns:
            List of (object_label, confidence) tuples, sorted by confidence.
        """
        if not self.available:
            return []
        
        try:
            # Run inference
            results = self._model(
                str(image_path),
                conf=confidence_threshold,
                verbose=False,
            )
            
            if not results or len(results) == 0:
                return []
            
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []
            
            # Extract detections
            detections: List[Tuple[str, float]] = []
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id < len(COCO_CLASSES):
                    label = COCO_CLASSES[cls_id]
                    detections.append((label, conf))
            
            # Sort by confidence, take top N
            detections.sort(key=lambda x: x[1], reverse=True)
            return detections[:max_objects]
            
        except Exception:
            return []
    
    def detect_objects_batch(
        self,
        image_paths: List[Union[str, Path]],
        confidence_threshold: float = 0.25,
        max_objects: int = 20,
        batch_size: int = 8,
    ) -> List[List[Tuple[str, float]]]:
        """
        Detect objects in multiple images (batch processing).
        
        Args:
            image_paths: List of image file paths.
            confidence_threshold: Minimum confidence score.
            max_objects: Maximum objects per image.
            batch_size: Batch size for processing.
            
        Returns:
            List of detection lists, one per image.
        """
        if not self.available:
            return [[] for _ in image_paths]
        
        results_list: List[List[Tuple[str, float]]] = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = self._process_batch(batch_paths, confidence_threshold, max_objects)
            results_list.extend(batch_results)
        
        return results_list
    
    def _process_batch(
        self,
        image_paths: List[Union[str, Path]],
        confidence_threshold: float,
        max_objects: int,
    ) -> List[List[Tuple[str, float]]]:
        """Process a single batch of images."""
        results_list: List[List[Tuple[str, float]]] = [[] for _ in image_paths]
        
        try:
            # Run batch inference
            results = self._model(
                [str(p) for p in image_paths],
                conf=confidence_threshold,
                verbose=False,
            )
            
            for idx, result in enumerate(results):
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                
                detections: List[Tuple[str, float]] = []
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls_id < len(COCO_CLASSES):
                        label = COCO_CLASSES[cls_id]
                        detections.append((label, conf))
                
                detections.sort(key=lambda x: x[1], reverse=True)
                results_list[idx] = detections[:max_objects]
                
        except Exception:
            pass
        
        return results_list
    
    def get_unique_objects(self, detections: List[Tuple[str, float]]) -> List[str]:
        """
        Get unique object labels from detections.
        
        Args:
            detections: List of (label, confidence) tuples.
            
        Returns:
            List of unique object labels.
        """
        seen = set()
        unique = []
        for label, _ in detections:
            if label not in seen:
                seen.add(label)
                unique.append(label)
        return unique
    
    def format_objects_string(self, detections: List[Tuple[str, float]]) -> str:
        """
        Format detections as a searchable string.
        
        Args:
            detections: List of (label, confidence) tuples.
            
        Returns:
            Comma-separated string of unique objects.
        """
        unique = self.get_unique_objects(detections)
        return ", ".join(unique) if unique else ""


def normalize_query_objects(query: str) -> List[str]:
    """
    Extract and normalize object references from a search query.
    
    Converts synonyms to canonical COCO class names.
    
    Args:
        query: User search query.
        
    Returns:
        List of canonical object labels found in query.
    """
    import re
    
    query_lower = query.lower()
    words = re.findall(r'\b\w+\b', query_lower)
    
    found_objects: List[str] = []
    for word in words:
        # Check if word is a known object or synonym
        if word in SYNONYM_TO_CANONICAL:
            canonical = SYNONYM_TO_CANONICAL[word]
            if canonical not in found_objects:
                found_objects.append(canonical)
        elif word in [c.lower() for c in COCO_CLASSES]:
            # Direct match to COCO class
            if word not in found_objects:
                found_objects.append(word)
    
    return found_objects


def objects_match_query(objects_string: str, query_objects: List[str]) -> float:
    """
    Check if detected objects match query objects.
    
    Args:
        objects_string: Comma-separated detected objects.
        query_objects: Normalized query object labels.
        
    Returns:
        Match score (0.0 to 1.0) based on proportion of query objects found.
    """
    if not query_objects or not objects_string:
        return 0.0
    
    objects_lower = objects_string.lower()
    
    matches = 0
    for obj in query_objects:
        if obj.lower() in objects_lower:
            matches += 1
    
    if matches == 0:
        return 0.0
    
    # Score based on proportion of query objects matched
    return matches / len(query_objects)
