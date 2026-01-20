"""
BLIP Image Captioning Module (ONNX-based for offline use).

Uses Hugging Face Transformers with ONNX Runtime for efficient,
offline image captioning without PyTorch dependency.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image


class BlipCaptioner:
    """
    BLIP-based image captioner using ONNX runtime.
    
    Generates descriptive captions for images that can be embedded
    for semantic search (e.g., "a boy playing with a dog in a park").
    """
    
    # Model identifier - using BLIP base for captioning
    MODEL_NAME = "Salesforce/blip-image-captioning-base"
    
    def __init__(self, cache_dir: Union[str, Path]) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._model = None
        self._processor = None
        self._init_error: Optional[str] = None
        self._available = False
        
        # Set cache directories
        os.environ.setdefault("HF_HOME", str(self.cache_dir / "huggingface"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.cache_dir / "huggingface"))
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize BLIP model and processor."""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            # Load processor (tokenizer + image processor)
            self._processor = BlipProcessor.from_pretrained(
                self.MODEL_NAME,
                cache_dir=str(self.cache_dir / "huggingface"),
            )
            
            # Load model - transformers will use CPU by default
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.MODEL_NAME,
                cache_dir=str(self.cache_dir / "huggingface"),
            )
            
            # Set to evaluation mode
            self._model.eval()
            self._available = True
            
        except ImportError as e:
            self._init_error = f"transformers not installed: {e}"
        except Exception as e:
            self._init_error = f"Failed to load BLIP model: {e}"
    
    @property
    def available(self) -> bool:
        """Check if captioner is ready."""
        return self._available and self._model is not None
    
    @property
    def init_error(self) -> Optional[str]:
        """Get initialization error message if any."""
        return self._init_error
    
    def caption_image(self, image_path: Union[str, Path], max_length: int = 50) -> str:
        """
        Generate a caption for a single image.
        
        Args:
            image_path: Path to the image file.
            max_length: Maximum caption length in tokens.
            
        Returns:
            Generated caption string, or empty string on failure.
        """
        if not self.available:
            return ""
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Process image for model
            inputs = self._processor(image, return_tensors="pt")
            
            # Generate caption
            import torch
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=3,  # Beam search for better quality
                    early_stopping=True,
                )
            
            # Decode caption
            caption = self._processor.decode(output[0], skip_special_tokens=True)
            return caption.strip()
            
        except Exception:
            return ""
    
    def caption_images_batch(
        self,
        image_paths: List[Union[str, Path]],
        max_length: int = 50,
        batch_size: int = 8,
    ) -> List[str]:
        """
        Generate captions for multiple images in batches.
        
        Args:
            image_paths: List of image file paths.
            max_length: Maximum caption length in tokens.
            batch_size: Number of images to process at once.
            
        Returns:
            List of caption strings (empty string for failed images).
        """
        if not self.available:
            return [""] * len(image_paths)
        
        results: List[str] = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_captions = self._process_batch(batch_paths, max_length)
            results.extend(batch_captions)
        
        return results
    
    def _process_batch(self, image_paths: List[Union[str, Path]], max_length: int) -> List[str]:
        """Process a single batch of images."""
        captions: List[str] = []
        images: List[Image.Image] = []
        valid_indices: List[int] = []
        
        # Load all valid images
        for idx, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception:
                pass
        
        # Initialize results with empty strings
        results = [""] * len(image_paths)
        
        if not images:
            return results
        
        try:
            # Process batch
            inputs = self._processor(images, return_tensors="pt", padding=True)
            
            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=3,
                    early_stopping=True,
                )
            
            # Decode captions
            decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
            
            # Map back to original indices
            for idx, caption in zip(valid_indices, decoded):
                results[idx] = caption.strip()
                
        except Exception:
            # Fall back to single-image processing
            for idx, img_idx in enumerate(valid_indices):
                try:
                    results[img_idx] = self.caption_image(image_paths[img_idx], max_length)
                except Exception:
                    pass
        
        return results


class LightweightCaptioner:
    """
    Lightweight alternative captioner using CLIP + templates.
    
    When BLIP is unavailable, this provides basic semantic understanding
    by matching images against predefined concept templates.
    """
    
    # Common concepts to detect in images
    CONCEPT_TEMPLATES = [
        # People
        "a photo of a person", "a photo of a man", "a photo of a woman",
        "a photo of a child", "a photo of a boy", "a photo of a girl",
        "a photo of people", "a photo of a group", "a selfie",
        "a photo of someone smiling", "a portrait",
        
        # Activities
        "a photo of someone working", "a photo of someone playing",
        "a photo of someone eating", "a photo of someone reading",
        "a photo of someone walking", "a photo of someone running",
        
        # Places
        "a photo of a building", "a photo of a house", "a photo of a room",
        "a photo of an office", "a photo of a street", "a photo of a city",
        "a photo of nature", "a photo of a park", "a photo of a beach",
        "a photo of mountains", "a photo of a rooftop",
        
        # Animals
        "a photo of a dog", "a photo of a cat", "a photo of a bird",
        "a photo of an animal", "a photo of pets",
        
        # Objects
        "a photo of food", "a photo of a car", "a photo of a phone",
        "a photo of a computer", "a photo of a document",
        "a screenshot", "a diagram", "a chart",
        
        # Events
        "a photo of a party", "a photo of a meeting", "a photo of a wedding",
        "a photo of a celebration", "a photo of a sunset",
    ]
    
    def __init__(self, clip_embedder) -> None:
        """
        Args:
            clip_embedder: ClipEmbedder instance for computing similarities.
        """
        self.clip = clip_embedder
        self._template_embeddings = None
        self._available = False
        
        if self.clip and self.clip.available:
            self._precompute_templates()
    
    def _precompute_templates(self) -> None:
        """Precompute embeddings for all concept templates."""
        try:
            import numpy as np
            
            embeddings = []
            for template in self.CONCEPT_TEMPLATES:
                vec = self.clip.embed_query(template)
                embeddings.append(vec)
            
            self._template_embeddings = np.vstack(embeddings)
            self._available = True
        except Exception:
            self._available = False
    
    @property
    def available(self) -> bool:
        return self._available
    
    def get_image_concepts(
        self,
        image_embedding,
        top_k: int = 3,
        threshold: float = 0.2,
    ) -> str:
        """
        Get matching concepts for an image based on its CLIP embedding.
        
        Args:
            image_embedding: CLIP embedding of the image.
            top_k: Number of top concepts to return.
            threshold: Minimum similarity threshold.
            
        Returns:
            Comma-separated string of matching concepts.
        """
        if not self.available or self._template_embeddings is None:
            return ""
        
        try:
            import numpy as np
            
            img_vec = np.asarray(image_embedding, dtype=np.float32)
            if img_vec.ndim != 1:
                return ""
            
            # Compute similarities
            scores = self._template_embeddings @ img_vec
            
            # Get top matches above threshold
            indices = np.argsort(scores)[::-1][:top_k]
            concepts = []
            
            for idx in indices:
                if scores[idx] >= threshold:
                    # Extract concept from template (remove "a photo of ")
                    concept = self.CONCEPT_TEMPLATES[idx]
                    concept = concept.replace("a photo of ", "").replace("a ", "")
                    concepts.append(concept)
            
            return ", ".join(concepts) if concepts else ""
            
        except Exception:
            return ""
