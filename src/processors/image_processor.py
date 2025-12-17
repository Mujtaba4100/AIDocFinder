"""Image processing utilities using CLIP, OCR, and color extraction."""

import torch
import clip
from PIL import Image
import pytesseract
from typing import Dict, Any, List
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import colorsys


class ImageProcessor:
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load CLIP model
        print(f"Loading CLIP model on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("CLIP model loaded successfully")
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return ""
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Save temp file for pytesseract
            temp_path = "temp_ocr.png"
            cv2.imwrite(temp_path, thresh)
            
            # Extract text
            text = pytesseract.image_to_string(Image.open(temp_path))
            
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
            
            return text.strip()
        except Exception as e:
            print(f"Error in OCR for {image_path}: {e}")
            return ""
    
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Get CLIP embedding for image"""
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                embedding = image_features.cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"Error getting embedding for {image_path}: {e}")
            return np.zeros(512)  # CLIP ViT-B/32 has 512-dimensional embeddings
    
    def extract_colors(self, image_path: str, num_colors: int = 5) -> List[Dict[str, Any]]:
        """Extract dominant colors from image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # Resize for faster processing
            image = cv2.resize(image, (100, 100))
            pixels = image.reshape(-1, 3)
            
            # Convert BGR to RGB
            pixels = [pixels[:, 2], pixels[:, 1], pixels[:, 0]]
            pixels = np.transpose(pixels)
            
            # Simple color quantization
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = []
            for center in kmeans.cluster_centers_:
                r, g, b = center
                # Convert to hex
                hex_color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
                
                # Convert to HSV for color name approximation
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                
                # Simple color naming
                color_name = self._get_color_name(h, s, v)
                
                colors.append({
                    "rgb": (int(r), int(g), int(b)),
                    "hex": hex_color,
                    "hsv": (h, s, v),
                    "name": color_name
                })
            
            return colors
        except Exception as e:
            print(f"Error extracting colors from {image_path}: {e}")
            return []
    
    def _get_color_name(self, h: float, s: float, v: float) -> str:
        """Approximate color name from HSV values"""
        if v < 0.2:
            return "black"
        elif v > 0.8 and s < 0.2:
            return "white"
        elif s < 0.3:
            return "gray"
        
        # Color ranges in degrees (0-360)
        h_deg = h * 360
        
        if h_deg < 15 or h_deg >= 345:
            return "red"
        elif 15 <= h_deg < 45:
            return "orange"
        elif 45 <= h_deg < 75:
            return "yellow"
        elif 75 <= h_deg < 165:
            return "green"
        elif 165 <= h_deg < 195:
            return "cyan"
        elif 195 <= h_deg < 255:
            return "blue"
        elif 255 <= h_deg < 285:
            return "purple"
        elif 285 <= h_deg < 345:
            return "pink"
        
        return "unknown"
    
    def generate_description(self, image_path: str) -> str:
        """Generate description using CLIP"""
        try:
            # Pre-defined prompts
            prompts = [
                "a photo of",
                "an image showing",
                "a picture of",
                "this is",
                "there is"
            ]
            
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            text = clip.tokenize(prompts).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(3)
                
                # Get top prompts
                top_prompts = [prompts[idx] for idx in indices.cpu().numpy()]
                
                # Create description
                if values[0] > 0.5:
                    description = top_prompts[0]
                else:
                    description = "an image containing various elements"
                
                # Add color info
                colors = self.extract_colors(image_path, 2)
                if colors:
                    color_names = [c["name"] for c in colors[:2]]
                    description += f" with {color_names[0]}"
                    if len(color_names) > 1:
                        description += f" and {color_names[1]} elements"
                
                return description
        except Exception as e:
            print(f"Error generating description for {image_path}: {e}")
            return "an image"
    
    def process(self, image_path: str) -> Dict[str, Any]:
        """Process image file and return structured data"""
        print(f"Processing image: {image_path}")
        
        # Get embedding
        embedding = self.get_image_embedding(image_path)
        
        # Extract text via OCR
        ocr_text = self.extract_text(image_path)
        
        # Extract colors
        colors = self.extract_colors(image_path)
        
        # Generate description
        description = self.generate_description(image_path)
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                format = img.format
                mode = img.mode
        except:
            width, height, format, mode = 0, 0, "unknown", "unknown"
        
        return {
            "file_type": "image",
            "embedding": embedding.tolist(),
            "ocr_text": ocr_text,
            "colors": colors,
            "description": description,
            "metadata": {
                "dimensions": f"{width}x{height}",
                "format": format,
                "mode": mode,
                "has_text": len(ocr_text) > 0
            },
            "summary": {
                "color_count": len(colors),
                "primary_colors": [c["name"] for c in colors[:3]],
                "text_length": len(ocr_text),
                "description": description
            },
            "processing_time": datetime.now().isoformat()
        }

# Test the processor
if __name__ == "__main__":
    processor = ImageProcessor(device="cpu")
    
    # Create a test image if none exists
    test_image = Path("test_image.png")
    if not test_image.exists():
        # Create a simple blue square image
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (200, 200), color='blue')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill='red')
        draw.text((60, 80), "Hostel Rules", fill='white')
        img.save("test_image.png")
        print("Created test_image.png")
    
    if test_image.exists():
        result = processor.process(str(test_image))
        print(f"Processed image. Embedding shape: {len(result['embedding'])}")
        print(f"Description: {result['description']}")
        print(f"Colors: {[c['name'] for c in result['colors'][:3]]}")
