"""File utilities for DocuFind AI."""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import mimetypes
from datetime import datetime

from src.utils.config import (
    SUPPORTED_TEXT_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    DOCUMENTS_DIR,
    IMAGES_DIR
)


class FileUtils:
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Generate MD5 hash for file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def get_file_metadata(file_path: str) -> Dict[str, Any]:
        """Extract basic file metadata"""
        path = Path(file_path)
        stat = path.stat()
        
        return {
            "filename": path.name,
            "filepath": str(path.absolute()),
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_hash": FileUtils.get_file_hash(file_path),
            "mime_type": mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        }
    
    @staticmethod
    def is_supported_file(file_path: str) -> bool:
        """Check if file extension is supported"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension in SUPPORTED_TEXT_EXTENSIONS:
            return True
        if extension in SUPPORTED_IMAGE_EXTENSIONS:
            return True
        return False
    
    @staticmethod
    def is_text_file(file_path: str) -> bool:
        """Check if file is a text/document file"""
        path = Path(file_path)
        return path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """Check if file is an image file"""
        path = Path(file_path)
        return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    
    @staticmethod
    def get_all_files(directory: Path) -> List[Path]:
        """Get all supported files from directory"""
        files = []
        for ext in SUPPORTED_TEXT_EXTENSIONS + SUPPORTED_IMAGE_EXTENSIONS:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))
        return list(set(files))
    
    @staticmethod
    def organize_uploaded_file(file_path: str):
        """Move uploaded file to appropriate directory"""
        path = Path(file_path)
        if FileUtils.is_text_file(file_path):
            target_dir = DOCUMENTS_DIR
        elif FileUtils.is_image_file(file_path):
            target_dir = IMAGES_DIR
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        # Create target filename with timestamp to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{path.name}"
        target_path = target_dir / new_filename
        
        # Move file
        path.rename(target_path)
        return str(target_path)
