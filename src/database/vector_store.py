"""Vector store implementation using ChromaDB and Sentence Transformers."""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import json

from src.utils.config import (
    CHROMA_PERSIST_DIR,
    TEXT_COLLECTION_NAME,
    IMAGE_COLLECTION_NAME,
    EMBEDDING_MODEL
)


class VectorStore:
    def __init__(self, embedding_model=None):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collections
        self.text_collection = self.client.get_or_create_collection(
            name=TEXT_COLLECTION_NAME,
            metadata={"description": "Text documents collection"}
        )
        
        self.image_collection = self.client.get_or_create_collection(
            name=IMAGE_COLLECTION_NAME,
            metadata={"description": "Image embeddings collection"}
        )
        
        # Initialize embedding model
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        # Lazy CLIP model placeholders for text->image search
        self._clip_model = None
        self._clip_device = "cpu"
    
    def add_text_document(self, document: Dict[str, Any]) -> str:
        """Add text document to vector store"""
        try:
            doc_id = document["id"]
            text = document.get("text", "")
            metadata = document.get("metadata", {})
            file_metadata = document.get("file_metadata", {})
            
            # Create chunks if not already chunked
            if "chunks" in document and document["chunks"]:
                chunks = document["chunks"]
            else:
                # Simple chunking
                from src.processors.pdf_processor import PDFProcessor
                processor = PDFProcessor()
                chunks = processor.chunk_text(text)
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_text = chunk["text"] if isinstance(chunk, dict) else chunk
                
                # Generate embedding
                embedding = self.embedding_model.encode(chunk_text)
                
                # Prepare metadata
                chunk_metadata = {
                    **metadata,
                    **file_metadata,
                    "chunk_id": i,
                    "chunk_start": chunk.get("start_char", 0),
                    "chunk_end": chunk.get("end_char", len(chunk_text)),
                    "document_id": doc_id,
                    "filename": file_metadata.get("filename", ""),
                    "file_type": document.get("file_type", ""),
                    "added_at": datetime.now().isoformat()
                }
                
                ids.append(chunk_id)
                documents.append(chunk_text)
                metadatas.append(chunk_metadata)
                embeddings.append(embedding.tolist())
            
            # Add to collection
            self.text_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            print(f"Added document {doc_id} with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            print(f"Error adding text document: {e}")
            raise
    
    def add_image(self, image_data: Dict[str, Any]) -> str:
        """Add image to vector store"""
        try:
            image_id = image_data["id"]
            embedding = image_data.get("embedding", [])
            description = image_data.get("description", "")
            metadata = image_data.get("metadata", {})
            file_metadata = image_data.get("file_metadata", {})
            colors = image_data.get("colors", [])
            
            # Prepare metadata
            image_metadata = {
                **metadata,
                **file_metadata,
                "description": description,
                "colors": json.dumps([c["name"] for c in colors[:3]]) if colors else "",
                "primary_colors": ", ".join([c["name"] for c in colors[:2]]) if colors else "",
                "has_text": len(image_data.get("ocr_text", "")) > 0,
                "document_id": image_id,
                "filename": file_metadata.get("filename", ""),
                "file_type": "image",
                "added_at": datetime.now().isoformat()
            }
            
            # Add to collection (images are stored by embedding only)
            self.image_collection.add(
                ids=[image_id],
                # Use description as document text for searchability
                documents=[description],
                metadatas=[image_metadata],
                embeddings=[embedding]
            )
            
            print(f"Added image {image_id}")
            return image_id
            
        except Exception as e:
            print(f"Error adding image: {e}")
            raise
    
    def search_text(self, query: str, n_results: int = 10, filters: Dict = None) -> List[Dict[str, Any]]:
        """Search for text documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Perform search
            results = self.text_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["ids"]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "score": 1 - results["distances"][0][i]  # Convert to similarity score
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching text: {e}")
            return []
    
    def search_images(self, query: str, n_results: int = 10, filters: Dict = None) -> List[Dict[str, Any]]:
        """Search for images using text query"""
        try:
            # Ensure we always filter to images
            if filters is None:
                filters = {"file_type": "image"}
            else:
                # enforce image file_type in filters
                filters = {**filters, "file_type": "image"}

            # Use CLIP text encoder to produce a query embedding compatible with image embeddings
            try:
                if self._clip_model is None:
                    import torch
                    import clip
                    # load CLIP model lazily on CPU (device selection auto)
                    device = self._clip_device
                    self._clip_model, _ = clip.load("ViT-B/32", device=device)
                    self._clip_model.eval()

                import torch
                import clip
                text_tokens = clip.tokenize([query]).to(self._clip_device)
                with torch.no_grad():
                    text_features = self._clip_model.encode_text(text_tokens)
                query_embedding = text_features.cpu().numpy()[0].flatten()
                print(f"[DEBUG] Image search - query embedding shape: {query_embedding.shape}")

                # Query by embeddings for image similarity
                results = self.image_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results,
                    where=filters,
                    include=["documents", "metadatas", "distances"]
                )

            except Exception as e:
                # Fallback: if CLIP isn't available, fall back to text matching on descriptions
                print(f"[WARN] CLIP text encoder unavailable, falling back to text query: {e}")
                results = self.image_collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=filters,
                    include=["documents", "metadatas", "distances"]
                )
            
            # Format results
            formatted_results = []
            num_found = 0
            if results and "ids" in results and results["ids"]:
                num_found = len(results["ids"][0])
                for i in range(num_found):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "description": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "score": 1 - results["distances"][0][i]
                    })

            print(f"[DEBUG] Image search - results found: {num_found}")
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching images: {e}")
            return []
    
    def hybrid_search(self, query: str, n_results: int = 5) -> Dict[str, List]:
        """Perform hybrid search across text and images"""
        text_results = self.search_text(query, n_results=n_results)
        image_results = self.search_images(query, n_results=n_results)
        
        return {
            "text_results": text_results,
            "image_results": image_results,
            "query": query,
            "total_results": len(text_results) + len(image_results)
        }
    
    def get_all_documents(self, collection: str = "text") -> List[Dict]:
        """Get all documents from collection"""
        try:
            if collection == "text":
                coll = self.text_collection
            else:
                coll = self.image_collection
            
            results = coll.get(include=["documents", "metadatas"])
            
            documents = []
            for i in range(len(results["ids"])):
                documents.append({
                    "id": results["ids"][i],
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
            
            return documents
        except Exception as e:
            print(f"Error getting all documents: {e}")
            return []
    
    def delete_document(self, doc_id: str, collection: str = "text") -> bool:
        """Delete document from vector store"""
        try:
            if collection == "text":
                # Delete all chunks for this document
                chunks = self.text_collection.get(
                    where={"document_id": doc_id}
                )
                if chunks["ids"]:
                    self.text_collection.delete(ids=chunks["ids"])
            else:
                self.image_collection.delete(ids=[doc_id])
            
            print(f"Deleted document {doc_id} from {collection}")
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        text_count = self.text_collection.count()
        image_count = self.image_collection.count()
        
        return {
            "text_documents": text_count,
            "image_documents": image_count,
            "total_documents": text_count + image_count,
            "collections": ["text", "images"],
            "persistence_path": CHROMA_PERSIST_DIR
        }

# Test the vector store
if __name__ == "__main__":
    print("Testing Vector Store...")
    
    # Create test data
    test_document = {
        "id": "test_doc_123",
        "text": "This is a test document about hostel rules and regulations.",
        "metadata": {"author": "Test Author", "year": "2023"},
        "file_metadata": {
            "filename": "test_rules.pdf",
            "filepath": "/path/to/test.pdf"
        },
        "file_type": "pdf"
    }
    
    # Initialize vector store
    store = VectorStore()
    
    # Add document
    print("Adding test document...")
    store.add_text_document(test_document)
    
    # Search
    print("\nSearching for 'hostel rules'...")
    results = store.search_text("hostel rules")
    
    for i, result in enumerate(results[:3]):
        print(f"Result {i+1}: {result['document'][:50]}... (Score: {result['score']:.3f})")
    
    # Get stats
    stats = store.get_stats()
    print(f"\nVector Store Stats: {stats}")
