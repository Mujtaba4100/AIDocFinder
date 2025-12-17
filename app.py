import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.processors.document_processor import DocumentProcessor
from src.database.vector_store import VectorStore
from src.utils.file_utils import FileUtils
from src.utils.config import DOCUMENTS_DIR, IMAGES_DIR, VECTOR_STORE_DIR

# Initialize components
processor = DocumentProcessor(device="cpu")
vector_store = VectorStore()
file_utils = FileUtils()

# Create FastAPI app
app = FastAPI(title="DocuFind AI", version="1.0.0")

# Mount static files directory
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Pydantic models
class SearchQuery(BaseModel):
	query: str
	file_type: str = "all"  # all, text, image
	limit: int = 10
	path: str = None  # optional folder path for on-the-fly semantic search
	# allow incoming JSON to use the key 'type' for document type
	doc_type: str = None

	class Config:
		allow_population_by_field_name = True
		fields = {"doc_type": {"alias": "type"}}

class UploadResponse(BaseModel):
	success: bool
	message: str
	file_id: str = None
	file_path: str = None

class SearchResult(BaseModel):
	id: str
	content: str
	metadata: Dict[str, Any]
	score: float
	type: str  # text or image

class StatsResponse(BaseModel):
	text_documents: int
	image_documents: int
	total_documents: int

@app.get("/", response_class=HTMLResponse)
async def home():
	"""Home page with API documentation"""
	html_content = """
	<!DOCTYPE html>
	<html>
	<head>
		<title>DocuFind AI API</title>
		<style>
			body { font-family: Arial, sans-serif; margin: 40px; }
			.endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
			code { background: #eee; padding: 2px 5px; }
			.method { color: #007bff; font-weight: bold; }
		</style>
	</head>
	<body>
		<h1>🔍 DocuFind AI API</h1>
		<p>Unified document and image search system</p>
        
		<h2>Available Endpoints:</h2>
        
		<div class="endpoint">
			<span class="method">POST</span> <code>/upload/</code>
			<p>Upload documents or images for processing</p>
		</div>
        
		<div class="endpoint">
			<span class="method">POST</span> <code>/search/</code>
			<p>Search across all documents and images</p>
		</div>
        
		<div class="endpoint">
			<span class="method">GET</span> <code>/stats/</code>
			<p>Get system statistics</p>
		</div>
        
		<div class="endpoint">
			<span class="method">GET</span> <code>/documents/</code>
			<p>List all indexed documents</p>
		</div>
        
		<h2>Quick Start:</h2>
		<pre><code>curl -X POST "http://localhost:8000/search/" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "hostel rules"}'</code></pre>
        
		<p>Check out the Streamlit UI at <a href="/ui">/ui</a></p>
	</body>
	</html>
	"""
	return HTMLResponse(content=html_content)

@app.post("/upload/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
	"""Upload and process a file"""
	try:
		# Validate file
		if not file_utils.is_supported_file(file.filename):
			raise HTTPException(
				status_code=400,
				detail=f"Unsupported file type. Supported: PDF, DOC, TXT, Images"
			)
        
		# Save uploaded file temporarily
		temp_path = Path("temp_upload") / file.filename
		temp_path.parent.mkdir(exist_ok=True)
        
		with open(temp_path, "wb") as buffer:
			content = await file.read()
			buffer.write(content)
        
		# Organize file to proper directory
		organized_path = file_utils.organize_uploaded_file(str(temp_path))
        
		# Process file
		result = processor.process_file(organized_path)
        
		# Add to vector store
		if result["file_type"] == "image":
			file_id = vector_store.add_image(result)
		else:
			file_id = vector_store.add_text_document(result)
        
		# Cleanup temp file
		temp_path.unlink(missing_ok=True)
        
		return UploadResponse(
			success=True,
			message=f"File processed successfully",
			file_id=file_id,
			file_path=organized_path
		)
        
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/", response_model=Dict[str, Any])
async def search_documents(search_query: SearchQuery):
	"""Search across all documents and images"""
	try:
		query = search_query.query.lower().strip()
		# If a folder path is provided, perform on-the-fly semantic search
		if search_query.path:
			folder_path = search_query.path
			from pathlib import Path

			p = Path(folder_path)
			if not p.exists() or not p.is_dir():
				raise HTTPException(status_code=400, detail=f"Path does not exist or is not a directory: {folder_path}")

			# Gather supported files from folder (text/pdf/docx/txt)
			files = FileUtils.get_all_files(p)
			# Filter to text/document extensions only
			text_files = [f for f in files if FileUtils.is_text_file(str(f))]

			# Process files and build chunk embeddings on the fly
			from src.processors.pdf_processor import PDFProcessor
			pdf_proc = PDFProcessor()

			chunks = []  # each item: dict with keys: file, text, embedding
			total_chunks = 0
			for f in text_files:
				try:
					# Use DocumentProcessor helpers to extract text
					result = processor.process_file(str(f))
					text = result.get("text", "")
					file_name = f.name

					# Get chunks: if processor returned chunks (PDF), use them, else chunk the text
					if "chunks" in result and result["chunks"]:
						file_chunks = result["chunks"]
					else:
						file_chunks = pdf_proc.chunk_text(text)

					# Generate embeddings for each chunk
					for ch in file_chunks:
						chunk_text = ch["text"] if isinstance(ch, dict) else str(ch)
						emb = vector_store.embedding_model.encode(chunk_text)
						chunks.append({"file": file_name, "text": chunk_text, "embedding": emb})
						total_chunks += 1
				except Exception as e:
					print(f"Failed to process {f}: {e}")

			if total_chunks == 0:
				return {"query": query, "results": [], "count": 0, "source": "folder"}

			# Encode query
			query_emb = vector_store.embedding_model.encode(query)
			import numpy as np

			print(f"[DEBUG] Folder search - total chunks: {total_chunks}")
			print(f"[DEBUG] Folder search - query embedding shape: {np.array(query_emb).shape}")

			# Compute cosine similarities
			sims = []
			qv = np.array(query_emb)
			qnorm = np.linalg.norm(qv)
			for item in chunks:
				ev = np.array(item["embedding"])
				denom = (np.linalg.norm(ev) * qnorm)
				score = float(np.dot(qv, ev) / denom) if denom != 0 else 0.0
				sims.append({"file": item["file"], "text": item["text"][:500], "score": score})

			# Sort and return top N
			sims = sorted(sims, key=lambda x: x["score"], reverse=True)
			top_n = sims[: search_query.limit]

			return {"query": query, "results": top_n, "count": len(top_n), "source": "folder"}
        
		# Perform search based on file type
		if search_query.file_type == "text":
			results = vector_store.search_text(query, n_results=search_query.limit)
			return {
				"query": query,
				"results": results,
				"type": "text_only",
				"count": len(results)
			}
		elif search_query.file_type == "image":
			results = vector_store.search_images(query, n_results=search_query.limit)
			return {
				"query": query,
				"results": results,
				"type": "image_only",
				"count": len(results)
			}
		else:
			# Hybrid search
			hybrid_results = vector_store.hybrid_search(query, n_results=search_query.limit//2)
			return hybrid_results
            
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/", response_model=StatsResponse)
async def get_stats():
	"""Get system statistics"""
	try:
		stats = vector_store.get_stats()
		return StatsResponse(
			text_documents=stats["text_documents"],
			image_documents=stats["image_documents"],
			total_documents=stats["total_documents"]
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/")
async def list_documents(collection: str = "text"):
	"""List all indexed documents"""
	try:
		documents = vector_store.get_all_documents(collection)
		return {
			"collection": collection,
			"count": len(documents),
			"documents": documents[:50]  # Limit to 50
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-process/")
async def batch_process_folder(folder_path: str = None):
	"""Process all files in a folder"""
	try:
		if folder_path is None or not Path(folder_path).exists():
			# Use default data directory
			folder_path = str(DOCUMENTS_DIR)
        
		# Get all supported files
		folder = Path(folder_path)
		all_files = []
        
		for ext in file_utils.SUPPORTED_TEXT_EXTENSIONS + file_utils.SUPPORTED_IMAGE_EXTENSIONS:
			all_files.extend(folder.glob(f"*{ext}"))
			all_files.extend(folder.glob(f"*{ext.upper()}"))
        
		# Remove duplicates
		all_files = list(set(all_files))
        
		# Process files
		results = processor.batch_process([str(f) for f in all_files])
        
		# Add to vector store
		added_ids = []
		for result in results:
			try:
				if result["file_type"] == "image":
					file_id = vector_store.add_image(result)
				else:
					file_id = vector_store.add_text_document(result)
				added_ids.append(file_id)
			except Exception as e:
				print(f"Failed to add {result.get('id', 'unknown')}: {e}")
        
		return {
			"success": True,
			"message": f"Processed {len(results)} files, added {len(added_ids)} to index",
			"processed_count": len(results),
			"added_count": len(added_ids),
			"file_ids": added_ids
		}
        
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/ui")
async def streamlit_ui_redirect():
	"""Redirect to Streamlit UI"""
	html_content = """
	<!DOCTYPE html>
	<html>
	<head>
		<meta http-equiv="refresh" content="0; url=http://localhost:8501" />
	</head>
	<body>
		<p>Redirecting to Streamlit UI... <a href="http://localhost:8501">Click here if not redirected</a></p>
	</body>
	</html>
	"""
	return HTMLResponse(content=html_content)

if __name__ == "__main__":
	print("🚀 Starting DocuFind AI Server...")
	print(f"📁 Data directory: {DOCUMENTS_DIR}")
	print(f"📊 Vector store: {VECTOR_STORE_DIR}")
	print(f"🌐 API running at: http://localhost:8000")
	print(f"🎨 Streamlit UI at: http://localhost:8501")
    
	# Create test data if none exists
	test_files = list(DOCUMENTS_DIR.glob("*")) + list(IMAGES_DIR.glob("*"))
	if len(test_files) == 0:
		print("\n📝 Creating test files...")
		# Create a sample text file
		sample_text = """Hostel Rules and Regulations
        
1. General Rules:
   - Curfew is at 10:00 PM for all residents
   - No loud music after 9:00 PM
   - Keep your room clean and tidy
   - Smoking is strictly prohibited

2. Visitor Policy:
   - Visitors allowed only in common areas
   - No overnight guests without permission
   - All visitors must sign in at reception

3. Safety Rules:
   - Fire exits must be kept clear at all times
   - Report any maintenance issues immediately
   - Emergency contact: 911

4. Meal Times:
   - Breakfast: 7:00 AM - 9:00 AM
   - Lunch: 12:00 PM - 2:00 PM
   - Dinner: 6:00 PM - 8:00 PM

This document was last updated on January 15, 2024."""
        
		with open(DOCUMENTS_DIR / "hostel_rules.txt", "w") as f:
			f.write(sample_text)
		print("Created: hostel_rules.txt")
    
	uvicorn.run(app, host="0.0.0.0", port=8000)
