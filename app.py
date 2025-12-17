import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import concurrent.futures
import hashlib
import pickle
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.processors.document_processor import DocumentProcessor
from src.database.vector_store import VectorStore
from src.utils.file_utils import FileUtils
from src.utils.config import (
	DOCUMENTS_DIR,
	IMAGES_DIR,
	VECTOR_STORE_DIR,
	SUPPORTED_IMAGE_EXTENSIONS,
	SUPPORTED_TEXT_EXTENSIONS,
)

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

@app.post("/search/", response_model=Dict[str, Any])
async def search_documents(search_query: SearchQuery):
	"""Search across all documents and images with optional on-the-fly folder search.

	Supports parallel processing and a simple folder cache to avoid recomputing embeddings.
	"""
	try:
		query = (search_query.query or "").strip()

		# If a folder path is provided, perform on-the-fly semantic search
		if search_query.path:
			folder_path = search_query.path
			from pathlib import Path
			import numpy as np
			from src.utils.folder_cache import FolderCache

			# Validate selectable options (as requested: file_type in [all, document, image])
			allowed_file_types = {"all", "document", "image"}

			ft = (search_query.file_type or "all").lower()
			if ft not in allowed_file_types:
				raise HTTPException(status_code=400, detail=f"Unsupported file_type: {ft}. Allowed: {sorted(list(allowed_file_types))}")

			p = Path(folder_path)
			if not p.exists() or not p.is_dir():
				raise HTTPException(status_code=400, detail=f"Path does not exist or is not a directory: {folder_path}")

			# Gather supported files from folder (non-recursive)
			supported = set([ext.lower() for ext in SUPPORTED_TEXT_EXTENSIONS + SUPPORTED_IMAGE_EXTENSIONS])
			files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in supported]

			# Filter by requested file_type
			def is_image_ext(ext: str) -> bool:
				return ext.lower() in SUPPORTED_IMAGE_EXTENSIONS

			if ft == "document":
				files = [f for f in files if not is_image_ext(f.suffix.lower())]
			elif ft == "image":
				files = [f for f in files if is_image_ext(f.suffix.lower())]

			if not files:
				return {"query": query, "results": [], "count": 0, "source": "folder"}

			cache = FolderCache()
			cache_data = cache.load(str(p))
			items = None

			if cache_data:
				# load cached items and convert embeddings to numpy arrays
				items = cache_data.get("items", [])
				for it in items:
					it["embedding"] = np.array(it["embedding"], dtype=float)

			if items is None:
				# extract and embed in parallel
				items = []
				failures = []

				# Step 1: extract content concurrently
				def extract_worker(fp: Path):
					try:
						res = processor.process_file(str(fp))
						return (fp.name, res, None)
					except Exception as e:
						return (str(fp.name), None, str(e))

				max_workers = min(8, (os.cpu_count() or 2) * 2)
				extracted = []
				with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
					futures = {ex.submit(extract_worker, f): f for f in files}
					for fut in concurrent.futures.as_completed(futures):
						fn, res, err = fut.result()
						if err:
							failures.append({"file": fn, "error": err})
						else:
							extracted.append((fn, res))

				# Step 2: separate text chunks and image embeddings
				text_chunks = []  # (file, text)
				image_tasks = []  # (file, fp)

				for fn, res in extracted:
					# infer image vs text
					if res.get("file_type") == "image":
						# try to get image embedding if provided, otherwise schedule generation
						emb = res.get("embedding")
						desc = res.get("description") or res.get("ocr_text") or ""
						if emb is not None:
							items.append({"file": fn, "text": desc, "embedding": np.array(emb, dtype=float), "is_image": True})
						else:
							# schedule image embedding generation via image processor
							image_tasks.append((fn, res))
						continue

					# documents
					chunks = res.get("chunks") or []
					if not chunks:
						txt = res.get("text") or ""
						if txt:
							text_chunks.append((fn, txt))
					else:
						for ch in chunks:
							chunk_text = ch["text"] if isinstance(ch, dict) else str(ch)
							text_chunks.append((fn, chunk_text))

				# Step 3: embed all text chunks in batches using the lightweight model
				if text_chunks:
					texts = [t for (_, t) in text_chunks]
					# encode in batches (convert_to_numpy ensures numpy output)
					try:
						embeddings = vector_store.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
					except TypeError:
						embeddings = vector_store.embedding_model.encode(texts)

					for (fn, _), emb in zip(text_chunks, embeddings):
						items.append({"file": fn, "text": (_ or "")[:2000], "embedding": np.array(emb, dtype=float), "is_image": False})

				# Step 4: compute image embeddings (if any) in parallel
				if image_tasks:
					def image_worker(pair):
						fn, res = pair
						try:
							# if processor exposes image_processor
							emb = None
							if "path" in res:
								# if res contains a path
								emb = processor.image_processor.get_image_embedding(res.get("path"))
							if emb is None and res.get("id"):
								emb = processor.image_processor.get_image_embedding(res.get("id"))
							# as fallback try to use description vectorization
							desc = res.get("description") or res.get("ocr_text") or ""
							return (fn, desc, emb, None)
						except Exception as e:
							return (fn, "", None, str(e))

					with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
						futures = [ex.submit(image_worker, t) for t in image_tasks]
						for fut in concurrent.futures.as_completed(futures):
							fn, desc, emb, err = fut.result()
							if err:
								failures.append({"file": fn, "error": err})
							else:
								if emb is not None:
									items.append({"file": fn, "text": (desc or "")[:2000], "embedding": np.array(emb, dtype=float), "is_image": True})
								else:
									# if no emb, attempt to encode desc with text model
									if desc:
										try:
											embt = vector_store.embedding_model.encode(desc)
											items.append({"file": fn, "text": (desc or "")[:2000], "embedding": np.array(embt, dtype=float), "is_image": False})
										except Exception:
											failures.append({"file": fn, "error": "no embedding available"})

				# Save cache (serialize embeddings to lists)
				to_cache = []
				for it in items:
					to_cache.append({"file": it["file"], "text": it.get("text", ""), "embedding": list(it["embedding"].astype(float)), "is_image": bool(it.get("is_image", False))})
				cache.save(str(p), to_cache)

			# At this point 'items' is a list of dicts with numpy embeddings
			if not items:
				return {"query": query, "results": [], "count": 0, "source": "folder"}

			# Build modality-specific lists
			text_items = [it for it in items if not it.get("is_image")]
			image_items = [it for it in items if it.get("is_image")]

			results_accum = []

			# Helper similarity
			def cosine(a, b):
				an = np.linalg.norm(a)
				bn = np.linalg.norm(b)
				if an == 0 or bn == 0:
					return 0.0
				return float(np.dot(a, b) / (an * bn))

			# If searching images only
			if ft == "image":
				if not image_items:
					return {"query": query, "results": [], "count": 0, "source": "folder"}
				# produce CLIP text embedding for query
				try:
					import clip
					import torch
					img_proc = processor.image_processor
					tokenized = clip.tokenize([query]).to(img_proc.device)
					with torch.no_grad():
						text_feat = img_proc.model.encode_text(tokenized)
					qv = text_feat.cpu().numpy()[0]
				except Exception:
					# fallback to text model for queries
					qv = np.array(vector_store.embedding_model.encode(query))

				for it in image_items:
					score = cosine(qv, np.array(it["embedding"]))
					results_accum.append({"file": it["file"], "text": it.get("text", "")[:500], "score": score})

			elif ft == "document":
				if not text_items:
					return {"query": query, "results": [], "count": 0, "source": "folder"}
				qv = np.array(vector_store.embedding_model.encode(query))
				for it in text_items:
					score = cosine(qv, np.array(it["embedding"]))
					results_accum.append({"file": it["file"], "text": it.get("text", "")[:500], "score": score})

			else:  # all: search both modalities and normalize across each modality
				# Text side
				text_results = []
				if text_items:
					qv_text = np.array(vector_store.embedding_model.encode(query))
					for it in text_items:
						s = cosine(qv_text, np.array(it["embedding"]))
						text_results.append({"file": it["file"], "text": it.get("text", "")[:500], "raw_score": s, "mod": "text"})
				# Image side
				image_results = []
				if image_items:
					try:
						import clip
						import torch
						img_proc = processor.image_processor
						tokenized = clip.tokenize([query]).to(img_proc.device)
						with torch.no_grad():
							text_feat = img_proc.model.encode_text(tokenized)
						qv_img = text_feat.cpu().numpy()[0]
					except Exception:
						qv_img = np.array(vector_store.embedding_model.encode(query))

					for it in image_items:
						s = cosine(qv_img, np.array(it["embedding"]))
						image_results.append({"file": it["file"], "text": it.get("text", "")[:500], "raw_score": s, "mod": "image"})

				# Normalize per-modality
				combined = []
				if text_results:
					max_t = max(r["raw_score"] for r in text_results) or 1.0
					for r in text_results:
						r["score"] = r["raw_score"] / max_t
						combined.append({"file": r["file"], "text": r["text"], "score": r["score"]})
				if image_results:
					max_i = max(r["raw_score"] for r in image_results) or 1.0
					for r in image_results:
						r["score"] = r["raw_score"] / max_i
						combined.append({"file": r["file"], "text": r["text"], "score": r["score"]})

				results_accum = combined

			# Sort, dedupe by file keeping best score per file, and return top-N
			results_accum = sorted(results_accum, key=lambda x: x["score"], reverse=True)
			seen = set()
			deduped = []
			for r in results_accum:
				if r["file"] not in seen:
					seen.add(r["file"])
					deduped.append(r)

			top_n = deduped[: max(1, int(search_query.limit or 10))]
			response = {"query": query, "results": top_n, "count": len(top_n), "source": "folder"}
			return response

		# Fallback to existing indexed searches
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
			hybrid_results = vector_store.hybrid_search(query, n_results=max(1, int(search_query.limit // 2)))
			return hybrid_results

	except HTTPException:
		raise
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
