import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.vector_store import VectorStore
from src.processors.document_processor import DocumentProcessor
from src.utils.config import DOCUMENTS_DIR, IMAGES_DIR
from src.utils.file_utils import FileUtils
import json

# Page config
st.set_page_config(
    page_title="DocuFind AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def get_vector_store():
    return VectorStore()

@st.cache_resource
def get_processor():
    return DocumentProcessor(device="cpu")

vector_store = get_vector_store()
processor = get_processor()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
    }
    .result-card {
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    .image-result {
        border-left: 5px solid #4CAF50;
    }
    .score-badge {
        background-color: #1E88E5;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🔍 DocuFind AI")
    st.markdown("---")
    
    # System Stats
    st.subheader("System Status")
    stats = vector_store.get_stats()
    st.metric("Text Documents", stats["text_documents"])
    st.metric("Images", stats["image_documents"])
    st.metric("Total", stats["total_documents"])
    
    st.markdown("---")
    
    # File Upload
    st.subheader("Upload Files")
    uploaded_file = st.file_uploader(
        "Choose files to index",
        type=['pdf', 'txt', 'doc', 'docx', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        # Save uploaded file
        upload_dir = DOCUMENTS_DIR if uploaded_file.name.lower().endswith(('.pdf', '.txt', '.doc', '.docx')) else IMAGES_DIR
        file_path = upload_dir / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process file
        with st.spinner(f"Processing {uploaded_file.name}..."):
            result = processor.process_file(str(file_path))
            
            if result["file_type"] == "image":
                file_id = vector_store.add_image(result)
                st.success(f"✅ Image indexed successfully!")
            else:
                file_id = vector_store.add_text_document(result)
                st.success(f"✅ Document indexed successfully!")
    
    st.markdown("---")
    
    # Batch Process
    if st.button("🔄 Process All Files in Data Folder"):
        with st.spinner("Processing all files..."):
            from src.utils.file_utils import FileUtils
            file_utils = FileUtils()
            
            all_files = file_utils.get_all_files(DOCUMENTS_DIR) + file_utils.get_all_files(IMAGES_DIR)
            
            if all_files:
                results = processor.batch_process([str(f) for f in all_files])
                
                added_count = 0
                for result in results:
                    try:
                        if result["file_type"] == "image":
                            vector_store.add_image(result)
                        else:
                            vector_store.add_text_document(result)
                        added_count += 1
                    except Exception as e:
                        st.error(f"Failed to add {result.get('id', 'unknown')}: {e}")
                
                st.success(f"✅ Processed {len(results)} files, added {added_count} to index")
            else:
                st.warning("No files found in data folder")

    st.markdown("---")
    st.subheader("Process Any Folder (enter full path)")
    folder_path_input = st.text_input(
        "Enter folder path to process (e.g., C:\\data\\mydocs or C:/data/mydocs):",
        value="",
        key="folder_path"
    )

    if st.button("Process Folder", key="process_folder_button"):
        if not folder_path_input:
            st.error("Please enter a folder path")
        else:
            folder = Path(folder_path_input)
            if not folder.exists() or not folder.is_dir():
                st.error("Folder not found or not a directory: " + folder_path_input)
            else:
                with st.spinner(f"Processing files in {folder}..."):
                    file_utils = FileUtils()
                    all_files = file_utils.get_all_files(folder)
                    if not all_files:
                        st.info("No supported files found in the provided folder")
                    else:
                        results = processor.batch_process([str(f) for f in all_files])
                        added_count = 0
                        for result in results:
                            try:
                                if result["file_type"] == "image":
                                    vector_store.add_image(result)
                                else:
                                    vector_store.add_text_document(result)
                                added_count += 1
                            except Exception as e:
                                st.error(f"Failed to add {result.get('id', 'unknown')}: {e}")

                        st.success(f"✅ Processed {len(results)} files, added {added_count} to index")

# Main Content
st.markdown('<h1 class="main-header">🔍 DocuFind AI</h1>', unsafe_allow_html=True)
st.markdown("### Unified Document & Image Search System")

# Search Bar
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "🔎 Search across all documents and images",
        placeholder="e.g., 'hostel rules' or 'images with blue cars'",
        key="search_query"
    )
with col2:
    search_type = st.selectbox(
        "Search Type",
        ["All", "Text Only", "Images Only"],
        key="search_type"
    )

# Search Button
if st.button("Search", type="primary", use_container_width=True):
    if query:
        with st.spinner("Searching..."):
            # Perform search
            if search_type == "Text Only":
                results = vector_store.search_text(query, n_results=10)
                
                st.markdown(f'<h3 class="sub-header">📄 Text Results ({len(results)})</h3>', unsafe_allow_html=True)
                
                if results:
                    for i, result in enumerate(results):
                        with st.container():
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>📄 {result['metadata'].get('filename', 'Document')} 
                                <span class="score-badge">Relevance: {result['score']:.2f}</span></h4>
                                <p><strong>Content:</strong> {result['document'][:300]}...</p>
                                <p><small>Type: {result['metadata'].get('file_type', 'Unknown')} | 
                                Path: {result['metadata'].get('filepath', 'N/A')}</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show more button
                            with st.expander("View Full Content"):
                                st.text(result['document'])
                else:
                    st.info("No text documents found matching your query.")
            
            elif search_type == "Images Only":
                results = vector_store.search_images(query, n_results=10)
                
                st.markdown(f'<h3 class="sub-header">🖼️ Image Results ({len(results)})</h3>', unsafe_allow_html=True)
                
                if results:
                    cols = st.columns(3)
                    for i, result in enumerate(results):
                        with cols[i % 3]:
                            try:
                                # Try to display image
                                img_path = result['metadata'].get('filepath')
                                if img_path and Path(img_path).exists():
                                    st.image(img_path, caption=result['description'][:50] + "...")
                                    st.caption(f"Score: {result['score']:.2f}")
                                    st.caption(f"Colors: {result['metadata'].get('primary_colors', 'N/A')}")
                                else:
                                    st.info("Image not found on disk")
                            except:
                                st.info("Could not display image")
                else:
                    st.info("No images found matching your query.")
            
            else:  # All (Hybrid Search)
                results = vector_store.hybrid_search(query, n_results=5)
                
                # Text Results
                if results['text_results']:
                    st.markdown(f'<h3 class="sub-header">📄 Text Results ({len(results["text_results"])})</h3>', unsafe_allow_html=True)
                    
                    for i, result in enumerate(results['text_results'][:3]):
                        with st.container():
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>📄 {result['metadata'].get('filename', 'Document')} 
                                <span class="score-badge">{result['score']:.2f}</span></h4>
                                <p>{result['document'][:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Image Results
                if results['image_results']:
                    st.markdown(f'<h3 class="sub-header">🖼️ Image Results ({len(results["image_results"])})</h3>', unsafe_allow_html=True)
                    
                    cols = st.columns(3)
                    for i, result in enumerate(results['image_results'][:6]):
                        with cols[i % 3]:
                            try:
                                img_path = result['metadata'].get('filepath')
                                if img_path and Path(img_path).exists():
                                    st.image(img_path, caption=result['description'][:30] + "...", width=200)
                                    st.caption(f"Score: {result['score']:.2f}")
                            except:
                                st.info("Image")
                
                if not results['text_results'] and not results['image_results']:
                    st.info("No results found for your query.")
    else:
        st.warning("Please enter a search query")

# Example Queries
st.markdown("---")
st.subheader("💡 Example Queries")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Find hostel rules"):
        st.session_state.search_query = "hostel rules"
        st.rerun()
with col2:
    if st.button("Search for images"):
        st.session_state.search_query = "image"
        st.session_state.search_type = "Images Only"
        st.rerun()
with col3:
    if st.button("Documents about policies"):
        st.session_state.search_query = "policy rules regulations"
        st.rerun()

# File Browser Section
st.markdown("---")
st.subheader("📁 File Browser")

tab1, tab2 = st.tabs(["Documents", "Images"])

with tab1:
    doc_files = list(DOCUMENTS_DIR.glob("*"))
    if doc_files:
        for file in doc_files[:10]:  # Show first 10
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.text(file.name)
            with col2:
                st.text(f"{file.stat().st_size / 1024:.1f} KB")
            with col3:
                if st.button("View", key=f"view_{file.name}"):
                    if file.suffix.lower() == '.txt':
                        with open(file, 'r') as f:
                            st.text_area("File Content", f.read(), height=200)
    else:
        st.info("No documents found. Upload some files!")

with tab2:
    img_files = list(IMAGES_DIR.glob("*"))
    if img_files:
        cols = st.columns(3)
        for i, file in enumerate(img_files[:9]):  # Show first 9
            with cols[i % 3]:
                try:
                    st.image(str(file), caption=file.name, use_column_width=True)
                except:
                    st.text(file.name)
    else:
        st.info("No images found. Upload some files!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🔍 DocuFind AI - Unified Document Intelligence System</p>
    <p>Supports: PDF, DOC, TXT, JPG, PNG, and more</p>
</div>
""", unsafe_allow_html=True)

# Run with: streamlit run src/ui/streamlit_app.py
