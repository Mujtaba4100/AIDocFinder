#!/usr/bin/env python3
"""
DocuFind AI - Unified Document Search System
Run this file to start the application
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import chromadb
        import streamlit
        import fastapi
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def create_sample_data():
    """Create sample data if none exists"""
    from src.utils.config import DOCUMENTS_DIR, IMAGES_DIR
    
    # Create sample text file
    sample_text = Path(DOCUMENTS_DIR) / "hostel_rules.txt"
    if not sample_text.exists():
        with open(sample_text, 'w') as f:
            f.write("""HOSTEL RULES AND REGULATIONS

1. GENERAL CONDUCT:
   - Respect all residents and staff
   - No noise after 10:00 PM
   - Keep common areas clean
   - No smoking in rooms

2. ROOM MAINTENANCE:
   - Weekly room inspection every Monday
   - Report damages immediately
   - Air conditioners should be turned off when not in room

3. VISITORS:
   - Visitors allowed only in common areas
   - Must leave by 8:00 PM
   - All visitors must register at reception

4. SAFETY:
   - Fire drills first Monday of every month
   - Emergency exits must be clear
   - Contact security: Ext. 911

Document updated: January 2024""")
        print(f"📝 Created sample document: {sample_text.name}")

def main():
    """Main entry point"""
    print("=" * 60)
    print("🚀 DOCUFIND AI - UNIFIED DOCUMENT SEARCH SYSTEM")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    print("\n📊 Available Interfaces:")
    print("1. 🌐 FastAPI Backend (http://localhost:8000)")
    print("2. 🎨 Streamlit UI (http://localhost:8501)")
    print("3. 💻 CLI Interface")
    
    print("\n🔧 Starting services...")
    
    try:
        # Start FastAPI in background
        print("\n🌐 Starting FastAPI backend...")
        import threading
        
        def run_fastapi():
            import uvicorn
            uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")
        
        api_thread = threading.Thread(target=run_fastapi, daemon=True)
        api_thread.start()
        
        # Wait a bit for API to start
        time.sleep(2)
        
        # Start Streamlit
        print("🎨 Starting Streamlit UI...")
        
        # Open browser after delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
            webbrowser.open("http://localhost:8000")
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Run Streamlit in main thread
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/ui/streamlit_app.py", 
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.serverAddress", "localhost",
            "--server.maxUploadSize", "50"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
