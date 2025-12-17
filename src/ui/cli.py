#!/usr/bin/env python3
"""
Command Line Interface for DocuFind AI
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.vector_store import VectorStore
from src.processors.document_processor import DocumentProcessor
from src.utils.file_utils import FileUtils
from src.utils.config import DOCUMENTS_DIR, IMAGES_DIR

def search_cli(args):
    """Search command"""
    store = VectorStore()
    
    print(f"\n🔍 Searching for: '{args.query}'")
    print("-" * 60)
    
    if args.type == "text":
        results = store.search_text(args.query, n_results=args.limit)
        for i, result in enumerate(results):
            print(f"\n📄 Result {i+1} (Score: {result['score']:.3f})")
            print(f"File: {result['metadata'].get('filename', 'Unknown')}")
            print(f"Content: {result['document'][:200]}...")
            print("-" * 40)
    
    elif args.type == "image":
        results = store.search_images(args.query, n_results=args.limit)
        for i, result in enumerate(results):
            print(f"\n🖼️  Result {i+1} (Score: {result['score']:.3f})")
            print(f"File: {result['metadata'].get('filename', 'Unknown')}")
            print(f"Description: {result['description']}")
            print(f"Path: {result['metadata'].get('filepath', 'Unknown')}")
            print("-" * 40)
    
    else:  # hybrid
        results = store.hybrid_search(args.query, n_results=args.limit//2)
        
        if results['text_results']:
            print(f"\n📄 TEXT RESULTS ({len(results['text_results'])})")
            print("=" * 40)
            for i, result in enumerate(results['text_results'][:3]):
                print(f"{i+1}. {result['metadata'].get('filename', 'Unknown')} ({result['score']:.3f})")
                print(f"   {result['document'][:100]}...")
        
        if results['image_results']:
            print(f"\n🖼️  IMAGE RESULTS ({len(results['image_results'])})")
            print("=" * 40)
            for i, result in enumerate(results['image_results'][:3]):
                print(f"{i+1}. {result['metadata'].get('filename', 'Unknown')} ({result['score']:.3f})")
                print(f"   {result['description']}")

def index_cli(args):
    """Index command"""
    processor = DocumentProcessor(device="cpu")
    store = VectorStore()
    
    if args.file:
        # Index single file
        print(f"📂 Indexing file: {args.file}")
        result = processor.process_file(args.file)
        
        if result["file_type"] == "image":
            store.add_image(result)
            print(f"✅ Image indexed: {args.file}")
        else:
            store.add_text_document(result)
            print(f"✅ Document indexed: {args.file}")
    
    elif args.folder:
        # Index folder
        print(f"📁 Indexing folder: {args.folder}")
        file_utils = FileUtils()
        all_files = file_utils.get_all_files(Path(args.folder))
        
        if not all_files:
            print("❌ No supported files found")
            return
        
        print(f"Found {len(all_files)} files to process")
        
        results = processor.batch_process([str(f) for f in all_files])
        
        added = 0
        for result in results:
            try:
                if result["file_type"] == "image":
                    store.add_image(result)
                else:
                    store.add_text_document(result)
                added += 1
                print(f"✓ {result['file_metadata'].get('filename', 'Unknown')}")
            except Exception as e:
                print(f"✗ Failed: {e}")
        
        print(f"\n✅ Indexed {added}/{len(results)} files successfully")

def stats_cli(args):
    """Stats command"""
    store = VectorStore()
    stats = store.get_stats()
    
    print("\n📊 DOCUFIND AI STATISTICS")
    print("=" * 40)
    print(f"📄 Text Documents: {stats['text_documents']}")
    print(f"🖼️  Images: {stats['image_documents']}")
    print(f"📈 Total: {stats['total_documents']}")
    print(f"💾 Storage: {stats['persistence_path']}")

def list_cli(args):
    """List command"""
    store = VectorStore()
    
    if args.type == "text":
        docs = store.get_all_documents("text")
        print(f"\n📄 TEXT DOCUMENTS ({len(docs)})")
        print("=" * 60)
        for doc in docs[:args.limit]:
            print(f"• {doc['metadata'].get('filename', 'Unknown')}")
            print(f"  ID: {doc['id']}")
            print(f"  Type: {doc['metadata'].get('file_type', 'Unknown')}")
            print()
    
    elif args.type == "image":
        images = store.get_all_documents("image")
        print(f"\n🖼️  IMAGES ({len(images)})")
        print("=" * 60)
        for img in images[:args.limit]:
            print(f"• {img['metadata'].get('filename', 'Unknown')}")
            print(f"  Description: {img['document'][:50]}...")
            print(f"  Colors: {img['metadata'].get('primary_colors', 'N/A')}")
            print()

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="DocuFind AI - Unified Document Search")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents and images")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-t", "--type", choices=["text", "image", "all"], 
                              default="all", help="Search type")
    search_parser.add_argument("-l", "--limit", type=int, default=10, 
                              help="Maximum results")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index files")
    index_group = index_parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("-f", "--file", help="File to index")
    index_group.add_argument("-d", "--folder", help="Folder to index")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List indexed documents")
    list_parser.add_argument("-t", "--type", choices=["text", "image"], 
                            default="text", help="Document type")
    list_parser.add_argument("-l", "--limit", type=int, default=20, 
                            help="Maximum items to list")
    
    args = parser.parse_args()
    
    if args.command == "search":
        search_cli(args)
    elif args.command == "index":
        index_cli(args)
    elif args.command == "stats":
        stats_cli(args)
    elif args.command == "list":
        list_cli(args)
    else:
        # Interactive mode
        print("🔍 DocuFind AI - Interactive Mode")
        print("-" * 40)
        
        while True:
            try:
                query = input("\nEnter search query (or 'quit' to exit): ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if query:
                    search_cli(argparse.Namespace(
                        query=query, type="all", limit=5
                    ))
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
