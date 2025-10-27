import argparse
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set, Dict, Any

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter # CORRECTED IMPORT
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

# ===========================================================
# Load environment & Configuration
# ===========================================================
load_dotenv()

VECTOR_DIR = Path(os.getenv("VECTOR_DIR", "data/vectorstore/faiss"))
RAW_DIR = Path(os.getenv("RAW_NEWS_DATA_DIR", "data/raw/news"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
BATCH_SIZE = 32 # Number of documents to process in a batch

PROCESSED_FILES_LOG = VECTOR_DIR / "processed_files.log"


class VectorStoreManager:
    """
    Manages the creation, updating, and querying of the FAISS vector store.
    """
    def __init__(self):
        self.vector_store: Optional[FAISS] = None
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ Initialized VectorStoreManager with model '{EMBEDDING_MODEL}'.")
        print(f"   Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")


    def _load_processed_files(self) -> Set[Path]:
        """Loads the set of already processed file paths from a log file."""
        if not PROCESSED_FILES_LOG.exists():
            return set()
        with open(PROCESSED_FILES_LOG, "r", encoding="utf-8") as f:
            return {Path(line.strip()) for line in f if line.strip()}


    def _save_processed_files(self, processed_files: Set[Path]) -> None:
        """Saves the set of processed file paths to a log file."""
        with open(PROCESSED_FILES_LOG, "w", encoding="utf-8") as f:
            for file_path in sorted(processed_files):
                f.write(f"{file_path}\n")


    def _load_and_chunk_documents(self, days_back: int) -> List[Document]:
        """Loads new JSON files and splits them into chunked Documents."""
        processed_files = self._load_processed_files()
        new_chunks = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        if not RAW_DIR.exists():
            print(f"❌ Raw data directory not found at: {RAW_DIR}")
            return []

        all_files = list(RAW_DIR.rglob("*.json"))
        new_files = [
            f for f in all_files
            if f not in processed_files and datetime.fromtimestamp(f.stat().st_mtime) >= cutoff_date
        ]

        if not new_files:
            return []

        print(f"ℹ️ Found {len(new_files)} new raw files to process.")

        for file_path in new_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                article = data.get("article")
                if not article or not article.get("content", "").strip():
                    print(f"⚠️ Skipping empty article: {file_path}")
                    continue

                # Create a single document to be split
                doc = Document(
                    page_content=article.get("content"),
                    metadata={
                        "source": data.get("source", "unknown"),
                        "sport": data.get("sport", "unknown"),
                        "competition": data.get("competition", "unknown"),
                        "title": article.get("title", "No Title"),
                        "url": article.get("url", ""),
                        "published_date": article.get("date_published", ""),
                        "author": article.get("author", "Unknown"),
                        "processing_status": article.get("processing_status", False),
                        "file_path": str(file_path)
                    }
                )
                # Add the title to the beginning of the content for better context
                doc.page_content = f"Article Title: {doc.metadata['title']}\n\n{doc.page_content}"

                # Split the document into chunks
                chunks = self.text_splitter.split_documents([doc])
                new_chunks.extend(chunks)

            except Exception as e:
                print(f"❌ Error processing file {file_path}: {e}")

        print(f"📚 Generated {len(new_chunks)} new chunks from {len(new_files)} files.")
        return new_chunks

    def create_or_update(self, days_back: int = 30) -> None:
        """Creates or updates the vector store with new documents."""
        new_chunks = self._load_and_chunk_documents(days_back)

        if not new_chunks:
            print("✅ Vector store is already up to date.")
            return

        print("🔄 Loading existing vector store...")
        self.load()

        # Add new documents in batches
        for i in range(0, len(new_chunks), BATCH_SIZE):
            batch = new_chunks[i:i + BATCH_SIZE]
            if self.vector_store:
                self.vector_store.add_documents(batch)
            else:
                self.vector_store = FAISS.from_documents(batch, self.embeddings)
            print(f"  ...embedded batch {i//BATCH_SIZE + 1}/{(len(new_chunks) - 1)//BATCH_SIZE + 1}")

        print("💾 Saving updated vector store to disk...")
        self.vector_store.save_local(str(VECTOR_DIR))

        # Update the processed files log
        processed_files = self._load_processed_files()
        newly_processed_files = {Path(chunk.metadata["file_path"]) for chunk in new_chunks}
        processed_files.update(newly_processed_files)
        self._save_processed_files(processed_files)
        print("✅ Vector store update complete.")

    def load(self) -> None:
        """Loads the FAISS index from disk."""
        if self.vector_store:
            return
        if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()): # CORRECTED TYPO
            try:
                print(f"ℹ️ Loading vector store from {VECTOR_DIR}...")
                self.vector_store = FAISS.load_local(
                    str(VECTOR_DIR), self.embeddings, allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"❌ Could not load vector store: {e}")
                self.vector_store = None
        else:
            print("ℹ️ No existing vector store found.")
            self.vector_store = None

    def query(self, query_text: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Queries the vector store for the most relevant chunks."""
        self.load()
        if not self.vector_store:
            print("❌ Vector store is not available. Cannot query.")
            return []

        results_with_scores = self.vector_store.similarity_search_with_score(query_text, k=k, filter=filters)

        formatted_results = []
        for doc, score in results_with_scores:
            result = {
                "score": score,
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            formatted_results.append(result)

        print(f"🔍 Found {len(formatted_results)} results for your query.")
        return formatted_results

    def clear(self) -> None:
        """Deletes the vector store and processed files log."""
        if VECTOR_DIR.exists():
            shutil.rmtree(VECTOR_DIR)
            print(f"🗑️ Deleted vector store directory: {VECTOR_DIR}")
        if PROCESSED_FILES_LOG.exists():
            PROCESSED_FILES_LOG.unlink()
            print(f"🗑️ Deleted processed files log: {PROCESSED_FILES_LOG}")
        print("✨ Cleared all existing data. Ready for a fresh start.")


# ===========================================================
# Main Execution
# ===========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the sports news vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a rebuild by deleting the existing vector store and logs.",
    )
    args = parser.parse_args()

    manager = VectorStoreManager()

    if args.rebuild:
        manager.clear()

    # 1. Create or update the vector store
    manager.create_or_update(days_back=30)

    # 2. Example query
    print("\n" + "="*50)
    print("🚀 Running an example query...")
    print("="*50)

    query = "Ποια ομάδα είχε προβλήματα στην άμυνα στα τελευταία παιχνίδια;"
    search_results = manager.query(query, k=3)

    if search_results:
        for i, result in enumerate(search_results):
            print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
            print(f"  Title: {result['metadata'].get('title', 'N/A')}")
            print(f"  Source: {result['metadata'].get('source', 'N/A')}")
            print(f"  URL: {result['metadata'].get('url', 'N/A')}")
            print(f"  Content Snippet: {result['content'][:300].strip()}...")
    else:
        print("\n🤷 No results found for the example query.")