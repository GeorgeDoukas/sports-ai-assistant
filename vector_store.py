# vector_store.py
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from pathlib import Path
import json
from datetime import datetime, timedelta

VECTOR_DIR = "data/vectorstore/faiss"
RAW_DIR = Path("data/raw")


def load_raw_news_from_sources(sources=None, days_back=30):
    """Load all article JSONs from data/raw/** recursively."""
    cutoff = datetime.now() - timedelta(days=days_back)
    docs = []

    for file in RAW_DIR.rglob("*.json"):
        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)

            article = data.get("article", {})
            if not article:
                continue

            source = data.get("source", "unknown")
            competition = data.get("competition", "unknown")
            scraped_at = data.get("scraped_at", None)

            if scraped_at:
                try:
                    scraped_date = datetime.fromisoformat(scraped_at)
                    if scraped_date < cutoff:
                        continue
                except Exception:
                    pass

            if sources and source not in sources:
                continue

            title = article.get("title", "")
            content = article.get("content", "")
            if not content.strip():
                continue

            full_text = f"{title}\n\n{content}".strip()
            doc = Document(
                page_content=full_text,
                metadata={
                    "title": title,
                    "author": article.get("author", ""),
                    "published": article.get("date_published", ""),
                    "url": article.get("url", ""),
                    "competition": competition,
                    "source": source,
                    "scraped_at": scraped_at,
                    "file": str(file),
                },
            )
            docs.append(doc)

        except Exception as e:
            print(f"⚠️ Skip {file}: {e}")

    print(f"📚 Loaded {len(docs)} articles into memory.")
    return docs


def load_vectorstore():
    """Load FAISS vector store from local disk, if it exists."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    try:
        return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        print("ℹ️ No existing vector store found, creating new one.")
        return None


def create_or_update_vectorstore(sources=None, days_back=30):
    """
    Incrementally build or update the FAISS vector store.

    Only new article files (not yet embedded) are added.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = load_vectorstore()

    # Load existing file paths to skip
    existing_files = set()
    if vectorstore and hasattr(vectorstore, "docstore"):
        for _, doc in vectorstore.docstore._dict.items():
            if "file" in doc.metadata:
                existing_files.add(doc.metadata["file"])

    all_docs = load_raw_news_from_sources(sources, days_back)
    new_docs = [doc for doc in all_docs if doc.metadata["file"] not in existing_files]

    if not new_docs:
        print("✅ No new documents found. Vectorstore is up to date.")
        return vectorstore

    print(f"🆕 Found {len(new_docs)} new documents to embed...")

    if vectorstore:
        vectorstore.add_documents(new_docs)
    else:
        vectorstore = FAISS.from_documents(new_docs, embeddings)

    Path(VECTOR_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(VECTOR_DIR)
    print(f"✅ Vectorstore updated (total {len(all_docs)} docs).")
    return vectorstore


def query_vectorstore(query: str, k: int = 5, include_content: bool = False):
    """
    Search the FAISS index for the most relevant articles.

    Args:
        query (str): The search query
        k (int): Number of top results
        include_content (bool): Whether to include full text content in results

    Returns:
        list[dict]: List of search results with metadata and optional content
    """
    vectorstore = load_vectorstore()
    if not vectorstore:
        print("❌ No vector store found. Run create_or_update_vectorstore() first.")
        return []

    results = vectorstore.similarity_search_with_score(query, k=k)
    formatted = []

    for doc, score in results:
        item = {
            "title": doc.metadata.get("title", ""),
            "competition": doc.metadata.get("competition", ""),
            "author": doc.metadata.get("author", ""),
            "published": doc.metadata.get("published", ""),
            "source": doc.metadata.get("source", ""),
            "url": doc.metadata.get("url", ""),
            "score": score,
        }
        if include_content:
            item["content"] = doc.page_content
        formatted.append(item)

    print(f"🔍 Found {len(formatted)} results for query: '{query}'")
    for r in formatted:
        print(f" - {r['title']} ({r['competition']}) [{r['score']:.4f}]")

    return formatted


if __name__ == "__main__":
    # 1️⃣ Build or incrementally update FAISS store
    create_or_update_vectorstore()

    # 2️⃣ Query it
    results = query_vectorstore("Ανάληση για Euroleague", include_content=True)
    print("\n🧾 Example result snippet:\n", results[0]["content"][:300] if results else "No results")
