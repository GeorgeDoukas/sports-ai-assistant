import argparse
import json
import os
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from vector_store import VectorStoreManager

# --- Configuration ---
load_dotenv()
# Directories
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / os.getenv("RAW_NEWS_DATA_DIR", "data/raw/news")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
# LLM Models
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b-it-qat")
FACT_CHECKER_MODEL = os.getenv("FACT_CHECKER_MODEL", LLM_MODEL)

# Language and Performance
LANGUAGE = os.getenv("LANGUAGE", "English")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))


# --- Pydantic Models for Structured Output ---
class ArticleSummary(BaseModel):
    """Data model for a final, verified summary created in a single pass."""
    summary: str = Field(description="The final, objective summary of 3-6 sentences.")
    highlights: list[str] = Field(description="The final, complete list of 4-10 key highlights.")

# --- Helper Functions ---
def get_llm():
    """
    Initializes and returns the correct LLM provider based on the .env file.
    
    Args:
        model_type (str): "main" for the primary model, "fact_checker" for the verifier.
    """
    provider = LLM_PROVIDER.lower()
    print(f"ℹ️  Initializing LLM using provider: {provider}")

    if provider == "openai_compatible":
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
    elif provider == "ollama":
        return ChatOllama(model=LLM_MODEL)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}.")


class ArticleProcessor:
    def __init__(self, language: str):
        self.language = language
        print(f"✅ Initialized ArticleProcessor for language: {self.language} with {MAX_WORKERS} workers.")

    def _summarize_content(self, content: str, llm_client) -> dict:
        parser = JsonOutputParser(pydantic_object=ArticleSummary)
        prompt = ChatPromptTemplate.from_template(
                """
                You are an elite sports journalist and editor. Your entire response MUST be in {language}.
                Your task is to read the following `Original Article` and produce a final, verified, and comprehensive summary.
                You must perform all steps internally—analysis, summarization, and fact-checking—before producing a single, perfect JSON output.
                The JSON must strictly follow this format: {format_instructions}

                **Your Internal Thought Process (Don't write this in the output, just do it):**
                1.  Read the article to understand the main outcome, key players, and statistics.
                2.  Mentally draft a 3-6 sentence summary.
                3.  Mentally identify 4-10 of the most crucial highlights (top stats, game-winners, key injuries, etc.).
                4.  **Self-Correction:** Reread your draft summary and highlights. Are they 100% supported by the article? Did you miss anything important? Fix any mistakes and add any omissions.

                **Final Output Instructions**:
                - **`summary`**: The final, corrected prose summary.
                - **`highlights`**: The final, comprehensive list of highlights.
                - **Preserve Names**: You MUST NOT translate proper nouns (player/team names). Keep them as they appear in the original article.

                **Original Article**:
                ```{content}```
                """,
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
        chain = prompt | llm_client | parser
        return chain.invoke({"content": content, "language": self.language})

    def _process_file(self, file_path: Path, llm_client):
        """Processes a single article file using the provided LLM clients."""
        try:
            print(f"Processing: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)

            content = data.get("article", {}).get("content")
            if not content:
                print(f"  ⚠️ Skipping {file_path.name}, no content found.")
                return
            # Step 1: Generate summary
            print("  - Generating summary...")
            summary_and_highlights = self._summarize_content(content, llm_client)

            # Step 2: Update the JSON data in memory
            data["summary"] = summary_and_highlights.get("summary")
            data["highlights"] = summary_and_highlights.get("highlights")
            data["processing_status"] = "processed"

            # Step 3: Write the updated data back to the original file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Updated: {file_path.name}")
        except Exception as e:
            print(f"  ❌ FAILED to process {file_path.name}: {e}")

    def _process_chunk(self, file_chunk: list[Path]):
        """Worker function that processes a specific list (chunk) of files."""
        # Create LLM clients ONCE for this thread/worker.
        llm_client = get_llm()
        
        print(f"Worker started, processing a chunk of {len(file_chunk)} files.")
        for file_path in file_chunk:
            self._process_file(file_path, llm_client)

    def evaluate_single_file(self, file_path: Path):
        """Runs the process on a single file for evaluation."""
        llm_client = get_llm()

        print("-" * 50)
        print(f"🔍 Evaluating LLM Performance for: {file_path.name}")
        print("-" * 50)

        with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
        content = data.get("article", {}).get("content")
        if not content:
            print("❌ Cannot evaluate: No content found in file.")
            return
        # Step 1: Generate summary
        print(f"\n[1] Generating Summary in {self.language}...")
        summary = self._summarize_content(content, llm_client)
        print("\n--- SUMMARY ---")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        print("-" * 50)

    def process_all_articles_in_parallel(self):
        """
        Finds unprocessed articles, manually partitions them into chunks,
        and processes them in parallel to prevent conflicts.
        """
        print("\n--- Starting Full Article Processing (Parallel) ---")

        # Step 1: Use VectorStoreManager to get a list of all known article file paths.
        vs_manager = VectorStoreManager()
        vs_manager.load()
        if not vs_manager.vector_store:
            print("❌ Vector store not found. Cannot determine which articles to process.")
            return

        all_known_files = [Path(doc.metadata["file_path"]) for doc in vs_manager.vector_store.docstore._dict.values() if "file_path" in doc.metadata]
        # Step 2: Create a to-do list of files that need summarization.
        files_to_process = []
        for file_path in all_known_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
                if "summary" not in data:
                    files_to_process.append(file_path)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"⚠️ Skipping file from vector store index due to error: {file_path} ({e})")

        if not files_to_process:
            print("✅ All articles in the vector store are already summarized. Nothing to do.")
            return

        print(f"Found {len(files_to_process)} articles needing summarization.")

        # Step 3: Determine number of workers and split work into chunks.
        if len(files_to_process) < MAX_WORKERS:
            num_workers = len(files_to_process)
        else:
            num_workers = MAX_WORKERS
        
        if num_workers == 0:
            return

        # Split the list of files into chunks for each worker
        chunk_size = math.ceil(len(files_to_process) / num_workers)
        chunks = [
            files_to_process[i : i + chunk_size]
            for i in range(0, len(files_to_process), chunk_size)
        ]
        
        print(f"Splitting work into {len(chunks)} chunks for {num_workers} workers.")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit each chunk to the new worker function
            executor.map(self._process_chunk, chunks)

        print(f"\n✅ Parallel processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize, fact-check, and evaluate sports articles.")
    parser.add_argument("--file", type=Path, help="Path to a single article JSON file to evaluate.")
    parser.add_argument("--all", action="store_true", help="Process all articles in the raw directory using parallel processing.")
    args = parser.parse_args()

    processor = ArticleProcessor(language=LANGUAGE)

    if args.file:
        if args.file.exists():
            processor.evaluate_single_file(args.file)
        else:
            print(f"❌ Error: File not found at '{args.file}'")
    elif args.all:
        processor.process_all_articles_in_parallel()
    else:
        print("❌ Error: Please specify either --file or --all.")