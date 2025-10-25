import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field  # <<< CORRECTED IMPORT

from vector_store import VectorStoreManager

# --- Configuration ---
load_dotenv()
# Directories
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / os.getenv("RAW_NEWS_DATA_DIR", "data/raw/news")

# LLM Models
LLM_MODEL = os.getenv("LLM_MODEL", "aya-expanse")
FACT_CHECKER_MODEL = os.getenv("FACT_CHECKER_MODEL", LLM_MODEL)  # Defaults to LLM_MODEL

# --- Pydantic Models for Structured Output ---
class ArticleSummary(BaseModel):
    """Data model for a structured article summary."""
    summary: str = Field(description="2-3 objective sentences capturing the key outcome and significance.")
    highlights: list[str] = Field(description="A list of 3-5 bullet points on standout performances or pivotal moments.")

class FactCheckResult(BaseModel):
    """Data model for the result of a fact-checking operation."""
    is_accurate: bool = Field(description="True if the summary is fully supported by the article, otherwise False.")
    reasoning: str = Field(description="A brief explanation of any inaccuracies, hallucinations, or omissions found. State 'Accurate' if no issues.")
    corrected_summary: ArticleSummary = Field(description="The corrected version of the summary. If the original was accurate, this will be identical.")

# --- Helper Functions ---
def get_llm(model_name=LLM_MODEL):
    """Initializes and returns the ChatOllama instance for general tasks."""
    return ChatOllama(model=model_name)

def get_fact_checker_llm():
    """Initializes and returns the ChatOllama instance for fact-checking."""
    print(f"  (Using fact-checker model: {FACT_CHECKER_MODEL})")
    return ChatOllama(model=FACT_CHECKER_MODEL)

class ArticleProcessor:
    def __init__(self):
        self.llm = get_llm()
        self.fact_checker_llm = get_fact_checker_llm()

    def get_initial_summary(self, content: str) -> dict:
        """Generates the first-pass summary of an article."""
        parser = JsonOutputParser(pydantic_object=ArticleSummary)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a sports analyst. Read the article and create a concise summary.
            Respond with a JSON object that strictly follows this format: {format_instructions}
            Base your analysis ONLY on the provided text. Do NOT invent facts. Do NOT reference 
            information outside the article. Do NOT include any introductory or concluding remarks.
            Do NOT change Player names or Team names.

            Article Content: {content}
            """,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        return chain.invoke({"content": content})

    def get_verified_summary(self, original_content: str, initial_summary: dict) -> dict:
        """Fact-checks and corrects the initial summary."""
        parser = JsonOutputParser(pydantic_object=FactCheckResult)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a meticulous fact-checker. Verify if the `Generated Summary` is factually consistent with the `Original Article`.
            Identify any "hallucinations" - details in the summary not supported by the article.
            Respond with a JSON object that strictly follows this format: {format_instructions}

            **Instructions**:
            1.  **Compare Carefully**: Cross-reference every claim in the summary against the original article.
            2.  **Determine Accuracy**: If every detail is supported, set `is_accurate` to `true`. Otherwise, `false`.
            3.  **Provide Reasoning**: Explain any inaccuracies found (e.g., wrong score, incorrect player stat). If accurate, state "Accurate."
            4.  **Correct the Summary**: Create a `corrected_summary`. If the original was accurate, this is an exact copy. If inaccurate, fix the mistakes to align perfectly with the article.

            **Original Article**:
            ```{original_content}```

            **Generated Summary (JSON)**:
            ```json
            {generated_summary}
            ```
            """,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.fact_checker_llm | parser
        return chain.invoke({
            "original_content": original_content,
            "generated_summary": json.dumps(initial_summary)
        })

    def process_and_update_file(self, file_path: Path):
        """Processes a single article file and updates it with summaries."""
        print(f"Processing: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        content = data.get("article", {}).get("content")
        if not content:
            print("  ⚠️ Skipping, no content found.")
            return

        # Step 1: Generate initial summary
        print("  - Generating initial summary...")
        initial_summary = self.get_initial_summary(content)

        # Step 2: Fact-check and correct the summary
        print("  - Fact-checking and correcting summary...")
        fact_check_result = self.get_verified_summary(content, initial_summary)

        # Step 3: Update the JSON data in memory
        data["llm_summary"] = initial_summary
        data["llm_summary_verified"] = fact_check_result

        # Step 4: Write the updated data back to the original file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Updated file with summaries: {file_path.name}")

    def evaluate_single_file(self, file_path: Path):
        """Runs the process on a single file and prints the comparison without saving."""
        print("-" * 50)
        print(f"🔍 Evaluating LLM Performance for: {file_path.name}")
        print("-" * 50)
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        content = data.get("article", {}).get("content")
        if not content:
            print("❌ Cannot evaluate: No content found in file.")
            return

        # Step 1: Generate initial summary
        print("\n[1] Generating Initial Summary...")
        initial_summary = self.get_initial_summary(content)
        print("\n--- INITIAL SUMMARY ---")
        print(json.dumps(initial_summary, indent=2, ensure_ascii=False))

        # Step 2: Fact-check and correct
        print("\n[2] Fact-Checking and Correcting...")
        fact_check_result = self.get_verified_summary(content, initial_summary)
        print("\n--- FACT-CHECK ANALYSIS ---")
        print(f"Accurate: {fact_check_result.get('is_accurate')}")
        print(f"Reasoning: {fact_check_result.get('reasoning')}")
        
        print("\n--- CORRECTED SUMMARY ---")
        print(json.dumps(fact_check_result.get('corrected_summary'), indent=2, ensure_ascii=False))
        print("-" * 50)


    def process_all_articles(self):
        """Processes every article in the raw directory."""
        print("\n--- Starting Full Article Processing ---")
        process_count = 0
        for article_file in RAW_DIR.rglob("*.json"):
            self.process_and_update_file(article_file)
            process_count += 1
        print(f"\n✅ Processed a total of {process_count} articles.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize, fact-check, and evaluate sports articles.")
    parser.add_argument("--file", type=Path, help="Path to a single article JSON file to evaluate.")
    parser.add_argument("--all", action="store_true", help="Process all articles in the raw directory.")
    args = parser.parse_args()

    processor = ArticleProcessor()

    if args.file:
        if args.file.exists():
            processor.evaluate_single_file(args.file)
        else:
            print(f"❌ Error: File not found at '{args.file}'")
    elif args.all:
        processor.process_all_articles()
    else:
        print("❌ Error: Please specify either --file or --all.")