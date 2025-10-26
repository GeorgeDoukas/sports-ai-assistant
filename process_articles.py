import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from vector_store import VectorStoreManager

# --- Configuration ---
load_dotenv()
# Directories
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / os.getenv("RAW_NEWS_DATA_DIR", "data/raw/news")

# LLM Models
LLM_MODEL = os.getenv("LLM_MODEL", "aya-expanse")
FACT_CHECKER_MODEL = os.getenv("FACT_CHECKER_MODEL", LLM_MODEL)

# Language Configuration
LANGUAGE = os.getenv("LANGUAGE", "English")


# --- Pydantic Models for Structured Output ---
class ArticleSummary(BaseModel):
    """Data model for a structured article summary."""

    summary: str = Field(
        description="2-3 objective sentences capturing the key outcome and significance."
    )
    highlights: list[str] = Field(
        description="A list of 3-5 bullet points on standout performances or pivotal moments."
    )


class FactCheckResult(BaseModel):
    """Data model for the result of a fact-checking operation."""

    is_accurate: bool = Field(
        description="True only if the summary is fully accurate AND complete, otherwise False."
    )
    reasoning: str = Field(
        description="A brief explanation of all inaccuracies or omissions found. State 'Accurate and Complete' if no issues."
    )
    corrected_summary_text: str = Field(
        description="The definitive, corrected version of the prose summary text ONLY."
    )
    corrected_highlights: list[str] = Field(
        description="The final, definitive list of highlights, incorporating corrections and additions."
    )


# --- Helper Functions ---
def get_llm(model_name=LLM_MODEL):
    """Initializes and returns the ChatOllama instance for general tasks."""
    return ChatOllama(model=model_name)


def get_fact_checker_llm():
    """Initializes and returns the ChatOllama instance for fact-checking."""
    print(f"  (Using fact-checker model: {FACT_CHECKER_MODEL})")
    return ChatOllama(model=FACT_CHECKER_MODEL)


class ArticleProcessor:
    def __init__(self, language: str):
        self.llm = get_llm()
        self.fact_checker_llm = get_fact_checker_llm()
        self.language = language
        print(f"✅ Initialized ArticleProcessor for language: {self.language}")

    def get_initial_summary(self, content: str) -> dict:
        """Generates the first-pass summary of an article in the specified language."""
        parser = JsonOutputParser(pydantic_object=ArticleSummary)

        prompt = ChatPromptTemplate.from_template(
            """
            You are a seasoned sports journalist. Your entire response MUST be in {language}.
            Your goal is to create a comprehensive yet readable summary of the article below.

            **Your Task**:
            Respond with a JSON object that strictly follows this format: {format_instructions}

            **Key Guidelines**:
            1.  **Write a Rich Summary**: Create an objective summary between 3 and 6 sentences. The length should be proportional to the detail in the article. A long, detailed game report warrants a longer summary. It should cover the main outcome, key context, and any significant implications (e.g., playoff chances, player milestones).
            2.  **Extract Comprehensive Highlights**: Identify between 4 and 10 key highlights. The number should reflect the richness of the article; a detailed report may have many highlights, while a brief news update might only have a few. Do not force highlights if they aren't present. Good highlights include:
                - Definitive statistics (top scorers, key percentages).
                - Game-changing plays or pivotal moments.
                - Important direct quotes or post-game reactions.
                - Significant records broken or milestones achieved.
                - Key injury updates mentioned in the text.
            3.  **Fundamental Rule on Names**: **Crucially, you MUST NOT translate or transliterate proper nouns** like player names, team names, or league names. Keep them in their original language as written in the article. For example, if the article says "LeBron James", your summary must also say "LeBron James", not a translated version.

            Article Content:
            {content}
            """,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        return chain.invoke({"content": content, "language": self.language})

    def get_verified_summary(
        self, original_content: str, initial_summary: dict
    ) -> dict:
        """Fact-checks, verifies completeness, and corrects the initial summary."""
        parser = JsonOutputParser(pydantic_object=FactCheckResult)

        prompt = ChatPromptTemplate.from_template(
            """
            You are the Editor-in-Chief at a major sports news agency. Your entire response MUST be in {language}.
            Your task is to meticulously review a `Generated Summary` from a junior journalist against the `Original Article`.
            Your goal is to produce the final, publishable version by ensuring it is both **100% accurate** and **comprehensively complete**.
            Respond with a JSON object that strictly follows this format: {format_instructions}

            **Your Editorial 5-Step Process**:
            1.  **Fact-Check Existing Content**: Meticulously verify every statement in the `summary` text and every point in the `highlights` list. Note all factual errors.
            2.  **Identify Missing Highlights**: Re-read the `Original Article` with a critical eye. Did the junior journalist miss any crucial highlights? Look for top scorer stats, game-winning plays, significant quotes, or injury updates.
            3.  **Formulate the Final Verdict**: Set `is_accurate` to `true` ONLY if the original summary was **both 100% factually correct AND contained all major highlights**. If there are any factual errors OR any important omissions, you must set it to `false`.
            4.  **Provide Comprehensive Reasoning**: In the `reasoning` field (in {language}), explain your verdict. Describe ALL issues found (e.g., "Incorrect final score in the summary, and the highlights missed Player X's record-breaking goal."). If there are no issues, state the equivalent of "Accurate and Complete" in {language}.
            5.  **Construct the Final Output**: This is the most important step. Populate the final JSON object with the corrected content:
                - **`corrected_summary_text`**: This field must contain ONLY the corrected prose summary text, written in {language}.
                - **`corrected_highlights`**: This field is critical. It must be a **new, complete list of strings** that includes: (a) all accurate highlights from the original summary, (b) your corrections to any inaccurate highlights, and (c) any important highlights you discovered that were missing. This is the final, definitive set of highlights.
                - **Preserve Names**: In all text you generate, you **MUST NOT translate or transliterate proper nouns** (player/team names). Keep them exactly as they appear in the original article.

            **Original Article**:
            ```{original_content}```

            **Generated Summary (JSON from junior journalist)**:
            ```json
            {generated_summary}
            ```
            """,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.fact_checker_llm | parser
        return chain.invoke(
            {
                "original_content": original_content,
                "generated_summary": json.dumps(initial_summary, ensure_ascii=False),
                "language": self.language,
            }
        )

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
        data["llm_summary_verified"] = {
            "is_accurate": fact_check_result.get("is_accurate"),
            "reasoning": fact_check_result.get("reasoning"),
            "summary": fact_check_result.get(
                "corrected_summary_text"
            ),  # Use the corrected text
            "highlights": fact_check_result.get(
                "corrected_highlights"
            ),  # Use the corrected highlights
        }

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
        print(f"\n[1] Generating Initial Summary in {self.language}...")
        initial_summary = self.get_initial_summary(content)
        print("\n--- INITIAL SUMMARY ---")
        print(json.dumps(initial_summary, indent=2, ensure_ascii=False))

        # Step 2: Fact-check and correct
        print(f"\n[2] Fact-Checking and Correcting in {self.language}...")
        fact_check_result = self.get_verified_summary(content, initial_summary)

        print("\n--- FACT-CHECK ANALYSIS ---")
        print(json.dumps(fact_check_result, indent=2, ensure_ascii=False))
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
    parser = argparse.ArgumentParser(
        description="Summarize, fact-check, and evaluate sports articles."
    )
    parser.add_argument(
        "--file", type=Path, help="Path to a single article JSON file to evaluate."
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all articles in the raw directory."
    )
    args = parser.parse_args()

    # Pass the language from .env to the processor
    processor = ArticleProcessor(language=LANGUAGE)

    if args.file:
        if args.file.exists():
            processor.evaluate_single_file(args.file)
        else:
            print(f"❌ Error: File not found at '{args.file}'")
    elif args.all:
        processor.process_all_articles()
    else:
        print("❌ Error: Please specify either --file or --all.")
