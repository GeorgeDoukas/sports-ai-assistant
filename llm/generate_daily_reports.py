import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm.llm_services import LANGUAGE, get_llm
from storage.vector_store import VectorStoreManager

# --- Configuration ---
load_dotenv()

# Directories
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / os.getenv("RAW_NEWS_DATA_DIR", "data/raw/news")
REPORTS_BASE_DIR = BASE_DIR / "data" / "reports" / "news"
REPORTS_BASE_DIR.mkdir(parents=True, exist_ok=True)


class ReportGenerator:
    """
    Generates structured daily and competition-level reports based on flexible command-line arguments.
    """

    def __init__(self, args):
        self.args = args
        print(f"ℹ️  Initializing ReportGenerator with task arguments...")
        self.llm = get_llm()
        self.vs_manager = VectorStoreManager()
        self.workload = self._load_and_filter_articles()

    def _load_and_filter_articles(self) -> dict:
        """
        Loads all processed articles and filters them based on command-line arguments.
        Returns a dictionary grouped by (sport, competition, date) for the specified workload.
        """
        print("🔎 Loading and filtering articles...")
        grouped_articles = defaultdict(list)
        all_files = list(RAW_DIR.rglob("*.json"))

        for article_file in all_files:
            try:
                with open(article_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("processing_status") != "processed":
                    continue

                # --- Filtering Logic ---
                sport = data.get("sport")
                comp = data.get("competition")
                # Normalize date format for reliable matching
                date_folder = (
                    data.get("article", {}).get("date_published", "").replace(" ", "-")
                )

                if self.args.sport and sport != self.args.sport:
                    continue
                if self.args.competition and comp != self.args.competition:
                    continue
                if self.args.date and date_folder != self.args.date.replace(" ", "-"):
                    continue

                grouped_articles[(sport, comp, date_folder)].append(data)
            except Exception as e:
                print(f"⚠️ Could not process file {article_file.name}: {e}")

        print(
            f"✅ Found {len(grouped_articles)} unique competition-dates matching filter criteria."
        )
        return grouped_articles

    def _generate_markdown_report(self, prompt_template: str, context: dict) -> str:
        """Helper function to invoke the LLM chain and return a markdown report."""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        try:
            context["language"] = LANGUAGE
            return chain.invoke(context)
        except Exception as e:
            print(f"  ❌ LLM Error: {e}")
            return f"# LLM Generation Error\n\nAn error occurred: {e}"

    def _get_content_from_summaries(self, articles: list) -> str:
        """Prepares a JSON string of summaries for the prompt context."""
        summaries_list = [
            {
                "source": article.get("source"),
                "title": article.get("article", {}).get("title"),
                "summary": article.get("summary"),
                "highlights": article.get("highlights", []),
            }
            for article in articles
        ]
        return json.dumps(summaries_list, indent=2, ensure_ascii=False)

    def _get_content_from_vectorstore(self, articles: list) -> str:
        """Fetches and concatenates full article content for the prompt context."""
        print("  - Fetching full content from source files...")
        full_contents = []
        for article_data in articles:
            # You should ensure 'file_path' is saved in your JSON metadata during scraping for this to be robust.
            file_path = article_data.get("file_path")
            if file_path:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                        full_contents.append(
                            raw_data.get("article", {}).get("content", "")
                        )
                except FileNotFoundError:
                    full_contents.append(
                        article_data.get("article", {}).get("content", "")
                    )  # Fallback
            else:
                full_contents.append(article_data.get("article", {}).get("content", ""))
        return "\n\n--- ARTICLE SEPARATOR ---\n\n".join(filter(None, full_contents))

    def run(self):
        """
        Main execution loop that generates reports based on the loaded workload.
        """
        if not self.workload:
            print("ℹ️ No articles match the specified criteria. Nothing to do.")
            return

        for (sport, comp, date), articles in self.workload.items():
            # Define output directory for this specific group
            output_dir = REPORTS_BASE_DIR / sport / comp / date
            output_dir.mkdir(parents=True, exist_ok=True)

            # --- 1. Generate Daily Report for Each Source ---
            articles_by_source = defaultdict(list)
            for article in articles:
                articles_by_source[article.get("source", "Unknown")].append(article)

            for source, source_articles in articles_by_source.items():
                report_path = output_dir / f"daily_report_{source}.md"
                # Smart Skip Logic: Skip if using --all and file exists.
                if self.args.all and report_path.exists():
                    print(f"⏭️ Skipping existing daily report: {report_path.name}")
                    continue

                print(f"📅 Generating daily report for {source} on {date}...")
                content_for_llm = (
                    self._get_content_from_summaries(source_articles)
                    if self.args.method == "summaries"
                    else self._get_content_from_vectorstore(source_articles)
                )

                # <<< --- IMPROVED PROMPT 1: Daily Source Report --- >>>
                prompt_template = """
                You are an elite sports journalist and editor. Your entire response MUST be in {language}.
                Your task is to compile a daily digest for **{date}** from the news source **{source}**.
                Your task is to read the following `Provided Context` and produce a final, verified, and comprehensive summary.
                You must perform all steps internally—analysis, summarization, and fact-checking—before producing a single, perfect report Markdown output.
                

                **Your Internal Thought Process (Don't write this in the output, just do it):**
                1.  **Identify the main story**: Read through all the provided context. What is the single most important event or result?
                2.  **Find supporting details**: What are the key statistics or performances that support the main story?
                3.  **Draft a narrative**: Mentally structure the report with a strong opening headline, followed by the supporting details in a logical flow.
                4.  **Self-Correction**: Is your draft a true synthesis, or just a list of the inputs? Ensure you are creating a cohesive narrative. Is everything 100% factually supported by the context? Did you miss anything important? Fix any mistakes and add any omissions.

                **Final Report Structure**:
                - **Top Headlines**: A paragraph summarizing the most significant results and news from this source.
                - **Key Performances**: Bullet points highlighting standout player performances mentioned.
                - **Preserve Names**: You MUST NOT translate proper nouns (player/team names). Keep them as they appear in the original article.
                
                **Provided Context from {source}:**
                ```{context}```
                """
                report_content = self._generate_markdown_report(
                    prompt_template,
                    {"date": date, "source": source, "context": content_for_llm},
                )
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
                print(f"  ✅ Saved daily report: {report_path.name}")

            # --- 2. Generate Combined Report for the Date ---
            combined_report_path = output_dir / "daily_summary_report.md"
            if self.args.all and combined_report_path.exists():
                print(
                    f"⏭️ Skipping existing combined report: {combined_report_path.name}"
                )
                continue

            print(f"📈 Generating combined summary for {comp} on {date}...")
            content_for_llm = (
                self._get_content_from_summaries(articles)
                if self.args.method == "summaries"
                else self._get_content_from_vectorstore(articles)
            )

            # <<< --- IMPROVED PROMPT 2: Combined Competition Report --- >>>
            prompt_template = """
            You are a senior sports analyst, writing in {language}. Your task is to create a single, high-level summary for the **{competition}** competition on **{date}**.
            You will be given context from multiple news sources. Your goal is to synthesize them into a single, cohesive narrative in **Markdown format**.

            **Your Internal Thought Process (Perform these steps before writing):**
            1.  **Identify the overarching theme**: After reading all context, what is the most important, agreed-upon story of the day for this competition? (e.g., a major upset, a dominant team performance).
            2.  **Consolidate key facts**: Extract the most critical statistics and performances. If sources report the same fact, you only need to state it once. If they conflict, note the discrepancy if it's significant.
            3.  **Structure the master narrative**: Plan the report. Start with the main theme, then provide the consolidated highlights as evidence.
            4.  **Self-Correction**: Review your mental draft. Does it accurately reflect the consensus of the sources? Is it a true synthesis, or just a collection of separate points? Ensure the narrative flows logically.

            **Final Report Structure**:
            - **Overall Summary**: A main paragraph that combines the key events from all sources into a single narrative.
            - **Consolidated Highlights**: A single, unified list of bullet points with the most impressive performances found across all sources.

            **Provided Context from all sources:**
            ```{context}```
            """
            report_content = self._generate_markdown_report(
                prompt_template,
                {"date": date, "competition": comp, "context": content_for_llm},
            )
            with open(combined_report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"  ✅ Saved combined report: {combined_report_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate flexible daily and combined sports reports."
    )

    # --- Filter Arguments ---
    parser.add_argument(
        "--date",
        type=str,
        help="Specify a single date to process (e.g., '25-Οκτωβρίου-2025'). Overrides existing reports for this date.",
    )
    parser.add_argument(
        "--sport", type=str, help="Filter by a specific sport (e.g., 'basketball')."
    )
    parser.add_argument(
        "--competition",
        type=str,
        help="Filter by a specific competition (e.g., 'euroleague').",
    )

    # --- Mode Arguments ---
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all dates found. Skips existing reports.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["summaries", "vectorstore"],
        default="summaries",
        help="Choose the data source method: 'summaries' (fast, from JSON) or 'vectorstore' (slower, from full content).",
    )
    args = parser.parse_args()

    # --- Argument Validation ---
    if not args.date and not args.all:
        parser.error("You must specify either --date OR --all.")
    if args.date and args.all:
        parser.error("You cannot use --date and --all at the same time.")

    generator = ReportGenerator(args)
    generator.run()

    print("\n✅ All specified generation tasks complete.")
