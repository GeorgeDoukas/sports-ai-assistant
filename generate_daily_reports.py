import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Import from our shared module ---
from llm_services import get_llm, LANGUAGE
from vector_store import VectorStoreManager

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
                date_folder = data.get("article", {}).get("date_published", "").replace(' ', '-')

                if self.args.sport and sport != self.args.sport:
                    continue
                if self.args.competition and comp != self.args.competition:
                    continue
                if self.args.date and date_folder != self.args.date.replace(' ', '-'):
                    continue
                
                grouped_articles[(sport, comp, date_folder)].append(data)
            except Exception as e:
                print(f"⚠️ Could not process file {article_file.name}: {e}")
        
        print(f"✅ Found {len(grouped_articles)} unique competition-dates matching filter criteria.")
        return grouped_articles

    def _generate_markdown_report(self, prompt_template: str, context: dict) -> str:
        """Helper function to invoke the LLM chain and return a markdown report."""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        try:
            context['language'] = LANGUAGE
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
                "highlights": article.get("highlights", [])
            } for article in articles
        ]
        return json.dumps(summaries_list, indent=2, ensure_ascii=False)

    def _get_content_from_vectorstore(self, articles: list) -> str:
        """Fetches and concatenates full article content for the prompt context."""
        print("  - Fetching full content from source files...")
        full_contents = []
        for article_data in articles:
            file_path = article_data.get("article", {}).get("url") # Assuming file path is stored in url or similar
            # A more robust way would be to query vector store if needed, but reading from file is direct
            try:
                 with open(article_data.get("file_path"), "r", encoding="utf-8") as f: # You need to ensure file_path is in your JSON
                     raw_data = json.load(f)
                     full_contents.append(raw_data.get("article", {}).get("content", ""))
            except:
                 # Fallback to what's in the current data if file_path is missing
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
                
                content_for_llm = ""
                if self.args.method == 'summaries':
                    content_for_llm = self._get_content_from_summaries(source_articles)
                else: # from-vectorstore
                    content_for_llm = self._get_content_from_vectorstore(source_articles)

                prompt_template = """
                You are a sports editor, writing in {language}, creating a daily report for **{date}** from news source **{source}**.
                Based on the provided context below, write a concise report in **Markdown format**.
                Structure it with a 'Top Headlines' paragraph and 'Key Performances' bullet points.
                Synthesize the information; do not just list the original points.

                **Provided Context:**
                ```{context}```
                """
                report_content = self._generate_markdown_report(
                    prompt_template,
                    {"date": date, "source": source, "context": content_for_llm}
                )
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
                print(f"  ✅ Saved daily report: {report_path.name}")

            # --- 2. Generate Combined Report for the Date ---
            combined_report_path = output_dir / "daily_summary_report.md"
            if self.args.all and combined_report_path.exists():
                print(f"⏭️ Skipping existing combined report: {combined_report_path.name}")
                continue

            print(f"📈 Generating combined summary for {comp} on {date}...")
            
            content_for_llm = ""
            if self.args.method == 'summaries':
                content_for_llm = self._get_content_from_summaries(articles)
            else: # from-vectorstore
                content_for_llm = self._get_content_from_vectorstore(articles)

            prompt_template = """
            You are a senior sports analyst, writing in {language}. Your task is to create a single, high-level summary for the **{competition}** competition on **{date}**.
            Read all the provided context from different news sources below. Synthesize them into a single, cohesive narrative in **Markdown format**.

            **Your Goal**:
            Produce a unified overview that answers: What were the most important stories and outcomes for this competition on this day, according to all available sources?
            Structure the report with an 'Overall Summary' paragraph and a 'Consolidated Highlights' list.

            **Provided Context from all sources:**
            ```{context}```
            """
            report_content = self._generate_markdown_report(
                prompt_template,
                {"date": date, "competition": comp, "context": content_for_llm}
            )
            with open(combined_report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"  ✅ Saved combined report: {combined_report_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate flexible daily and combined sports reports.")
    
    # --- Filter Arguments ---
    parser.add_argument("--date", type=str, help="Specify a single date to process (e.g., '25-Οκτωβρίου-2025'). Overrides existing reports for this date.")
    parser.add_argument("--sport", type=str, help="Filter by a specific sport (e.g., 'basketball').")
    parser.add_argument("--competition", type=str, help="Filter by a specific competition (e.g., 'euroleague').")
    
    # --- Mode Arguments ---
    parser.add_argument("--all", action="store_true", help="Process all dates found. Skips existing reports.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["summaries", "vectorstore"],
        default="summaries",
        help="Choose the data source method: 'summaries' (fast, from JSON) or 'vectorstore' (slower, from full content)."
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