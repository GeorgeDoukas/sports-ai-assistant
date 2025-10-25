import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
load_dotenv()

# Directories
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / os.getenv("RAW_NEWS_DATA_DIR", "data/raw/news")
DAILY_REPORTS_DIR = BASE_DIR / os.getenv("DAILY_REPORTS_DIR", "data/daily_reports")
# Ensure the directory exists
DAILY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# LLM Model
LLM_MODEL = os.getenv("LLM_MODEL", "aya-expanse")

# --- Helper Functions ---
def get_llm(model_name=LLM_MODEL):
    """Initializes and returns the ChatOllama instance for general tasks."""
    return ChatOllama(model=model_name)

def run_report_generation():
    """
    Aggregates verified summaries from raw article files into daily reports, organized by source.
    """
    print("\n--- Starting Daily Report Generation ---")
    llm = get_llm()
    reports_generated = 0

    # 1. Group summaries by date and source
    summaries_by_date_source = defaultdict(lambda: defaultdict(list))

    for article_file in RAW_DIR.rglob("*.json"):
        try:
            with open(article_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Use the verified summary and original metadata
            verified_summary = data.get("llm_summary_verified", {}).get("corrected_summary")
            if not verified_summary:
                continue  # Skip if no verified summary

            article_meta = data.get("article", {})
            date_str = article_meta.get("date_published", "").split(" ")[0]
            source = data.get("source", "Unknown")

            if not date_str:
                continue

            # Prepare a clean object for the report
            report_entry = {
                "title": article_meta.get("title"),
                "competition": data.get("competition"),
                "summary": verified_summary.get("summary"),
                "highlights": verified_summary.get("highlights")
            }
            summaries_by_date_source[date_str][source].append(report_entry)

        except Exception as e:
            print(f"⚠️ Could not process file {article_file.name}: {e}")

    if not summaries_by_date_source:
        print("ℹ️ No new verified summaries found to generate reports.")
        return

    # 2. Generate reports for each date and source
    prompt = ChatPromptTemplate.from_template(
        """
        You are a sports journalist compiling a daily digest for {date} from {source}.
        Using the provided JSON summaries, write a concise daily report in **Markdown format**.

        **Structure**:
        1.  **Top Headlines**: A paragraph summarizing the most significant results.
        2.  **Key Performances**: Bullet points on standout player performances.

        **Instructions**:
        - Only use facts from the summaries. Do NOT add outside information.
        - Format the output clearly with Markdown headings.

        **Verified Summaries for {date} from {source}:**
        ```json
        {summaries}
        ```
        """
    )
    chain = prompt | llm | StrOutputParser()

    for date, source_data in summaries_by_date_source.items():
        for source, summaries in source_data.items():
            print(f"📅 Generating report for {date} from {source}...")
            report_path = DAILY_REPORTS_DIR / f"report_{date.replace('/', '-')}_{source}.md"

            try:
                response = chain.invoke({
                    "date": date,
                    "source": source,
                    "summaries": json.dumps(summaries, indent=2, ensure_ascii=False)
                })
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(response)
                print(f"  ✅ Report saved to {report_path}")
                reports_generated += 1
            except Exception as e:
                print(f"  ❌ Failed to generate report for {date} from {source}: {e}")

    print(f"\n✅ Daily report generation complete. Generated {reports_generated} reports.")

if __name__ == "__main__":
    run_report_generation()