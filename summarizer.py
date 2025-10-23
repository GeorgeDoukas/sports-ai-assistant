# summarizer.py
import os
import re
import json
from datetime import datetime
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from vector_store import load_vectorstore, query_vectorstore

load_dotenv()

RAW_DIR = Path("data/raw")
SUMMARY_DIR = Path("data/summaries")
REPORT_DIR = Path("data/reports")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using LLM model: {os.getenv('LLM_MODEL', 'qwen3-4b')}")


def get_llm(model=os.getenv("LLM_MODEL", "qwen3-4b")):
    return ChatOllama(model=model)


def remove_think_section(text: str) -> str:
    """
    Removes all <think>...</think> sections from the input text.

    Args:
        text (str): The input string, potentially containing <think> tags.

    Returns:
        str: The text with all <think>...</think> sections removed.
    """
    # Use a non-greedy regex to match everything between <think> and </think>
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def summarize_article(article: dict, llm=None):
    """Summarize article with bullet points highlights."""
    if llm is None:
        llm = get_llm()

    prompt_template = """
You are a professional sports analyst with expertise in delivering concise, data-informed insights.  
Carefully read the article below and provide your analysis in **Markdown format** with the following structure:

- A **Summary** section containing 2–3 clear, objective sentences that capture the key outcome, context, and significance of the event.  
- A **Key Highlights** section with 3–5 bullet points focusing on standout performances, pivotal moments, relevant statistics, or strategic decisions.

⚠️ **Critical Instructions**:  
- **Do NOT invent scores, stats, player names, teams, or events**. If a category (e.g., injuries) isn’t mentioned in the summaries, **omit that section entirely**.  
- If there is very little news, your report may be shorter—**2–3 sentences total is acceptable**. It’s better to be brief and accurate than to fabricate content to meet length expectations.  
- Only include facts explicitly supported by the provided summaries.
- Use **clear section breaks** with headings for readability.

Write in a tone suitable for a sports news outlet—professional, factual, and engaging. Avoid speculation or filler.

**Example Output:**
```markdown
### Summary
The Los Angeles Lakers secured a 112–108 overtime victory against the Golden State Warriors, fueled by a clutch 15-point fourth quarter from Anthony Davis. The win keeps them in contention for a top-six playoff spot in the Western Conference.

### Key Highlights
- Anthony Davis recorded 32 points, 14 rebounds, and 4 blocks, including the go-ahead basket with 12 seconds left in OT.
- Stephen Curry missed a potential game-winning three-pointer at the end of regulation, finishing with 28 points on 9-of-24 shooting.
- The Lakers outscored the Warriors 18–6 in second-chance points, capitalizing on 16 offensive rebounds.
```
**Article Content:**
Article:
{content}
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    try:
        summary = chain.invoke({"content": article["page_content"]})
    except Exception as e:
        summary = f"LLM error: {e}"
    return summary


def summarize_all_articles(days_back=7):
    """Get all articles from FAISS and generate summaries."""
    vectorstore = load_vectorstore()
    if not vectorstore:
        print("❌ No FAISS index found. Run create_or_update_vectorstore() first.")
        return []

    llm = get_llm()
    results = []

    # Iterate over all documents in the vectorstore
    for doc_id, doc in vectorstore.docstore._dict.items():
        file_name = doc.metadata.get("file", "unknown")
        # article_date = doc.metadata.get("published", doc.metadata.get("scraped_at", "unknown"))

        summary_file = SUMMARY_DIR / f"{Path(file_name).stem}_summary.md"
        if summary_file.exists():
            continue  # skip already summarized

        print(f"📝 Summarizing article: {doc.metadata.get('title', 'No title')}")
        summary_text = summarize_article({"page_content": doc.page_content}, llm=llm)
        summary_text = remove_think_section(summary_text)
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_text)

        results.append(summary_text)

    print(f"✅ Summarized {len(results)} articles.")
    return results


def generate_daily_reports():
    """Aggregate all summaries by date and produce daily reports."""
    summaries = []
    for file in SUMMARY_DIR.glob("*_summary.json"):
        try:
            with open(file, encoding="utf-8") as f:
                summaries.append(json.load(f))
        except Exception:
            continue

    # Group by date
    reports_by_date = {}
    for s in summaries:
        date = s.get("published", "unknown").split("T")[0]
        reports_by_date.setdefault(date, []).append(s)

    llm = get_llm()

    for date, daily_articles in reports_by_date.items():
        print(f"📅 Generating report for {date}")
        content_text = "\n\n".join(
            [f"{a['title']}:\n{a['summary']}" for a in daily_articles]
        )

        prompt_template = """
You are a seasoned sports journalist compiling a daily digest for a major sports network.  
Using the article summaries provided below—dated **{date}**—write a **concise daily report** in **Markdown format** that covers:

- The **most significant results or developments** across leagues (e.g., upsets, milestones, playoff implications).  
- **Notable injuries** (player, team impact, expected absence).  
- **Key player news** (trades, returns from injury, standout performances, disciplinary actions, or contract updates).

⚠️ **Critical Instructions**:  
- **Do NOT invent scores, stats, player names, teams, or events**. If a category (e.g., injuries) isn’t mentioned in the summaries, **omit that section entirely**.  
- If there is very little news, your report may be shorter—**2–3 sentences total is acceptable**. It’s better to be brief and accurate than to fabricate content to meet length expectations.  
- Only include facts explicitly supported by the provided summaries.
- Use **clear section breaks** with headings for readability.

Keep the tone professional, factual, and engaging—suitable for informed fans. Prioritize relevance and impact over volume.  
Limit the report to **4–6 short paragraphs total**, with clear section breaks.

**Example Output:**
```markdown
### Top Headlines – October 22, 2025  
The Boston Celtics opened their season with a dominant 118–94 win over the Miami Heat, led by Jayson Tatum’s 31-point performance. Meanwhile, in the NFL, the Kansas City Chiefs survived a late scare to defeat the Buffalo Bills 27–24 in overtime.

### Injuries to Watch  
- **Giannis Antetokounmpo (Bucks)**: Left Tuesday’s game with a right ankle sprain; listed as day-to-day ahead of Friday’s matchup.  
- **Tua Tagovailoa (Dolphins)**: Cleared concussion protocol and is expected to start Sunday after missing two games.

### Player & Roster News  
Lakers guard Austin Reaves signed a three-year, $60M extension, solidifying his role alongside LeBron James. In soccer, Erling Haaland scored his 15th goal of the Premier League season, setting a new club record for October.
```

Now generate the daily report based on the following summaries:

Summaries:
{content}
"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()

        try:
            daily_report = chain.invoke({"content": content_text, "date": date})
        except Exception as e:
            daily_report = f"LLM error: {e}"

        report_file = REPORT_DIR / f"report_{date}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(daily_report)

        print(f"✅ Daily report saved: {report_file}")


if __name__ == "__main__":
    # 1️⃣ Summarize all articles
    summarize_all_articles()

    # 2️⃣ Generate daily reports from summaries
    generate_daily_reports()
