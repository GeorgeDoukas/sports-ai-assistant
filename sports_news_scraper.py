import configparser
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from vector_store import VectorStoreManager

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from requests.packages.urllib3.exceptions import InsecureRequestWarning

GREEK_MONTH_MAP = {
    1: "Ιανουαρίου",
    2: "Φεβρουαρίου",
    3: "Μαρτίου",
    4: "Απριλίου",
    5: "Μαΐου",
    6: "Ιουνίου",
    7: "Ιουλίου",
    8: "Αυγούστου",
    9: "Σεπτεμβρίου",
    10: "Οκτωβρίου",
    11: "Νοεμβρίου",
    12: "Δεκεμβρίου",
}


# ===========================================================
# Load environment
# ===========================================================
load_dotenv()

COMPETITION_MAPPING = {
    "nba": "basketball",
    "euroleague": "basketball",
    "champions-league": "football",
    "superleague": "football",
}

if os.getenv("VERIFY_SSL", "false").lower() != "true":
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# ===========================================================
# Core Config
# ===========================================================
RAW_NEWS_DATA_DIR = Path(os.getenv("RAW_NEWS_DATA_DIR", "data/raw/news"))
RAW_NEWS_DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0"),
    "Accept-Language": "el-GR,el;q=0.9,en-US;q=0.8,en;q=0.7",
}

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() == "true"
DELAY_MIN = float(os.getenv("DELAY_MIN", "2"))
DELAY_MAX = float(os.getenv("DELAY_MAX", "4"))

# ===========================================================
# Source Config Loader
# ===========================================================
SOURCES_FILE = os.getenv("SOURCES_FILE", ".env")


def load_sources():
    """Reads the sources.env file into a list of source configs."""
    parser = configparser.ConfigParser()
    parser.read(SOURCES_FILE, encoding="utf-8")

    sources = []
    for section in parser.sections():
        name = parser.get(section, "NAME", fallback=section)
        competition_urls = [
            (competition_url.split("@")[0], competition_url.split("@")[1].rstrip("/"))
            for competition_url in parser.get(
                section, "COMPETITION_URLS", fallback=""
            ).split(",")
        ]
        separator_str = str(parser.get(section, "DATETIME_SEPARATOR", fallback=None))
        separator = separator_str.replace('"', "")
        selectors = {
            "list": parser.get(section, "LIST", fallback=None),
            "title": parser.get(section, "TITLE", fallback=None),
            "author": parser.get(section, "AUTHOR", fallback=None),
            "date": parser.get(section, "DATE", fallback=None),
            "datetime_separator": separator,
            "content": parser.get(section, "CONTENT", fallback=None),
        }
        sources.append(
            {"name": name, "competition_urls": competition_urls, "selectors": selectors}
        )
    return sources


# ===========================================================
# Helpers
# ===========================================================


def clean_html_text(element):
    if not element:
        return ""
    for tag in element.find_all(["strong", "em", "a", "span", "p", "div", "br"]):
        if tag.next_sibling and not str(tag.next_sibling).startswith(" "):
            tag.insert_after(" ")
    return " ".join(element.get_text(separator=" ", strip=True).split())


def list_article_files(sport: str, competition: str):
    base_path = RAW_NEWS_DATA_DIR / sport / competition
    if not base_path.exists():
        return []
    return [f.name for f in base_path.rglob("*.json")]


def normalize_and_format_date(date_string: str) -> str:
    """
    Parses various date formats (DD/MM/YYYY, MM/DD/YYYY with /.- separators),
    standardizes them, and optionally translates to a target language.
    """
    try:
        # Step 1: Split the date string by any common separator
        parts = re.split(r"[/.-]", date_string.strip())
        if len(parts) != 3:
            # If it's not a recognizable structure, return original
            return date_string

        p1, p2, p3 = map(int, parts)

        # Step 2: Intelligently determine the date format (DMY vs MDY)
        # This heuristic assumes if a value > 12, it must be the day.
        if p1 > 12:  # Format is likely Day/Month/Year
            day, month, year = p1, p2, p3
        elif p2 > 12:  # Format is likely Month/Day/Year
            month, day, year = p1, p2, p3
        else:
            # Ambiguous (e.g., 04.05.2025). Assume Day/Month/Year as it's common in Greece/Europe.
            day, month, year = p1, p2, p3

        # Ensure year is four digits
        if year < 2000:
            year += 2000

        # Step 3: Create a datetime object for validation and formatting
        dt_obj = datetime(year, month, day)

        # Step 4: Format the date based on the Greek language
        return f"{dt_obj.day} {GREEK_MONTH_MAP[dt_obj.month]} {dt_obj.year}"

    except (ValueError, IndexError, KeyError):
        # If any parsing fails, return the original string to avoid crashing
        return date_string


# ===========================================================
# Scraper Logic
# ===========================================================
def scrape_article_page(article_url: str, selectors: dict):
    print(f"   📰 Fetching: {article_url}")
    try:
        res = requests.get(
            article_url, headers=HEADERS, timeout=REQUEST_TIMEOUT, verify=VERIFY_SSL
        )
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")

        title = clean_html_text(soup.select_one(selectors.get("title")))
        author = clean_html_text(soup.select_one(selectors.get("author")))
        date = clean_html_text(soup.select_one(selectors.get("date")))
        date_published = date.split(selectors.get("datetime_separator"))[0]
        date_published = normalize_and_format_date(date_published)
        content = clean_html_text(soup.select_one(selectors.get("content")))

        if not content.strip():
            print("   ⚠️ Empty content, skipping.")
            return None

        return {
            "title": title,
            "author": author,
            "date_published": date_published,
            "content": content,
            "url": article_url,
        }

    except Exception as e:
        print(f"   ❌ Error fetching article: {e}")
        return None


def save_article(article: dict, source: str, sport: str, competition: str):
    date_folder = article.get("date_published") or "unknown"
    folder = RAW_NEWS_DATA_DIR / sport / competition / date_folder / source
    folder.mkdir(parents=True, exist_ok=True)

    article_id = article["url"].rstrip("/").split("/")[-1]
    file_path = folder / f"{article_id}.json"

    data = {
        "scraped_at": datetime.now().isoformat(),
        "processing_status": "scraped",
        "source": source,
        "sport": sport,
        "competition": competition,
        "article": article,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"   ✅ Saved: {file_path}")


def scrape_source(source: dict):
    name = source["name"]
    print(f"\n📡 Scraping source: {name}")

    for competition, url in source["competition_urls"]:
        sport = COMPETITION_MAPPING.get(competition, "unknown")
        print(f"\n🔍 {sport}/{competition}: {url}")

        existing = list_article_files(sport, competition)
        try:
            res = requests.get(
                url, headers=HEADERS, timeout=REQUEST_TIMEOUT, verify=VERIFY_SSL
            )
            res.raise_for_status()
        except Exception as e:
            print(f"❌ Failed to fetch {url}: {e}")
            continue

        soup = BeautifulSoup(res.text, "lxml")
        link_selector = source["selectors"].get("list")
        if not link_selector:
            print(
                f"   ⚠️ No 'list' selector found for source {name}, skipping competition {competition}."
            )
            continue

        article_links = [
            urljoin(url, a["href"]) for a in soup.select(link_selector) if a.get("href")
        ]

        print(f"   Found {len(article_links)} links")

        for article_url in article_links:
            article_url = article_url.rstrip("/")
            article_id = article_url.split("/")[-1]
            if article_id + ".json" in existing:
                print(f"   ↪️ Already saved: {article_url}")
                continue

            article = scrape_article_page(article_url, source["selectors"])
            if article:
                save_article(article, name, sport, competition)

            delay = random.uniform(DELAY_MIN, DELAY_MAX)
            print(f"   ⏳ Sleeping {delay:.1f}s...")
            time.sleep(delay)


# ===========================================================
# Entrypoint
# ===========================================================
if __name__ == "__main__":
    sources = load_sources()
    if not sources:
        print("❌ No sources found in config file.")
        exit(1)

    print(f"🚀 Starting scraping for {len(sources)} sources in parallel...")
    with ThreadPoolExecutor() as executor:
        # Submit all source scraping tasks
        future_to_source = {executor.submit(scrape_source, src): src for src in sources}

        # As each task completes, print a message
        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                future.result()  # This will re-raise any exception from the task
                print(f"✅ Source '{source['name']}' completed successfully.")
            except Exception as e:
                print(f"❌ Source '{source['name']}' encountered an error: {e}")

    print("\n✅ All sources scraped successfully.")
    
    manager = VectorStoreManager()
    
    manager.create_or_update(days_back=30)

    print("\n✅ Vector store updated successfully.")