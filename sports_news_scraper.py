import os
import time
import json
import random
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
import configparser

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from requests.packages.urllib3.exceptions import InsecureRequestWarning


# ===========================================================
# Load environment
# ===========================================================
load_dotenv()

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
        urls = [u.strip() for u in parser.get(section, "URLS", fallback="").split(",") if u.strip()]
        separator_str = str(parser.get(section, "DATETIME_SEPARATOR", fallback=None))
        separator = separator_str.replace('"', '')
        selectors = {
            "list": parser.get(section, "LIST", fallback=None),
            "title": parser.get(section, "TITLE", fallback=None),
            "author": parser.get(section, "AUTHOR", fallback=None),
            "date": parser.get(section, "DATE", fallback=None),
            "datetime_separator": separator,
            "content": parser.get(section, "CONTENT", fallback=None),
        }
        sources.append({"name": name, "urls": urls, "selectors": selectors})
    return sources


# ===========================================================
# Helpers
# ===========================================================
def extract_path_parts(url: str):
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    sport = parts[0] if len(parts) > 0 else "unknown"
    competition = parts[1] if len(parts) > 1 else "unknown"
    return sport, competition


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


# ===========================================================
# Scraper Logic
# ===========================================================
def scrape_article_page(article_url: str, selectors: dict):
    print(f"   📰 Fetching: {article_url}")
    # try:
    res = requests.get(article_url, headers=HEADERS, timeout=REQUEST_TIMEOUT, verify=VERIFY_SSL)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "lxml")

    title = clean_html_text(soup.select_one(selectors.get("title")))
    author = clean_html_text(soup.select_one(selectors.get("author")))
    date = clean_html_text(soup.select_one(selectors.get("date")))
    date_published = date.split(selectors.get("datetime_separator"))[0]
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

    # except Exception as e:
    #     print(f"   ❌ Error fetching article: {e}")
    #     return None


def save_article(article: dict, source: str, sport: str, competition: str):
    date_folder = article.get("date_published") or "unknown"
    folder = RAW_NEWS_DATA_DIR / sport / competition / date_folder / source
    folder.mkdir(parents=True, exist_ok=True)

    article_id = article["url"].rstrip("/").split("/")[-1]
    file_path = folder / f"{article_id}.json"

    data = {
        "scraped_at": datetime.now().isoformat(),
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

    for main_url in source["urls"]:
        sport, competition = extract_path_parts(main_url)
        print(f"\n🔍 {sport}/{competition}: {main_url}")

        existing = list_article_files(sport, competition)
        try:
            res = requests.get(main_url, headers=HEADERS, timeout=REQUEST_TIMEOUT, verify=VERIFY_SSL)
            res.raise_for_status()
        except Exception as e:
            print(f"❌ Failed to fetch {main_url}: {e}")
            continue

        soup = BeautifulSoup(res.text, "lxml")
        link_selector = source["selectors"].get("list")
        article_links = [urljoin(main_url, a["href"]) for a in soup.select(link_selector) if a.get("href")]

        print(f"   Found {len(article_links)} links")

        for article_url in article_links:
            article_id = article_url.split("/")[-1]
            if any(article_id in f for f in existing):
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

    for src in sources:
        scrape_source(src)

    print("\n✅ All sources scraped successfully.")
