import configparser
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from sports_news_scraper import normalize_and_format_date_to_greek

# Utility map for Greek months (kept for legacy reference, but new reverse map is used)
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

# New reverse map for converting Greek month names to numbers (for news folder conversion)
REVERSE_GREEK_MONTH_MAP = {
    "Ιανουαρίου": 1,
    "Φεβρουαρίου": 2,
    "Μαρτίου": 3,
    "Απριλίου": 4,
    "Μαΐου": 5,
    "Ιουνίου": 6,
    "Ιουλίου": 7,
    "Αυγούστου": 8,
    "Σεπτεμβρίου": 9,
    "Οκτωβρίου": 10,
    "Νοεμβρίου": 11,
    "Δεκεμβρίου": 12,
}


# ===========================================================
# Load environment & Core Config
# ===========================================================
load_dotenv()

COMPETITION_MAPPING = {
    "nba": "basketball",
    "euroleague": "basketball",
    "champions-league": "football",
    "superleague": "football",
}


RAW_NEWS_DATA_DIR = Path(os.getenv("RAW_NEWS_DATA_DIR", "data/raw/news"))
RAW_STATS_DATA_DIR = Path(os.getenv("RAW_STATS_DATA_DIR", "data/raw/stats"))
RAW_STATS_DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0"),
    "Accept-Language": "el-GR,el;q=0.9,en-US;q=0.8,en;q=0.7",
}

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() == "true"
DELAY_MIN = float(os.getenv("DELAY_MIN", "2"))
DELAY_MAX = float(os.getenv("DELAY_MAX", "4"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
SOURCES_FILE = os.getenv("SOURCES_FILE", ".env")

# Configure headless Chrome
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Remove if you want to see the browser
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)


# ===========================================================
# Source Config Loader (Modified to only load STATS sources)
# ===========================================================
def load_stats_sources() -> list[dict]:
    """Reads the config file and returns only sections intended for stats scraping."""
    parser = configparser.ConfigParser()
    parser.read(SOURCES_FILE, encoding="utf-8")

    stats_sources = []

    for section in parser.sections():
        # Only load sources explicitly marked for STATS (e.g., [STATS_SOURCE_1])
        if "STATS_SOURCE" in section.upper():
            name = parser.get(section, "NAME", fallback=section)
            try:
                # Extract competition name and URL from the config
                competition_urls = [
                    (
                        competition_url.split("@")[0],
                        competition_url.split("@")[1].rstrip("/"),
                    )
                    for competition_url in parser.get(
                        section, "COMPETITION_URLS", fallback=""
                    ).split(",")
                ]
                stats_sources.append(
                    {
                        "name": name,
                        "competition_urls": competition_urls,
                        "section": section,
                    }
                )
            except Exception as e:
                print(f"⚠️ Error parsing COMPETITION_URLS in section {section}: {e}")
                continue
    return stats_sources


# ===========================================================
# Utility Functions
# ===========================================================


def clean_html_text(element):
    """Safely extracts and cleans text from a BeautifulSoup element."""
    if not element:
        return ""
    return " ".join(element.get_text(separator=" ", strip=True).split())


def convert_greek_date_to_numeric(greek_date_str: str) -> str | None:
    """Converts a Greek date string (e.g., '22 Οκτωβρίου 2025') to DD.MM.YYYY."""
    try:
        parts = greek_date_str.split()
        if len(parts) != 3:
            return None

        day = int(parts[0])
        month_name = parts[1]
        year = int(parts[2])

        month = REVERSE_GREEK_MONTH_MAP.get(month_name)
        if not month:
            return None

        # Ensure leading zeros for day and month
        return f"{day:02d}.{month:02d}.{year}"
    except Exception:
        return None


def normalize_and_format_date(date_string: str) -> str:
    """
    Parses DD.MM. date format (from livescore) and standardizes it to DD.MM.YYYY.
    This is the new standard format for matching and folder names.
    """
    date_string = date_string.strip()
    try:
        # Check for livescore's DD.MM. format (e.g., '29.10. 19:45' -> '29.10.')
        match_date_time = re.match(r"^(\d{1,2})\.(\d{1,2})\.", date_string)
        if match_date_time:
            day, month = map(int, match_date_time.groups())
            # Assume current year, as news dates are usually recent
            current_year = datetime.now().year
            # Return DD.MM.YYYY format with leading zeros
            return f"{day:02d}.{month:02d}.{current_year}"

        # If the input date is already in the target format, return it
        if re.match(r"^\d{2}\.\d{2}\.\d{4}$", date_string):
            return date_string

        # If it's a full Greek date string (which shouldn't happen here, but for robustness)
        # We can try to convert it just in case
        numeric_date = convert_greek_date_to_numeric(date_string)
        return numeric_date if numeric_date else date_string

    except (ValueError, IndexError, KeyError):
        return date_string


def slugify(text: str) -> str:
    """Converts text to a safe filename slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-\.]", "", text)  # Allow dots for date format
    text = re.sub(r"[-\s]+", "-", text)
    return text


def get_target_dates(sport: str, competition: str) -> set:
    """
    Gets all unique date folder names from the news data (Greek format)
    and converts them to the standardized DD.MM.YYYY format for filtering.
    """
    base_path = RAW_NEWS_DATA_DIR / sport / competition
    if not base_path.exists():
        return set()

    # 1. Get raw Greek folder names
    raw_target_dates = {d.name for d in base_path.iterdir() if d.is_dir()}

    # 2. Convert raw Greek dates to numeric DD.MM.YYYY format
    numeric_target_dates = set()
    for raw_date in raw_target_dates:
        numeric_date = convert_greek_date_to_numeric(raw_date)
        if numeric_date:
            numeric_target_dates.add(numeric_date)

    if not numeric_target_dates:
        print(
            f"   ⚠️ No news dates found for {sport}/{competition}. Stats will be filtered by date, but the list is empty."
        )
        return set()

    print(
        f"   🎯 Found {len(numeric_target_dates)} news dates (in DD.MM.YYYY format) to target for stats."
    )
    print(f"      Target Dates: {sorted(numeric_target_dates)}")
    return numeric_target_dates


# ===========================================================
# Scraping Logic
# ===========================================================


def scrape_stats_competition(source: dict):
    name = source["name"]
    print(f"\n📈 Starting stats scrape for source: {name}")

    for competition, base_url in source["competition_urls"]:
        sport = COMPETITION_MAPPING.get(competition, "unknown")
        print(f"\n   🔍 {sport}/{competition}: Fetching results from {base_url}")

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=chrome_options
        )
        wait = WebDriverWait(driver, 15)

        # 1. Get target dates (now in DD.MM.YYYY numeric format)
        # target_dates = get_target_dates(sport, competition)

        # if not target_dates:
        #     print(
        #         f"   ⚠️ Skipping {competition}. No news data found to filter match dates."
        #     )
        #     continue

        try:
            # Fetch the results page
            driver.get(base_url)
            time.sleep(2)  # Initial load

            try:
                # Wait for and click "Reject All" button
                reject_button = wait.until(
                    EC.element_to_be_clickable((By.ID, "onetrust-reject-all-handler"))
                )
                reject_button.click()
                print("✅ Rejected all cookies.")
            except Exception as e:
                print(f"⚠️ Could not find or click 'Reject All' button: {e}")

            # Wait for matches to appear (adjust selector if needed)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#live-table")))
            # Find all match containers (adjust selector based on actual structure)
            match_elements = driver.find_elements(By.CSS_SELECTOR, ".event__match")
            print(f"Found {len(match_elements)} match containers.")

            match_data_list = []
            for match_el in match_elements:
                try:
                    date_el = match_el.find_element(
                        By.CSS_SELECTOR, ".event__time"
                    ).text
                    date_text = date_el.strip()

                    match_date = normalize_and_format_date(date_text)
                    greek_match_date = normalize_and_format_date_to_greek(match_date)
                    # Filter by target dates
                    # if match_date not in target_dates:
                    #     print(
                    #         f" → Skipping match on {match_date} (not in target news dates)."
                    #     )
                    #     continue
                    if sport == "basketball":
                        home_team = match_el.find_element(By.CSS_SELECTOR, ".event__participant--home").text.strip()
                        away_team = match_el.find_element(By.CSS_SELECTOR, ".event__participant--away").text.strip()
                    elif sport == "football":
                        home_team = match_el.find_element(
                            By.CSS_SELECTOR, ".event__homeParticipant"
                        ).text.strip()
                        away_team = match_el.find_element(
                            By.CSS_SELECTOR, ".event__awayParticipant"
                        ).text.strip()
                    home_score = match_el.find_element(
                        By.CSS_SELECTOR, ".event__score--home"
                    ).text.strip()
                    away_score = match_el.find_element(
                        By.CSS_SELECTOR, ".event__score--away"
                    ).text.strip()

                    # Extract match link
                    link_el = match_el.find_element(By.TAG_NAME, "a")
                    match_path = link_el.get_attribute("href")
                    if not match_path:
                        continue

                    # Build stats URL
                    stats_url = match_path.replace(
                        "/?mid=", "/summary/player-stats/top/?mid="
                    )
                    # stats_url = match_path.replace("/?mid=", "/summary/player-stats/overall/?mid=")
                    print(
                        f" → Processing: {home_team} vs {away_team} ({greek_match_date})"
                    )
                    print(f"   Stats URL: {stats_url}")
                    match_data = {
                        "filename": f"{home_team}-{away_team}~~~{home_score}-{away_score}.csv",
                        "home_team": home_team,
                        "away_team": away_team,
                        "date": greek_match_date,
                        "stats_url": stats_url,
                    }
                    match_data_list.append(match_data)
                except Exception as e:
                    print(
                        f"Error processing match {home_team}-{away_team}@{match_date}: {e}"
                    )
                    continue

            # home_team = "Ολυμπιακ"
            # away_team = "ΑΕΚ"

            main_window = driver.current_window_handle
            # match_data_list = [{"filename":"oly-aek","date":"28 Οκτωβρίου 2025","stats_url":"https://www.livescore.in/gr/match/football/aek-ANpZncAM/olympiakos-piraeus-hnzvnHPS/summary/player-stats/top/?mid=WtALBSvS"}]
            for i, match in enumerate(match_data_list):
                stats_url = match["stats_url"]
                filename = match["filename"]

                print(f"Processing {i+1}/{len(match_data_list)}: {filename}")

                try:
                    # Open new tab
                    driver.execute_script("window.open('');")
                    driver.switch_to.window(
                        driver.window_handles[-1]
                    )  # Switch to new tab

                    # Load stats page
                    driver.get(stats_url)

                    # time.sleep(2)  # Optional: let JS finish

                    if sport == "basketball":
                        wait.until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, ".playerStatsTable")
                            )
                        )

                        headers = ["Παίκτης", "Ομάδα"]
                        all_rows = []

                        player_stats_table_headers = driver.find_elements(
                            By.CSS_SELECTOR, ".ui-table__headerCell"
                        )
                        for header in player_stats_table_headers:
                            title = header.get_attribute("title").strip()
                            if title:
                                headers.append(title)

                        player_stats_table_rows = driver.find_elements(
                            By.CSS_SELECTOR, ".ui-table__row"
                        )
                        for row in player_stats_table_rows:
                            player_row = []
                            player_stats_table_cells = row.find_elements(
                                By.CSS_SELECTOR, ".playerStatsTable__cell"
                            )
                            for cell in player_stats_table_cells:
                                player_row.append(cell.text)
                            all_rows.append(player_row)
                    elif sport == "football":
                        wait.until(
                            EC.presence_of_element_located((By.TAG_NAME, "table"))
                        )

                        headers = []
                        all_rows = []

                        headers_section = driver.find_element(By.TAG_NAME, "tr")
                        player_stats_table_headers = headers_section.find_elements(
                            By.TAG_NAME, "button"
                        )
                        for header in player_stats_table_headers:
                            title = header.text
                            # print(title)
                            if title:
                                headers.append(title)
                        headers[0] = "Όνομα"
                        headers.append("Ομάδα")
                        headers.append("Θέση")

                        # home team stats
                        player_stats_table_headers[0].click()
                        home_div = wait.until(
                            EC.element_to_be_clickable(
                                (By.CSS_SELECTOR, 'div[data-analytics-alias="HOME"]')
                            )
                        )
                        home_div.click()
                        table_section = driver.find_element(By.TAG_NAME, "tbody")
                        player_stats_table_rows = table_section.find_elements(
                            By.TAG_NAME, "tr"
                        )
                        for row in player_stats_table_rows:

                            player_row = []
                            player_stats_table_cells = row.find_elements(
                                By.TAG_NAME, "td"
                            )
                            for cell in player_stats_table_cells:
                                # print(cell.text)
                                player_row.append(cell.text)
                            player_row.append(match["home_team"])
                            name, position = player_row[0].split("\n")
                            player_row[0] = name
                            player_row.append(position)
                            all_rows.append(player_row)

                        # away team
                        player_stats_table_headers[0].click()
                        away_div = wait.until(
                            EC.element_to_be_clickable(
                                (By.CSS_SELECTOR, 'div[data-analytics-alias="AWAY"]')
                            )
                        )
                        away_div.click()
                        table_section = driver.find_element(By.TAG_NAME, "tbody")
                        player_stats_table_rows = table_section.find_elements(
                            By.TAG_NAME, "tr"
                        )
                        for row in player_stats_table_rows:

                            player_row = []
                            player_stats_table_cells = row.find_elements(
                                By.TAG_NAME, "td"
                            )
                            for cell in player_stats_table_cells:
                                # print(cell.text)
                                player_row.append(cell.text)
                            player_row.append(match["away_team"])
                            name, position = player_row[0].split("\n")
                            player_row[0] = name
                            player_row.append(position)
                            all_rows.append(player_row)

                    df = pd.DataFrame(all_rows, columns=headers)
                    df_sorted = df.sort_values(by="Αξιολόγηση παίκτη", ascending=False)
                    print(df_sorted.head())
                    output_path = (
                        f"data/raw/stats/{sport}/{competition}/{match['date']}/"
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    df_sorted.to_csv(
                        f"{output_path}/{filename}", index=False, encoding="utf-8"
                    )
                    print(f"✅ Saved: {filename}")

                except Exception as e:
                    print(f"❌ Error on {stats_url}: {e}")

                finally:
                    # Close current tab and return to main
                    driver.close()
                    driver.switch_to.window(main_window)
                    time.sleep(1)  # Brief pause before next iteration
        finally:
            driver.quit()
            print("Browser closed.")


# ===========================================================
# Entrypoint
# ===========================================================
if __name__ == "__main__":
    stats_sources = load_stats_sources()

    if not stats_sources:
        print(
            "❌ No stats sources found in the config file. Please ensure you have sections like [STATS_SOURCE_1] with COMPETITION_URLS defined."
        )
        exit(1)

    print(
        f"🚀 Starting stats scraping for {len(stats_sources)} sources using {MAX_WORKERS} workers..."
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_source = {
            executor.submit(scrape_stats_competition, src): src for src in stats_sources
        }

        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                future.result()
                print(f"✅ Source '{source['name']}' thread completed.")
            except Exception as e:
                print(f"❌ Source '{source['name']}' encountered a fatal error: {e}")

    print("\n\n" + "=" * 50)
    print("✅ All stats scraping tasks completed.")
