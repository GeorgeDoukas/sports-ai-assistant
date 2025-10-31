import configparser
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Assume this function is available from the other script
from sports_news_scraper import normalize_and_format_date_to_greek


# ===========================================================
# Load environment & Core Config
# ===========================================================
load_dotenv()

# --- Constants ---
COMPETITION_MAPPING = {
    "nba": "basketball",
    "euroleague": "basketball",
    "champions-league": "football",
    "superleague": "football",
}

REVERSE_GREEK_MONTH_MAP = {
    "Ιανουαρίου": 1, "Φεβρουαρίου": 2, "Μαρτίου": 3, "Απριλίου": 4,
    "Μαΐου": 5, "Ιουνίου": 6, "Ιουλίου": 7, "Αυγούστου": 8,
    "Σεπτεμβρίου": 9, "Οκτωβρίου": 10, "Νοεμβρίου": 11, "Δεκεμβρίου": 12,
}

# --- Environment Variables ---
RAW_NEWS_DATA_DIR = Path(os.getenv("RAW_NEWS_DATA_DIR", "data/raw/news"))
RAW_STATS_DATA_DIR = Path(os.getenv("RAW_STATS_DATA_DIR", "data/raw/stats"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
SOURCES_FILE = os.getenv("SOURCES_FILE", ".env")
WEBDRIVER_WAIT_TIMEOUT = 15

# Ensure the base stats directory exists
RAW_STATS_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================
# Selenium WebDriver Setup
# ===========================================================

def get_driver_options() -> Options:
    """Configures and returns Chrome options for Selenium."""
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment to run headlessly
    chrome_options.add_argument("--incognito")  # Open in incognito mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    return chrome_options


def init_driver() -> Tuple[WebDriver, WebDriverWait]:
    """Initializes and returns a new Chrome driver and a WebDriverWait instance."""
    options = get_driver_options()
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, WEBDRIVER_WAIT_TIMEOUT)
    return driver, wait


# ===========================================================
# Source Config Loader
# ===========================================================

def load_stats_sources() -> List[Dict]:
    """Reads the config file and returns only sections intended for stats scraping."""
    parser = configparser.ConfigParser()
    parser.read(SOURCES_FILE, encoding="utf-8")

    stats_sources = []
    for section in parser.sections():
        if "STATS_SOURCE" in section.upper():
            name = parser.get(section, "NAME", fallback=section)
            try:
                # Extract (competition_name, url) tuples
                competition_urls = [
                    (
                        comp_url.split("@")[0].strip(),
                        comp_url.split("@")[1].strip().rstrip("/"),
                    )
                    for comp_url in parser.get(section, "COMPETITION_URLS", fallback="").split(",")
                    if "@" in comp_url
                ]

                if not competition_urls:
                    print(f"⚠️ No valid COMPETITION_URLS found for source {name}. Skipping.")
                    continue

                stats_sources.append({
                    "name": name,
                    "competition_urls": competition_urls,
                    "section": section,
                })
            except Exception as e:
                print(f"❌ Error parsing COMPETITION_URLS in section {section}: {e}")
                continue
    return stats_sources


# ===========================================================
# Utility Functions
# ===========================================================

def convert_greek_date_to_numeric(greek_date_str: str) -> Optional[str]:
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

        # Return DD.MM.YYYY format with leading zeros
        return f"{day:02d}.{month:02d}.{year}"
    except (ValueError, IndexError):
        print(f"⚠️ Could not parse Greek date string: {greek_date_str}")
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
            current_year = datetime.now().year
            # Return DD.MM.YYYY format with leading zeros
            return f"{day:02d}.{month:02d}.{current_year}"

        # If the input date is already in the target format, return it
        if re.match(r"^\d{2}\.\d{2}\.\d{4}$", date_string):
            return date_string

        # Try to convert from full Greek date string as a fallback
        numeric_date = convert_greek_date_to_numeric(date_string)
        return numeric_date if numeric_date else date_string

    except (ValueError, IndexError):
        print(f"⚠️ Could not normalize date string: {date_string}")
        return date_string


def get_target_dates(sport: str, competition: str) -> Set[str]:
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
            f"  ⚠️ No news dates found for {sport}/{competition}. Stats will be filtered, but the target list is empty."
        )
    else:
        print(
            f"  🎯 Found {len(numeric_target_dates)} news dates (DD.MM.YYYY) to target for {sport}/{competition} stats."
        )
    return numeric_target_dates


def save_stats_csv(
    df: pd.DataFrame,
    sport: str,
    competition: str,
    date_folder_parts: Tuple[str, str, str],
    filename: str
):
    """Saves a DataFrame to the new directory structure: .../{year}/{month}/{day}/"""
    try:
        year, month, day = date_folder_parts
        
        output_dir = RAW_STATS_DATA_DIR / sport / competition / year / month / day
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename is already pre-sanitized by the calling function
        output_path = output_dir / filename
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"    ✅ Saved: {output_path}")
    except Exception as e:
        print(f"    ❌ Failed to save CSV {filename}: {e}")


# ===========================================================
# Statistics Scraping Logic (Basketball)
# ===========================================================

def scrape_basketball_stats(driver: WebDriver, wait: WebDriverWait) -> Optional[pd.DataFrame]:
    """Scrapes the player stats table for a basketball match."""
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".playerStatsTable")))
        
        headers = ["Παίκτης", "Ομάδα"]
        all_rows = []

        # Get headers
        header_elements = driver.find_elements(By.CSS_SELECTOR, ".ui-table__headerCell")
        for header in header_elements:
            title = header.get_attribute("title").strip()
            if title:
                headers.append(title)

        # Get rows
        row_elements = driver.find_elements(By.CSS_SELECTOR, ".ui-table__row")
        for row in row_elements:
            cells = row.find_elements(By.CSS_SELECTOR, ".playerStatsTable__cell")
            all_rows.append([cell.text for cell in cells])

        if not all_rows:
            print("    ⚠️ No basketball stats rows found on page.")
            return None

        df = pd.DataFrame(all_rows, columns=headers)
        return df

    except Exception as e:
        print(f"    ❌ Error scraping basketball stats: {e}")
        return None


# ===========================================================
# Statistics Scraping Logic (Football)
# ===========================================================

def _scrape_football_team_stats(
    driver: WebDriver, wait: WebDriverWait, team_alias: str, team_name: str
) -> List[List[str]]:
    """Helper function to scrape stats for a single football team (Home or Away)."""
    team_rows = []
    try:
        # Click the 'HOME' or 'AWAY' button
        team_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f'div[data-analytics-alias="{team_alias}"]'))
        )
        team_button.click()
        
        # Wait for table body to update (slight pause)
        time.sleep(0.5) 

        table_body = driver.find_element(By.TAG_NAME, "tbody")
        rows = table_body.find_elements(By.TAG_NAME, "tr")

        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            player_row = [cell.text for cell in cells]
            
            # Add team name
            player_row.append(team_name)
            
            # Split "Name\nPosition" into two columns
            try:
                name, position = player_row[0].split("\n")
                player_row[0] = name
                player_row.append(position)
                team_rows.append(player_row)
            except (ValueError, IndexError):
                print(f"    ⚠️ Could not parse player name/position for row: {player_row}")
                
    except Exception as e:
        print(f"    ❌ Error scraping stats for team {team_name} ({team_alias}): {e}")
    
    return team_rows


def scrape_football_stats(
    driver: WebDriver, wait: WebDriverWait, match_data: Dict
) -> Optional[pd.DataFrame]:
    """Scrapes the player stats table for a football match by cycling through Home/Away."""
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        
        headers = []
        all_rows = []

        # Get headers
        header_row = driver.find_element(By.TAG_NAME, "tr")
        header_buttons = header_row.find_elements(By.TAG_NAME, "button")
        for header in header_buttons:
            title = header.text.strip()
            if title:
                headers.append(title)
        
        # Manually correct/add headers
        headers[0] = "Όνομα"
        headers.append("Ομάδα")
        headers.append("Θέση")

        # Click to open dropdown
        header_buttons[0].click()
        time.sleep(0.5) # Wait for dropdown

        # Scrape Home Team
        home_rows = _scrape_football_team_stats(driver, wait, "HOME", match_data["home_team"])
        all_rows.extend(home_rows)

        # Scrape Away Team
        header_buttons[0].click() # Re-click to open dropdown
        time.sleep(0.5)
        away_rows = _scrape_football_team_stats(driver, wait, "AWAY", match_data["away_team"])
        all_rows.extend(away_rows)

        if not all_rows:
            print(f"    ⚠️ No football stats rows found for match: {match_data['filename']}")
            return None

        df = pd.DataFrame(all_rows, columns=headers)
        return df.sort_values(by="Αξιολόγηση παίκτη", ascending=False)

    except Exception as e:
        print(f"    ❌ Error scraping football stats: {e}")
        return None


# ===========================================================
# Match & Competition Scraping Logic
# ===========================================================

def get_match_list(driver: WebDriver, wait: WebDriverWait, base_url: str, sport: str) -> List[Dict]:
    """Fetches the main competition page and scrapes the list of matches."""
    match_data_list = []
    try:
        driver.get(base_url)
        time.sleep(2)  # Initial load
    except Exception as e:
        print(f"  ❌ Failed to load base URL {base_url}: {e}")
        return []

    # 1. Handle cookie banner
    try:
        reject_button = wait.until(
            EC.element_to_be_clickable((By.ID, "onetrust-reject-all-handler"))
        )
        reject_button.click()
        print(f"  Clicked 'Reject All' cookies for {base_url}")
    except TimeoutException:
        print(f"  ⚠️ Could not find 'Reject All' button. Continuing anyway.")
    except Exception as e:
        print(f"  ⚠️ Error clicking 'Reject All' button: {e}")

    # 2. Wait for matches and extract them
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#live-table")))
        match_elements = driver.find_elements(By.CSS_SELECTOR, ".event__match")
        print(f"  Found {len(match_elements)} match containers on page.")

        # TODO: Date filtering is disabled as per original script's commented-out state.
        # To enable, uncomment the next 2 lines and the filter block below.
        # target_dates = get_target_dates(sport, competition)
        # if not target_dates: ...

        for match_el in match_elements:
            try:
                date_text = match_el.find_element(By.CSS_SELECTOR, ".event__time").text.strip()
                match_date = normalize_and_format_date(date_text)
                
                # This converts DD.MM.YYYY to "DD <MonthName> YYYY" for folder naming
                greek_match_date = normalize_and_format_date_to_greek(match_date)

                # Parse Greek date for folder structure
                date_parts = greek_match_date.split()
                if len(date_parts) == 3:
                    day, month_name, year = date_parts[0], date_parts[1], date_parts[2]
                else:
                    print(f"    ⚠️ Could not parse Greek date '{greek_match_date}'. Using 'unknown' folders.")
                    day, month_name, year = "unknown", "unknown", "unknown"

                # TODO: This filter is disabled to match the original script's functionality.
                # if match_date not in target_dates:
                #     print(f"    Skipping match on {match_date} (not in target dates).")
                #     continue

                # Get team names (selectors differ by sport)
                if sport == "basketball":
                    home_team_el = match_el.find_element(By.CSS_SELECTOR, ".event__participant--home")
                    away_team_el = match_el.find_element(By.CSS_SELECTOR, ".event__participant--away")
                else:  # football
                    home_team_el = match_el.find_element(By.CSS_SELECTOR, ".event__homeParticipant")
                    away_team_el = match_el.find_element(By.CSS_SELECTOR, ".event__awayParticipant")
                
                home_team = home_team_el.text.strip()
                away_team = away_team_el.text.strip()
                
                home_score = match_el.find_element(By.CSS_SELECTOR, ".event__score--home").text.strip()
                away_score = match_el.find_element(By.CSS_SELECTOR, ".event__score--away").text.strip()

                link_el = match_el.find_element(By.TAG_NAME, "a")
                match_path = link_el.get_attribute("href")
                if not match_path:
                    continue

                # Build stats URL
                if sport == "basketball":
                    stats_url = match_path.replace("/?mid=", "/summary/player-stats/overall/?mid=")
                else:
                    stats_url = match_path.replace("/?mid=", "/summary/player-stats/top/?mid=")
                
                print(f"    → Found match: {home_team} vs {away_team} ({greek_match_date})")
                
                match_data = {
                    "filename": f"{home_team}-{away_team}~~~{home_score}-{away_score}.csv",
                    "home_team": home_team,
                    "away_team": away_team,
                    "date_folder_parts": (year, month_name, day), # Tuple for saving
                    "stats_url": stats_url,
                    "sport": sport
                }
                match_data_list.append(match_data)

            except Exception as e:
                print(f"    ⚠️ Error processing a match element: {e}")
                continue

    except TimeoutException:
        print(f"  ❌ Timed out waiting for match table at {base_url}")
    except Exception as e:
        print(f"  ❌ Error finding matches at {base_url}: {e}")

    return match_data_list


def scrape_match_stats_in_new_tab(
    driver: WebDriver, main_window: str, match_data: Dict
) -> Optional[pd.DataFrame]:
    """
    Opens a new tab, scrapes stats for a single match, and closes the tab.
    """
    stats_url = match_data["stats_url"]
    sport = match_data["sport"]
    
    try:
        # Open and switch to new tab
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1])
        tab_wait = WebDriverWait(driver, WEBDRIVER_WAIT_TIMEOUT)

        driver.get(stats_url)

        if sport == "basketball":
            df = scrape_basketball_stats(driver, tab_wait)
        elif sport == "football":
            df = scrape_football_stats(driver, tab_wait, match_data)
        else:
            print(f"    ⚠️ Unknown sport '{sport}', cannot scrape stats.")
            df = None
        
        return df

    except Exception as e:
        print(f"    ❌ Error scraping stats from {stats_url}: {e}")
        return None
    finally:
        # Close tab and switch back to main window
        try:
            driver.close()
            driver.switch_to.window(main_window)
            time.sleep(1)  # Brief pause
        except Exception as e:
            print(f"    ❌ Error closing tab or switching window: {e}")
            # If window switching fails, we might be in a bad state.
            # Force switch back to main, if it still exists.
            if main_window in driver.window_handles:
                 driver.switch_to.window(main_window)
            else:
                # This is bad, the main window might be gone.
                # The driver.quit() in the outer function will handle cleanup.
                print("    ❌ Main window handle lost.")


def scrape_single_competition(task: Tuple[str, str, str]):
    """
    Main scraping function for a *single competition*.
    This function is designed to be run in a thread pool.
    """
    competition, sport, base_url = task
    print(f"\n📈 Starting scrape for: {sport}/{competition} at {base_url}")

    driver, wait = init_driver()

    try:
        # 1. Get all matches from the competition's main page
        match_data_list = get_match_list(driver, wait, base_url, sport)
        
        if not match_data_list:
            print(f"  ⚠️ No matches found for {sport}/{competition}. Skipping.")
            return
        
        main_window = driver.current_window_handle
        
        # 2. Process each match in a new tab
        for i, match_data in enumerate(match_data_list):
            filename = match_data["filename"]
            year, month, day = match_data["date_folder_parts"]

            # Sanitize filename once
            safe_filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
            
            # --- Check if file already exists ---
            expected_path = RAW_STATS_DATA_DIR / sport / competition / year / month / day / safe_filename
            if expected_path.exists():
                print(f"  Skipping {i+1}/{len(match_data_list)} (already exists): {safe_filename}")
                continue
            # ------------------------------------

            print(f"  Processing {i+1}/{len(match_data_list)}: {safe_filename}")
            
            df_stats = scrape_match_stats_in_new_tab(driver, main_window, match_data)

            # 3. Save the results
            if df_stats is not None and not df_stats.empty:
                save_stats_csv(
                    df_stats,
                    sport,
                    competition,
                    match_data["date_folder_parts"],
                    safe_filename,
                )
            else:
                print(f"    ⚠️ No stats DataFrame returned for {safe_filename}")

    except Exception as e:
        print(f"  ❌ A fatal error occurred during scrape for {sport}/{competition}: {e}")
    finally:
        driver.quit()
        print(f"Browser closed for {sport}/{competition}.")


# ===========================================================
# Entrypoint
# ===========================================================
if __name__ == "__main__":
    stats_sources = load_stats_sources()

    if not stats_sources:
        print(
            f"❌ No stats sources found in {SOURCES_FILE}. Ensure sections like [STATS_SOURCE_1] "
            "with COMPETITION_URLS are defined."
        )
        exit(1)

    # Create a flat list of tasks, one for each competition
    competition_tasks = []
    for source in stats_sources:
        for competition, base_url in source["competition_urls"]:
            sport = COMPETITION_MAPPING.get(competition, "unknown")
            competition_tasks.append((competition, sport, base_url))

    if not competition_tasks:
        print("❌ No competitions found in any source. Exiting.")
        exit(1)

    print(
        f"🚀 Starting stats scraping for {len(competition_tasks)} competitions "
        f"across {len(stats_sources)} sources using {MAX_WORKERS} workers..."
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(scrape_single_competition, task): task
            for task in competition_tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            competition, sport, _ = task
            try:
                future.result()  # Get result (or raise exception)
                print(f"✅ Task '{sport}/{competition}' thread completed.")
            except Exception as e:
                print(
                    f"❌ Task '{sport}/{competition}' encountered a fatal error: {e}"
                )

    print("\n" + "=" * 50)
    print("✅ All stats scraping tasks completed.")