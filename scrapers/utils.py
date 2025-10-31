import re
from datetime import datetime

# --- Greek Month Maps ---
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

GREEK_MONTH_NOMINATIVE_MAP = {
    "Ιανουαρίου": "Ιανουάριος",
    "Φεβρουαρίου": "Φεβρουάριος",
    "Μαρτίου": "Μάρτιος",
    "Απριλίου": "Απρίλιος",
    "Μαΐου": "Μάιος",
    "Ιουνίου": "Ιούνιος",
    "Ιουλίου": "Ιούλιος",
    "Αυγούστου": "Αύγουστος",
    "Σεπτεμβρίου": "Σεπτέμβριος",
    "Οκτωβρίου": "Οκτώβριος",
    "Νοεμβρίου": "Νοέμβριος",
    "Δεκεμβρίου": "Δεκέμβριος",
}


# --- Date Helpers ---
def normalize_and_format_date_to_greek(date_string: str) -> str:
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


def get_date_path_from_greek_date(date_string: str) -> str:
    day, month, year = date_string.split(" ")
    return f"{year}/{GREEK_MONTH_NOMINATIVE_MAP[month]}/{day}"



def clean_html_text(element):
    if not element:
        return ""
    for tag in element.find_all(["strong", "em", "a", "span", "p", "div", "br"]):
        if tag.next_sibling and not str(tag.next_sibling).startswith(" "):
            tag.insert_after(" ")
    return " ".join(element.get_text(separator=" ", strip=True).split())
