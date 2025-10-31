import json
import os
import shutil
import sqlite3
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

load_dotenv()

# Configuration (environment-friendly, similar style to VectorStoreManager)
RAW_STATS_DIR = Path(os.getenv("RAW_STATS_DATA_DIR", "data/raw/stats"))
DB_DIR = Path(os.getenv("DB_DIR", "data/db"))
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = DB_DIR / os.getenv("STATS_DB_FILE", "sports_stats.sqlite")
PROCESSED_FILES_LOG = DB_DIR / "processed_stats_files.log"

# Numeric fields mapping (Greek header -> canonical column)
NUM_FIELDS = {
    "Πόντοι": "points",
    "Ασίστς": "assists",
    "Σύνολο ριμπάουντ": "rebounds",
    "Λεπτά που παίχτηκαν": "minutes",
}

# Greek month nominative map expected in folder paths (Οκτώβριος etc.)
GREEK_MONTH_NOMINATIVE_TO_NUM = {
    "Ιανουάριος": 1,
    "Φεβρουάριος": 2,
    "Μάρτιος": 3,
    "Απρίλιος": 4,
    "Μάιος": 5,
    "Ιούνιος": 6,
    "Ιούλιος": 7,
    "Αύγουστος": 8,
    "Σεπτέμβριος": 9,
    "Οκτώβριος": 10,
    "Νοέμβριος": 11,
    "Δεκέμβριος": 12,
}


def parse_minutes(value: Optional[str]) -> Optional[float]:
    """Convert 'MM:SS' to minutes as float. Returns None on bad input."""
    if not value:
        return None
    value = str(value).strip()
    if ":" not in value:
        try:
            return float(value)
        except Exception:
            return None
    try:
        m, s = value.split(":")
        return int(m) + int(s) / 60.0
    except Exception:
        return None


def parse_numeric_stats(row: Dict[str, str]) -> Dict[str, Optional[float]]:
    """Extract canonical numeric stats from CSV row using NUM_FIELDS map."""
    out = {"points": None, "assists": None, "rebounds": None, "minutes": None}
    for greek, canonical in NUM_FIELDS.items():
        v = row.get(greek)
        if v is None or str(v).strip() in ("", "-", "—"):
            out[canonical] = None
            continue
        if canonical == "minutes":
            out["minutes"] = parse_minutes(v)
            continue
        try:
            out[canonical] = float(str(v).replace(",", "."))
        except Exception:
            out[canonical] = None
    return out


class DBStoreManager:
    """
    SQLite-backed manager for structured stats.
    """

    def __init__(self, db_path: Path = DB_FILE, raw_dir: Path = RAW_STATS_DIR):
        self.db_path = Path(db_path)
        self.raw_dir = Path(raw_dir)
        self._ensure_dirs()
        print(f"✅ DBStoreManager using DB: {self.db_path}")
        print(f"   Raw stats dir: {self.raw_dir}")
        self._init_db()

    def _ensure_dirs(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- processed files log helpers (mimic vectorstore) ----------
    def _load_processed_files(self) -> Set[Path]:
        if not PROCESSED_FILES_LOG.exists():
            return set()
        with open(PROCESSED_FILES_LOG, "r", encoding="utf-8") as f:
            return {Path(line.strip()) for line in f if line.strip()}

    def _save_processed_files(self, processed: Set[Path]) -> None:
        with open(PROCESSED_FILES_LOG, "w", encoding="utf-8") as f:
            for p in sorted(processed):
                f.write(f"{p}\n")

    # ---------- DB initialization ----------
    def _conn(self):
        # Use detect_types to allow JSON string handling; keep simple sqlite3 usage.
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        q = """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            team_id INTEGER,
            FOREIGN KEY(team_id) REFERENCES teams(id)
        );
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sport TEXT,
            competition TEXT,
            match_date TEXT,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_score INTEGER,
            away_score INTEGER,
            UNIQUE(home_team_id, away_team_id, match_date),
            FOREIGN KEY(home_team_id) REFERENCES teams(id),
            FOREIGN KEY(away_team_id) REFERENCES teams(id)
        );
        CREATE TABLE IF NOT EXISTS match_players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            player_id INTEGER,
            points REAL,
            rebounds REAL,
            assists REAL,
            minutes REAL,
            raw_stats_json TEXT,
            FOREIGN KEY(match_id) REFERENCES matches(id),
            FOREIGN KEY(player_id) REFERENCES players(id)
        );
        CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
        CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
        CREATE INDEX IF NOT EXISTS idx_match_players_player ON match_players(player_id);
        """
        with self._conn() as c:
            c.executescript(q)

    # ---------- helpers for get_or_create ----------
    def _get_or_create_team(self, conn: sqlite3.Connection, name: str) -> int:
        name = name.strip()
        cur = conn.execute("SELECT id FROM teams WHERE name = ?", (name,))
        r = cur.fetchone()
        if r:
            return r[0]
        cur = conn.execute("INSERT INTO teams (name) VALUES (?)", (name,))
        return cur.lastrowid

    def _get_or_create_player(self, conn: sqlite3.Connection, name: str, team_id: Optional[int]) -> int:
        name = name.strip()
        # Prefer exact match with team_id if available
        if team_id is not None:
            cur = conn.execute("SELECT id FROM players WHERE name = ? AND team_id = ?", (name, team_id))
            r = cur.fetchone()
            if r:
                return r[0]
        else:
            cur = conn.execute("SELECT id FROM players WHERE name = ?", (name,))
            r = cur.fetchone()
            if r:
                return r[0]
        cur = conn.execute("INSERT INTO players (name, team_id) VALUES (?, ?)", (name, team_id))
        return cur.lastrowid

    # ---------- parse helpers for file path / filename ----------
    def _parse_date_from_path(self, file_path: Path) -> str:
        """
        Expected folder structure example:
          .../stats/basketball/euroleague/2025/Οκτώβριος/8/file.csv
        Returns ISO date string 'YYYY-MM-DD'.
        """
        parts = file_path.parts
        # Safely find year/month/day from path end; tolerate slightly different depths.
        # Look for a 4-digit year in the last 6 parts.
        last_six = parts[-6:]
        year = None
        for p in last_six:
            if p.isdigit() and len(p) == 4:
                year = int(p)
                break
        if not year:
            # fallback to file mtime
            return datetime.fromtimestamp(file_path.stat().st_mtime).date().isoformat()
        # find index of year inside parts
        idx = parts.index(str(year))
        try:
            month_name = parts[idx + 1]
            day_part = parts[idx + 2]
            month_num = GREEK_MONTH_NOMINATIVE_TO_NUM.get(month_name)
            day = int(day_part)
            return datetime(year, month_num, day).date().isoformat()
        except Exception:
            return datetime.fromtimestamp(file_path.stat().st_mtime).date().isoformat()

    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """
        Parse filename like:
        "Χάποελ Τελ-Αβίβ-Μακάμπι Τελ Αβίβ~~~90-103.csv"
        Returns dict with home_team, away_team, home_score, away_score.
        """
        stem = Path(filename).stem
        if "~~~" not in stem:
            return None
        try:
            teams_part, score_part = stem.split("~~~", 1)
            # Use rsplit to tolerate internal hyphens in names; split by last '-' between teams
            parts = teams_part.rsplit("-", 1)
            if len(parts) == 2:
                home_team = parts[0].strip()
                away_team = parts[1].strip()
            else:
                # fallback: split on ' - ' or first hyphen
                home_team, away_team = teams_part.split("-", 1)
                home_team = home_team.strip()
                away_team = away_team.strip()
            home_score_str, away_score_str = score_part.split("-", 1)
            return {
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(home_score_str),
                "away_score": int(away_score_str),
            }
        except Exception:
            return None

    # ---------- ingestion pipeline ----------
    def _ingest_file(self, file_path: Path, sport: str, competition: str, conn: sqlite3.Connection) -> bool:
        """Ingest a single CSV file into the DB. Returns True on success."""
        file_meta = self._parse_filename(file_path.name)
        if not file_meta:
            print(f"    ⚠️ Skipping file with unexpected name format: {file_path}")
            return False

        match_date = self._parse_date_from_path(file_path)
        home_name = file_meta["home_team"]
        away_name = file_meta["away_team"]
        home_score = file_meta["home_score"]
        away_score = file_meta["away_score"]

        # get/create teams
        home_id = self._get_or_create_team(conn, home_name)
        away_id = self._get_or_create_team(conn, away_name)

        # insert match (guard against unique constraint)
        cur = conn.execute(
            "SELECT id FROM matches WHERE home_team_id=? AND away_team_id=? AND match_date=?",
            (home_id, away_id, match_date),
        )
        row = cur.fetchone()
        if row:
            match_id = row[0]
        else:
            cur = conn.execute(
                "INSERT INTO matches (sport, competition, match_date, home_team_id, away_team_id, home_score, away_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sport, competition, match_date, home_id, away_id, home_score, away_score),
            )
            match_id = cur.lastrowid

        # read CSV and upsert players + match_players
        inserted_any = False
        with file_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_name = (row.get("Παίκτης") or row.get("Όνομα") or "").strip()
                team_name = (row.get("Ομάδα") or "").strip()
                if not player_name:
                    continue

                # team may be empty for some rows — attempt to fallback to home/away by heuristic
                if not team_name:
                    # If player's team substring appears in home/away name, pick that
                    if home_name in player_name or home_name.split()[0] in player_name:
                        team_name = home_name
                    elif away_name in player_name or away_name.split()[0] in player_name:
                        team_name = away_name

                team_id = self._get_or_create_team(conn, team_name) if team_name else None
                player_id = self._get_or_create_player(conn, player_name, team_id)

                numeric = parse_numeric_stats(row)
                raw_json = json.dumps(row, ensure_ascii=False)

                conn.execute(
                    """INSERT INTO match_players
                       (match_id, player_id, points, rebounds, assists, minutes, raw_stats_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        match_id,
                        player_id,
                        numeric.get("points"),
                        numeric.get("rebounds"),
                        numeric.get("assists"),
                        numeric.get("minutes"),
                        raw_json,
                    ),
                )
                inserted_any = True

        return inserted_any

    def _load_and_ingest(self, days_back: int = 30) -> int:
        """
        Walk RAW_STATS_DIR, find new CSVs modified within days_back,
        ingest them and update processed_files log. Returns number of files ingested.
        """
        processed = self._load_processed_files()
        cutoff = datetime.now() - timedelta(days=days_back)
        all_csvs = list(self.raw_dir.rglob("*.csv"))
        candidates = [p for p in all_csvs if datetime.fromtimestamp(p.stat().st_mtime) >= cutoff and p not in processed]

        if not candidates:
            return 0

        ingested_count = 0
        with self._conn() as conn:
            for p in candidates:
                # Determine sport & competition from path: .../stats/{sport}/{competition}/...
                parts = p.parts
                # best-effort: find 'stats' in path and take next parts
                sport = competition = "unknown"
                if "stats" in parts:
                    idx = parts.index("stats")
                    try:
                        sport = parts[idx + 1]
                        competition = parts[idx + 2]
                    except Exception:
                        pass
                try:
                    ok = self._ingest_file(p, sport, competition, conn)
                    if ok:
                        processed.add(p)
                        ingested_count += 1
                        print(f"   ✅ Ingested: {p}")
                    else:
                        print(f"   ⚠️ No rows inserted for: {p}")
                except Exception as e:
                    print(f"   ❌ Error ingesting {p}: {e}")
                    # continue with next file
        # update processed files log
        self._save_processed_files(processed)
        return ingested_count

    # ---------- public API ----------
    def create_or_update(self, days_back: int = 30) -> None:
        """
        Ingest new files changed within `days_back` days.
        Mirrors VectorStoreManager.create_or_update semantics.
        """
        if not self.raw_dir.exists():
            print(f"❌ Raw stats dir not found: {self.raw_dir}")
            return

        print(f"🔍 Scanning for new CSV stats in last {days_back} days...")
        n = self._load_and_ingest(days_back=days_back)
        if n == 0:
            print("✅ DBStore is already up to date.")
        else:
            print(f"✅ Ingested {n} new files.")

    def load(self) -> None:
        """Compatibility no-op (keeps similar API to VectorStoreManager)."""
        if self.db_path.exists():
            print(f"ℹ️ DB exists at {self.db_path}.")
        else:
            print("ℹ️ No DB file found; run create_or_update() to create one.")

    # ---------- query helpers (useful for LLM / Validator) ----------
    def recent_player_stats(self, player_name: str, limit: int = 5) -> List[Dict]:
        q = """
        SELECT m.match_date, t.name AS team, mp.points, mp.minutes, mp.rebounds, mp.assists
        FROM match_players mp
        JOIN players p ON p.id = mp.player_id
        JOIN matches m ON m.id = mp.match_id
        LEFT JOIN teams t ON p.team_id = t.id
        WHERE p.name = ?
        ORDER BY m.match_date DESC
        LIMIT ?
        """
        with self._conn() as conn:
            rows = conn.execute(q, (player_name, limit)).fetchall()
        return [
            {"date": r[0], "team": r[1], "points": r[2], "minutes": r[3], "rebounds": r[4], "assists": r[5]}
            for r in rows
        ]

    def player_aggregates(self, player_name: str, days: int = 14) -> Dict:
        q = """
        SELECT AVG(mp.points), AVG(mp.minutes), COUNT(mp.id)
        FROM match_players mp
        JOIN players p ON p.id = mp.player_id
        JOIN matches m ON m.id = mp.match_id
        WHERE p.name = ? AND m.match_date >= date('now', ?)
        """
        with self._conn() as conn:
            row = conn.execute(q, (player_name, f"-{days} days")).fetchone()
        return {"avg_points": row[0], "avg_minutes": row[1], "games": row[2]}

    def matches_for_team(self, team_name: str, limit: int = 50) -> List[Dict]:
        q = """
        SELECT m.id, m.match_date, th.name AS home_team, ta.name AS away_team, m.home_score, m.away_score, m.sport, m.competition
        FROM matches m
        JOIN teams th ON m.home_team_id = th.id
        JOIN teams ta ON m.away_team_id = ta.id
        WHERE th.name = ? OR ta.name = ?
        ORDER BY m.match_date DESC
        LIMIT ?
        """
        with self._conn() as conn:
            rows = conn.execute(q, (team_name, team_name, limit)).fetchall()
        return [
            {
                "match_id": r[0],
                "date": r[1],
                "home_team": r[2],
                "away_team": r[3],
                "home_score": r[4],
                "away_score": r[5],
                "sport": r[6],
                "competition": r[7],
            }
            for r in rows
        ]

    def player_stats_for_match(self, match_id: int) -> List[Dict]:
        q = """
        SELECT p.name, t.name, mp.points, mp.minutes, mp.rebounds, mp.assists, mp.raw_stats_json
        FROM match_players mp
        JOIN players p ON p.id = mp.player_id
        LEFT JOIN teams t ON p.team_id = t.id
        WHERE mp.match_id = ?
        """
        with self._conn() as conn:
            rows = conn.execute(q, (match_id,)).fetchall()
        return [
            {
                "player": r[0],
                "team": r[1],
                "points": r[2],
                "minutes": r[3],
                "rebounds": r[4],
                "assists": r[5],
                "raw": json.loads(r[6]) if r[6] else None,
            }
            for r in rows
        ]

    # ---------- validator helper: find match by teams/date fuzzy ----------
    def find_match(self, home_team: Optional[str], away_team: Optional[str], match_date: Optional[str]) -> Optional[Dict]:
        """
        Attempt to find a match given home/away names and/or ISO date.
        If exact match isn't found, try loose matching by LIKE.
        """
        clauses = []
        params = []
        if match_date:
            clauses.append("m.match_date = ?")
            params.append(match_date)
        if home_team:
            clauses.append("th.name LIKE ?")
            params.append(f"%{home_team}%")
        if away_team:
            clauses.append("ta.name LIKE ?")
            params.append(f"%{away_team}%")
        where = " AND ".join(clauses) if clauses else "1=1"
        q = f"""
        SELECT m.id, m.match_date, th.name, ta.name, m.home_score, m.away_score
        FROM matches m
        JOIN teams th ON m.home_team_id = th.id
        JOIN teams ta ON m.away_team_id = ta.id
        WHERE {where}
        ORDER BY m.match_date DESC
        LIMIT 1
        """
        with self._conn() as conn:
            row = conn.execute(q, params).fetchone()
        if not row:
            return None
        return {
            "match_id": row[0],
            "date": row[1],
            "home_team": row[2],
            "away_team": row[3],
            "home_score": row[4],
            "away_score": row[5],
        }

    # ---------- maintenance ----------
    def clear(self) -> None:
        """Delete DB file and processed files log (like VectorStoreManager.clear)."""
        if self.db_path.exists():
            try:
                self.db_path.unlink()
                print(f"🗑️ Deleted DB file: {self.db_path}")
            except Exception as e:
                print(f"❌ Could not delete DB file: {e}")
        if PROCESSED_FILES_LOG.exists():
            try:
                PROCESSED_FILES_LOG.unlink()
                print(f"🗑️ Deleted processed files log: {PROCESSED_FILES_LOG}")
            except Exception as e:
                print(f"❌ Could not delete processed files log: {e}")
        # Recreate empty DB
        self._init_db()
        print("✨ Cleared DBStore and reinitialized schema.")


# --------------------------
# Example CLI usage
# --------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Manage SQLite stats DB (db_store).")
    p.add_argument("--rebuild", action="store_true", help="Delete DB and processed log, reinit schema.")
    p.add_argument("--days", type=int, default=30, help="Days back to ingest (create_or_update).")
    args = p.parse_args()

    mgr = DBStoreManager()
    if args.rebuild:
        mgr.clear()

    # ingest
    mgr.create_or_update(days_back=args.days)

    # example query
    print("\n--- Example: recent stats for 'Nunn' ---")
    print(mgr.recent_player_stats("Nunn", limit=5))

    print("\n--- Example: matches for team 'MAC' ---")
    print(mgr.matches_for_team("MAC", limit=5))
