import csv
import datetime
import pathlib
import re
from pathlib import Path

from db_models import (
    BasketballStats,
    Competition,
    FootballStats,
    Match,
    Player,
    Sport,
    Team,
)
from sqlalchemy import select, text

RAW_DIR = Path("data/raw/stats")
LOG_FILE = Path("data/db/processed_stats_files.log")

_score = re.compile(r"~~~(\d+)-(\d+)\.csv$")
_months = {
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


def _float(v):
    try:
        return float(v.replace(",", ".")) if v not in ("", None) else None
    except:
        return None


def _int(v):
    try:
        return int(float(v)) if v not in ("", None) else 0
    except:
        return 0


def _team_from_row(row, home, away):
    name = row.get("Ομάδα") or row.get("Team")
    return home if name.strip() == home.name else away


def _get_or_create(session, model, defaults=None, **kwargs):
    instance = session.scalar(select(model).filter_by(**kwargs))
    if instance:
        return instance
    params = {**kwargs, **(defaults or {})}
    instance = model(**params)
    session.add(instance)
    session.flush()
    return instance


def ingest_files(session):
    processed = set()
    if LOG_FILE.exists():
        processed = set(LOG_FILE.read_text(encoding="utf-8").splitlines())

    for csv_file in RAW_DIR.rglob("*.csv"):
        fpath = str(csv_file)

        if fpath in processed:
            continue
        print(f"Processing: {csv_file}")
        _process_file(session, csv_file)
        processed.add(fpath)
        LOG_FILE.write_text("\n".join(processed), encoding="utf-8")
        print(f"Completed: {csv_file}")


def _process_file(session, file: pathlib.Path):
    parts = file.parts
    sport, comp, year, month, day = parts[-6:-1]
    year, day = int(year), int(day)
    month = _months[month]

    m = _score.search(file.name)
    home_score, away_score = int(m.group(1)), int(m.group(2))

    t = file.stem.split("~~~")[0]
    home_team, away_team = t.split(" vs ")

    sport_obj = _get_or_create(session, Sport, name=sport)
    comp_obj = _get_or_create(session, Competition, name=comp, sport_id=sport_obj.id)

    home = _get_or_create(
        session, Team, name=home_team, sport_id=sport_obj.id, competition_id=comp_obj.id
    )
    away = _get_or_create(
        session, Team, name=away_team, sport_id=sport_obj.id, competition_id=comp_obj.id
    )

    match_date = datetime.date(year, month, day)
    match = _get_or_create(
        session,
        Match,
        date=match_date,
        sport_id=sport_obj.id,
        competition_id=comp_obj.id,
        home_team_id=home.id,
        away_team_id=away.id,
        home_score=home_score,
        away_score=away_score,
    )

    with open(file, encoding="utf-8") as f:
        rows = csv.DictReader(f)
        for row in rows:
            name = list(row.values())[0].strip()
            team = _team_from_row(row, home, away)
            player = _get_or_create(session, Player, name=name, team_id=team.id)

            if sport == "football":
                _store_football(session, match, player, row)
            else:
                _store_basketball(session, match, player, row)


def _store_football(session, match, player, row):
    session.add(
        FootballStats(
            match_id=match.id,
            player_id=player.id,
            rating=_float(row.get("Αξιολόγηση παίκτη")),
            shots=_float(row.get("Συνολικά Σουτ")),
            xg=_float(row.get("Αναμενόμενα γκολ (xG)")),
            passes=row.get("Επιτυχημένες Πάσες"),
            touches=_float(row.get("Επαφές με τη μπάλα")),
            touches_box=_float(row.get("Επαφές με μπάλα στην αντίπαλη περιοχή")),
            dribbles=row.get("Επιτυχημένες ντρίμπλες"),
            duels=_float(row.get("Προσωπικές μονομαχίες")),
            position=row.get("Θέση"),
        )
    )


def _store_basketball(session, match, player, row):
    session.add(
        BasketballStats(
            match_id=match.id,
            player_id=player.id,
            points=_float(row.get("Πόντοι")),
            rebounds_total=_float(row.get("Σύνολο ριμπάουντ")),
            assists=_float(row.get("Ασίστς")),
            minutes=_float(row.get("Λεπτά που παίχτηκαν")),
            fg_made=_float(row.get("Ευστοχα σουτ εντός πεδιάς")),
            fg_attempts=_float(row.get("Σουτ εντός πεδιάς")),
            two_made=_float(row.get("Ευστοχα σουτ 2π εντός πεδιάς")),
            two_attempts=_float(row.get("Σουτ 2π εντός πεδιάς")),
            three_made=_float(row.get("Ευστοχα σουτ 3π εντός πεδιάς")),
            three_attempts=_float(row.get("Σουτ 3π εντός πεδιάς")),
            ft_made=_float(row.get("Εύστοχες ελεύθερες βολές")),
            ft_attempts=_float(row.get("Ελεύθερες βολές")),
            plus_minus=_int(row.get("+/- Πόντοι") or 0),
            off_rebounds=_float(row.get("Επιθετικά ριμπάουντ")),
            def_rebounds=_float(row.get("Αμυντικά ριμπάουντ")),
            fouls=_float(row.get("Προσωπικά φάουλ")),
            steals=_float(row.get("Κλεψίματα")),
            turnovers=_float(row.get("Λάθη")),
            blocks=_float(row.get("Μπλοκς")),
            blocks_against=_float(row.get("Μπλοκς κατά")),
            tech_fouls=_float(row.get("Τεχνικές Ποινές")),
        )
    )

def build_aggregates(session):
    print("Building aggregates...")

    # Football totals
    session.execute(text("""
        INSERT OR REPLACE INTO football_player_totals
        (player_id, games, rating, shots, xg, touches, touches_box, duels)
        SELECT player_id,
               COUNT(*),
               AVG(rating),
               SUM(shots),
               SUM(xg),
               SUM(touches),
               SUM(touches_box),
               SUM(duels)
        FROM football_stats
        GROUP BY player_id
    """))

    # Football per game
    session.execute(text("""
        INSERT OR REPLACE INTO football_player_pergame
        (player_id, games, rating, shots, xg, touches, touches_box, duels)
        SELECT player_id,
               COUNT(*),
               AVG(rating),
               AVG(shots),
               AVG(xg),
               AVG(touches),
               AVG(touches_box),
               AVG(duels)
        FROM football_stats
        GROUP BY player_id
    """))

    # Basketball totals
    session.execute(text("""
        INSERT OR REPLACE INTO basketball_player_totals
        (player_id, games, points, rebounds, assists, steals, blocks, turnovers, minutes)
        SELECT player_id,
               COUNT(*),
               SUM(points),
               SUM(rebounds_total),
               SUM(assists),
               SUM(steals),
               SUM(blocks),
               SUM(turnovers),
               SUM(minutes)
        FROM basketball_stats
        GROUP BY player_id
    """))

    # Basketball per game
    session.execute(text("""
        INSERT OR REPLACE INTO basketball_player_pergame
        (player_id, games, points, rebounds, assists, steals, blocks, turnovers, minutes)
        SELECT player_id,
               COUNT(*),
               AVG(points),
               AVG(rebounds_total),
               AVG(assists),
               AVG(steals),
               AVG(blocks),
               AVG(turnovers),
               AVG(minutes)
        FROM basketball_stats
        GROUP BY player_id
    """))

    print("Aggregates built.")
