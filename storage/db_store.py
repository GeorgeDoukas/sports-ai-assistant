import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from storage.db_ingest import build_aggregates, ingest_files
from storage.db_models import (
    Base,
    BasketballPlayerPerGame,
    BasketballStats,
    FootballPlayerPerGame,
    FootballStats,
    Match,
    Player,
    Team,
)

# ===========================================================
# Load environment & Configuration
# ===========================================================
load_dotenv()

DB_DIR = Path(os.getenv("DB_DIR", "data/storage/db"))
DB_PATH = DB_DIR / "stats.db"
PROCESSED_STATS_FILES_LOG = DB_DIR / "processed_stats_files.log"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class DBStore:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    def init_db(self):
        Base.metadata.create_all(self.engine)

    def run(self):
        self.init_db()
        with self.SessionLocal() as session:
            ingest_files(session)
            session.commit()
            build_aggregates(session)
            session.commit()
            print("Done.")

    def get_players_by_surname(self, surname: str) -> List[Dict[str, str]]:
        """
        Searches for players based on a (potentially ambiguous) surname or partial name.
        Returns a list of players, their teams, and their sport.
        """
        with self.SessionLocal() as session:
            search_term = f"%{surname}%"
            players = (
                session.query(Player)
                .join(Team)
                .filter(Player.name.like(search_term))
                .limit(10)  # Limit for performance
                .all()
            )

            results = []
            for player in players:
                sport_name = (
                    player.team.sport.name
                    if player.team and player.team.sport
                    else "Unknown"
                )
                team_name = player.team.name if player.team else "Unknown"

                results.append(
                    {"full_name": player.name, "team": team_name, "sport": sport_name}
                )

            return results

    def get_team_last_matches(self, team_name: str, limit: int = 5) -> str:
        """Retrieves the score and opponents for a team's last X matches."""
        with self.SessionLocal() as session:
            # 1. Find the Team
            team = session.query(Team).filter(Team.name.like(f"%{team_name}%")).first()
            if not team:
                return f"Could not find a team matching '{team_name}'."

            # 2. Query the Match history, ordered by date descending
            matches = (
                session.query(Match)
                .filter(
                    (Match.home_team_id == team.id) | (Match.away_team_id == team.id)
                )
                .order_by(Match.date.desc())
                .limit(limit)
                .all()
            )

            if not matches:
                return f"Found team '{team.name}', but no match history is available."

            # 3. Format the results
            results = [
                f"**Last {len(matches)} matches for {team.name} ({team.sport.name}):**"
            ]
            for match in matches:
                # Fetch the names of the opposing teams
                home_team = (
                    session.query(Team).filter(Team.id == match.home_team_id).first()
                )
                away_team = (
                    session.query(Team).filter(Team.id == match.away_team_id).first()
                )

                home_name = home_team.name if home_team else "Unknown Home"
                away_name = away_team.name if away_team else "Unknown Away"

                # Determine the score/opponent from the perspective of the queried team
                if match.home_team_id == team.id:
                    score = f"{match.home_score} - {match.away_score}"
                    opponent_name = away_name
                    result = (
                        "Νίκη (W)"
                        if match.home_score > match.away_score
                        else (
                            "Ήττα (L)"
                            if match.home_score < match.away_score
                            else "Ισοπαλία (D)"
                        )
                    )
                else:
                    score = f"{match.away_score} - {match.home_score}"  # Flip score to be from team's perspective
                    opponent_name = home_name
                    result = (
                        "Νίκη (W)"
                        if match.away_score > match.home_score
                        else (
                            "Ήττα (L)"
                            if match.away_score < match.home_score
                            else "Ισοπαλία (D)"
                        )
                    )

                results.append(
                    f" - {match.date.strftime('%Y-%m-%d')} vs {opponent_name}: **{result} {score}**"
                )

            return "\n".join(results)

    def get_player_last_games(self, player_name: str, limit: int = 5) -> str:
        """Retrieves a player's individual stats for their last X matches."""
        with self.SessionLocal() as session:
            # 1. Find the Player and Sport
            player = (
                session.query(Player)
                .filter(Player.name.like(f"%{player_name}%"))
                .first()
            )
            if not player:
                return f"Could not find a player matching '{player_name}'."

            sport_name = (
                player.team.sport.name.lower()
                if player.team and player.team.sport
                else "unknown"
            )

            results = [
                f"**{player.name}'s individual performance in the last {limit} games ({sport_name.capitalize()}):**"
            ]

            if "basketball" in sport_name:
                stats_data = (
                    session.query(BasketballStats, Match)
                    .join(Match, BasketballStats.match_id == Match.id)
                    .filter(BasketballStats.player_id == player.id)
                    .order_by(Match.date.desc())
                    .limit(limit)
                    .all()
                )
                for stats, match in stats_data:
                    is_home = match.home_team_id == player.team_id
                    opponent_id = match.away_team_id if is_home else match.home_team_id
                    opponent = (
                        session.query(Team).filter(Team.id == opponent_id).first()
                    )
                    results.append(
                        f" - {match.date.strftime('%Y-%m-%d')} vs {opponent.name}: "
                        f"**Πόντοι**: {stats.points or 'N/A'}, **Ριμπ**: {stats.rebounds_total or 'N/A'}, **Ασιστ**: {stats.assists or 'N/A'}, **Λεπτά**: {stats.minutes or 'N/A'}"
                    )

            elif "football" in sport_name:
                stats_data = (
                    session.query(FootballStats, Match)
                    .join(Match, FootballStats.match_id == Match.id)
                    .filter(FootballStats.player_id == player.id)
                    .order_by(Match.date.desc())
                    .limit(limit)
                    .all()
                )
                for stats, match in stats_data:
                    is_home = match.home_team_id == player.team_id
                    opponent_id = match.away_team_id if is_home else match.home_team_id
                    opponent = (
                        session.query(Team).filter(Team.id == opponent_id).first()
                    )
                    results.append(
                        f" - {match.date.strftime('%Y-%m-%d')} vs {opponent.name}: "
                        f"**Βαθμολ.** (Rating): {stats.rating or 'N/A'}, **Σουτ** (Shots): {stats.shots or 'N/A'}, **xG**: {stats.xg or 'N/A'}"
                    )
            else:
                return f"Individual game analysis is not supported for the sport '{sport_name}'."

            return "\n".join(results)

    def get_player_averages(self, player_name: str, metric: str = "all") -> str:
        with self.SessionLocal() as session:
            player = (
                session.query(Player)
                .filter(Player.name.like(f"%{player_name}%"))
                .first()
            )

            if not player:
                return f"Could not find a player matching '{player_name}' for average stats."

            sport_name = (
                player.team.sport.name.lower()
                if player.team and player.team.sport
                else "unknown"
            )

            # Basketball Averages Logic
            if "basketball" in sport_name:
                stats = (
                    session.query(BasketballPlayerPerGame)
                    .filter(BasketballPlayerPerGame.player_id == player.id)
                    .first()
                )
                if not stats:
                    return f"No basketball averages found for {player.name}."

                metric_map = {
                    "points": stats.points,
                    "ποντοι": stats.points,
                    "rebounds": stats.rebounds,
                    "ριμπαουντ": stats.rebounds,
                    "assists": stats.assists,
                    "ασιστς": stats.assists,
                    "steals": stats.steals,
                    "κλεψιματα": stats.steals,
                }

                if metric.lower() == "all":
                    return f"🏀 **{player.name}** Μέσος Όρος (Averages): Πόντοι: {stats.points}, Ριμπάουντ: {stats.rebounds}, Ασίστ: {stats.assists}, Κλεψίματα: {stats.steals}"
                elif metric.lower() in metric_map:
                    return f"🏀 **{player.name}** {metric.capitalize()} ανά παιχνίδι: {metric_map[metric.lower()] or 'N/A'}"

            # Football Averages Logic
            elif "football" in sport_name:
                stats = (
                    session.query(FootballPlayerPerGame)
                    .filter(FootballPlayerPerGame.player_id == player.id)
                    .first()
                )
                if not stats:
                    return f"No football averages found for {player.name}."

                metric_map = {
                    "rating": stats.rating,
                    "βαθμολογια": stats.rating,
                    "shots": stats.shots,
                    "σουτ": stats.shots,
                    "xg": stats.xg,
                    "duels": stats.duels,
                    "μονομαχιες": stats.duels,
                }

                if metric.lower() == "all":
                    return f"⚽ **{player.name}** Μέσος Όρος (Averages): Βαθμολογία (Rating): {stats.rating}, Σουτ (Shots): {stats.shots}, xG: {stats.xg}"
                elif metric.lower() in metric_map:
                    return f"⚽ **{player.name}** {metric.capitalize()} ανά παιχνίδι: {metric_map[metric.lower()] or 'N/A'}"

            return f"Averages are not supported for the sport '{sport_name}'."


if __name__ == "__main__":
    store = DBStore()
    store.run()
