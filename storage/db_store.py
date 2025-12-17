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

    def _get_basketball_key_players(
        self, team_id: int, limit: int
    ) -> List[Dict]:
        """Helper to get top basketball players by points."""
        with self.SessionLocal() as session:
            players = session.query(Player).filter(Player.team_id == team_id).all()
            player_stats = []

            for player in players:
                stats = (
                    session.query(BasketballPlayerPerGame)
                    .filter(BasketballPlayerPerGame.player_id == player.id)
                    .first()
                )
                if stats and stats.points:
                    player_stats.append(
                        {
                            "name": player.name,
                            "points": stats.points or 0,
                            "rebounds": stats.rebounds or 0,
                            "assists": stats.assists or 0,
                            "steals": stats.steals or 0,
                        }
                    )

            player_stats.sort(key=lambda x: x["points"], reverse=True)
            return player_stats[:limit]

    def _get_football_key_players(
        self, team_id: int, limit: int
    ) -> List[Dict]:
        """Helper to get top football players by rating."""
        with self.SessionLocal() as session:
            players = session.query(Player).filter(Player.team_id == team_id).all()
            player_stats = []

            for player in players:
                stats = (
                    session.query(FootballPlayerPerGame)
                    .filter(FootballPlayerPerGame.player_id == player.id)
                    .first()
                )
                if stats and stats.rating:
                    player_stats.append(
                        {
                            "name": player.name,
                            "rating": stats.rating or 0,
                            "shots": stats.shots or 0,
                            "xg": stats.xg or 0,
                            "duels": stats.duels or 0,
                        }
                    )

            player_stats.sort(key=lambda x: x["rating"], reverse=True)
            return player_stats[:limit]

    def get_team_key_players(self, team_name: str, limit: int = 5) -> str:
        """
        Retrieves the top scorers/contributors from a team based on season averages.
        Returns player names with their key stats.
        """
        with self.SessionLocal() as session:
            # 1. Find the Team
            team = session.query(Team).filter(Team.name.like(f"%{team_name}%")).first()
            if not team:
                return f"Could not find a team matching '{team_name}'."

            sport_name = team.sport.name.lower() if team.sport else "unknown"

            # 2. Get all players from the team
            players = session.query(Player).filter(Player.team_id == team.id).all()
            if not players:
                return f"No players found for team '{team.name}'."

            results = []

            if "basketball" in sport_name:
                player_stats = self._get_basketball_key_players(team.id, limit)
                results = [f"🏀 **Top Players for {team.name}:**"]
                for i, p in enumerate(player_stats, 1):
                    results.append(
                        f"{i}. **{p['name']}** - Πόντοι: {p['points']}, Ριμπάουντ: {p['rebounds']}, Ασίστ: {p['assists']}"
                    )

            elif "football" in sport_name:
                player_stats = self._get_football_key_players(team.id, limit)
                results = [f"⚽ **Top Players for {team.name}:**"]
                for i, p in enumerate(player_stats, 1):
                    results.append(
                        f"{i}. **{p['name']}** - Βαθμολογία: {p['rating']}, Σουτ: {p['shots']}, xG: {p['xg']}"
                    )
            else:
                return f"Sport '{sport_name}' is not supported for key players analysis."

            if not results or len(results) == 1:
                return f"No player statistics available for {team.name}."

            return "\n".join(results)

    def get_upcoming_matches(self, team1_name: str, team2_name: str) -> str:
        """
        Searches for upcoming matches between two teams in the database.
        Returns match details if found.
        """
        with self.SessionLocal() as session:
            # 1. Find both teams
            team1 = session.query(Team).filter(Team.name.like(f"%{team1_name}%")).first()
            team2 = session.query(Team).filter(Team.name.like(f"%{team2_name}%")).first()

            if not team1:
                return f"Could not find team '{team1_name}'."
            if not team2:
                return f"Could not find team '{team2_name}'."

            # 2. Search for matches between these teams (in both directions)
            matches = (
                session.query(Match)
                .filter(
                    (
                        (Match.home_team_id == team1.id)
                        & (Match.away_team_id == team2.id)
                    )
                    | (
                        (Match.home_team_id == team2.id)
                        & (Match.away_team_id == team1.id)
                    )
                )
                .order_by(Match.date.desc())
                .limit(10)
                .all()
            )

            if not matches:
                return f"No matches found between {team1.name} and {team2.name}."

            results = [
                f"📋 **Matches between {team1.name} and {team2.name}:**"
            ]
            for match in matches:
                home_team = (
                    session.query(Team).filter(Team.id == match.home_team_id).first()
                )
                away_team = (
                    session.query(Team).filter(Team.id == match.away_team_id).first()
                )
                results.append(
                    f" - {match.date.strftime('%Y-%m-%d')}: {home_team.name} vs {away_team.name} ({match.home_score}-{match.away_score})"
                )

            return "\n".join(results)

    def _get_h2h_basketball_stats(
        self, team1_id: int, team2_id: int, team1_name: str, team2_name: str
    ) -> List[str]:
        """Helper to get basketball head-to-head player stats."""
        with self.SessionLocal() as session:
            results = []

            # Get top scorers from each team
            team1_players = session.query(Player).filter(
                Player.team_id == team1_id
            ).all()
            team2_players = session.query(Player).filter(
                Player.team_id == team2_id
            ).all()

            team1_stats = []
            for player in team1_players:
                stats = (
                    session.query(BasketballPlayerPerGame)
                    .filter(BasketballPlayerPerGame.player_id == player.id)
                    .first()
                )
                if stats and stats.points:
                    team1_stats.append((player.name, stats))

            team1_stats.sort(key=lambda x: x[1].points or 0, reverse=True)

            team2_stats = []
            for player in team2_players:
                stats = (
                    session.query(BasketballPlayerPerGame)
                    .filter(BasketballPlayerPerGame.player_id == player.id)
                    .first()
                )
                if stats and stats.points:
                    team2_stats.append((player.name, stats))

            team2_stats.sort(key=lambda x: x[1].points or 0, reverse=True)

            results.append(f"**{team1_name} - Top Scorers:**")
            for i, (name, stats) in enumerate(team1_stats[:3], 1):
                results.append(
                    f"  {i}. {name}: {stats.points} PPG, {stats.assists} APG, {stats.rebounds} RPG"
                )

            results.append(f"\n**{team2_name} - Top Scorers:**")
            for i, (name, stats) in enumerate(team2_stats[:3], 1):
                results.append(
                    f"  {i}. {name}: {stats.points} PPG, {stats.assists} APG, {stats.rebounds} RPG"
                )

            return results

    def _get_h2h_football_stats(
        self, team1_id: int, team2_id: int, team1_name: str, team2_name: str
    ) -> List[str]:
        """Helper to get football head-to-head player stats."""
        with self.SessionLocal() as session:
            results = []

            # Get top rated players from each team
            team1_players = session.query(Player).filter(
                Player.team_id == team1_id
            ).all()
            team2_players = session.query(Player).filter(
                Player.team_id == team2_id
            ).all()

            team1_stats = []
            for player in team1_players:
                stats = (
                    session.query(FootballPlayerPerGame)
                    .filter(FootballPlayerPerGame.player_id == player.id)
                    .first()
                )
                if stats and stats.rating:
                    team1_stats.append((player.name, stats))

            team1_stats.sort(key=lambda x: x[1].rating or 0, reverse=True)

            team2_stats = []
            for player in team2_players:
                stats = (
                    session.query(FootballPlayerPerGame)
                    .filter(FootballPlayerPerGame.player_id == player.id)
                    .first()
                )
                if stats and stats.rating:
                    team2_stats.append((player.name, stats))

            team2_stats.sort(key=lambda x: x[1].rating or 0, reverse=True)

            results.append(f"**{team1_name} - Top Rated Players:**")
            for i, (name, stats) in enumerate(team1_stats[:3], 1):
                results.append(
                    f"  {i}. {name}: Rating {stats.rating}, {stats.shots} shots, {stats.xg} xG"
                )

            results.append(f"\n**{team2_name} - Top Rated Players:**")
            for i, (name, stats) in enumerate(team2_stats[:3], 1):
                results.append(
                    f"  {i}. {name}: Rating {stats.rating}, {stats.shots} shots, {stats.xg} xG"
                )

            return results

    def get_head_to_head_player_stats(
        self, team1_name: str, team2_name: str, sport: str = "basketball"
    ) -> str:
        """
        Compares key players from two opposing teams to help predict key players for upcoming match.
        Returns comparison of top players from each team.
        """
        with self.SessionLocal() as session:
            # 1. Find both teams
            team1 = session.query(Team).filter(Team.name.like(f"%{team1_name}%")).first()
            team2 = session.query(Team).filter(Team.name.like(f"%{team2_name}%")).first()

            if not team1:
                return f"Could not find team '{team1_name}'."
            if not team2:
                return f"Could not find team '{team2_name}'."

            sport_name = team1.sport.name.lower() if team1.sport else sport.lower()

            results = [
                f"⚔️ **Key Players Comparison: {team1.name} vs {team2.name}**\n"
            ]

            if "basketball" in sport_name:
                h2h_results = self._get_h2h_basketball_stats(
                    team1.id, team2.id, team1.name, team2.name
                )
                results.extend(h2h_results)

            elif "football" in sport_name:
                h2h_results = self._get_h2h_football_stats(
                    team1.id, team2.id, team1.name, team2.name
                )
                results.extend(h2h_results)

            else:
                return f"Sport '{sport_name}' is not supported for comparison."

            return "\n".join(results)


if __name__ == "__main__":
    store = DBStore()
    store.run()
